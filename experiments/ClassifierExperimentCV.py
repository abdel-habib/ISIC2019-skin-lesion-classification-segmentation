"""
This module represents an image classifier experiment and contains a class that handles
the experiment lifecycle
"""
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

import sys
import torch
import time
import json
import os
import numpy as np
from loguru import logger
from pathlib import Path

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


from utils.utils import create_directory_if_not_exists, epoch_time
from utils.loggers import log_to_file

from callbacks.early_stopping import EarlyStopper

from torchsummary import summary

# Custom log format
fmt = "{message}"
config = {
    "handlers": [
        {"sink": sys.stderr, "format": fmt},
    ],
}
logger.configure(**config)


class ClassifierExperimentCV:
    """
    This class implements the basic life cycle for a classification task with different classifier architectures.
    The basic life cycle of a ClassifierExperiment is:

        run():
            for epoch in max_epochs:
                train()
                validate()
        test()
    """
    def __init__(self, args, train_loader, val_loader, n_classes, checkpoint_file, Network, fold_path, fold_num, c_weights=None):
        self.max_epochs         = args.max_epochs   # max epochs to iterate
        self.epoch              = 0                 # current epoch
        self.fold               = fold_num          # current fold
        self.max_folds          = args.num_folds    # max folds to iterate
        self._time_start        = ""
        self._time_end          = ""
        self.train_loader       = train_loader
        self.val_loader         = val_loader
        self.n_classes          = n_classes
        self.fold_path          = fold_path
        self.checkpoint_file    = os.path.join(self.fold_path, checkpoint_file)
        self.logs_path          = os.path.join(self.fold_path, 'experiment_logs.txt')
        self.verbose            = args.verbose

        logger.add(self.logs_path, rotation="10 MB", level="INFO")

        logger.info(f"Creating a classifier experiment using {args.network_name} network.")

        # checking if CUDA is available
        if not torch.cuda.is_available():
            logger.warning("WARNING: No CUDA device is found. This may take significantly longer!")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # asssert the verbose level
        # 0: no verbose, 1: epoch level logging, 2: 
        assert self.verbose <= 2, 'Verbose logging can be set to be either 0 (default, no verbose), 1 (epoch level logging), 2 (epoch and patch level logging).'

        # create output folders
        create_directory_if_not_exists(self.fold_path)

        # configure the model
        if args.normalize_attn:
            logger.info(f"Normalizing the attention network.")
            # only for certain networks as VGG16_BN_Attention
            self.model = Network(num_classes=self.n_classes, normalize_attn=args.normalize_attn)
        else:
            self.model = Network(num_classes=self.n_classes)
        self.model.to(self.device)

        # print the summary
        # summary(self.model, input_size=(3, 224, 224))

        # configure the loss function
        if c_weights is not None:
            cWeight=torch.tensor(c_weights).float().to(self.device) # class weights
            loss_fx = torch.nn.CrossEntropyLoss(weight = cWeight)
        else:
            loss_fx = torch.nn.CrossEntropyLoss()

        self.loss_function = loss_fx

        # We are using standard SGD method to optimize our weights
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.base_lr)

        # Scheduler helps us update learning rate automatically
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=args.patience, verbose=True)

        # Set up Tensorboard. By default it saves data into runs folder. You need to launch
        self.tensorboard_train_writer = SummaryWriter(comment="_train")
        self.tensorboard_val_writer = SummaryWriter(comment="_val")

        # set up early stopping callback
        self.early_stopper = EarlyStopper(patience=args.patience, delta=0, trace_func=logger.info, logs_path=self.logs_path, path=self.checkpoint_file)

    def train(self):
        """
        This method is executed once per epoch and takes 
        care of model weight update cycle
        """
        # Set the model to training mode
        self.model.train()

        loss_list = []

        # Placeholder for the actual training logic
        # Iterate over the training data
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Zero the gradients
            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(data)

            # Compute the loss
            loss = self.loss_function(output, target)

            # Backward pass
            loss.backward()

            # Update the weights
            self.optimizer.step()

            # Log training information to Tensorboard
            self.tensorboard_train_writer.add_scalar('Loss', loss.item(), batch_idx + 1)

            # accumulate the epoch losses
            loss_list.append(loss.item())

            # log batch details only when verbose == 2
            message = f"[CV {self.fold}/{self.max_folds}]: Epoch: {self.epoch+1}, train batch {batch_idx + 1}, loss: {loss}, {100*(batch_idx+1)/len(self.train_loader):.1f}% complete"
            should_print = (self.verbose == 2) and (batch_idx + 1) % 10 == 0
            logger.info(message) if should_print else None
            log_to_file(message, Path(self.logs_path)) if self.verbose <= 1 and ((batch_idx + 1) % 10 == 0) else None

        # average the losses
        epoch_loss = np.mean(loss_list)
        self.tensorboard_train_writer.add_scalar('Average Epoch Loss', epoch_loss, self.epoch+1)
        
        return epoch_loss

    def validate(self):
        """
        This method runs validation cycle, using same metrics as 
        Train method. Note that model needs to be switched to eval
        mode and no_grad needs to be called so that gradients do not 
        propagate
        """
        # Set the model to evaluation mode
        self.model.eval()
        loss_list = []
        predictions = []
        targets = []

        # Disable gradient calculation
        with torch.no_grad():
            # Iterate over the validation data
            for batch_idx, (data, target) in enumerate(self.val_loader):
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                output = self.model(data)

                # Compute the loss
                loss = self.loss_function(output, target)

                # Log validation information to Tensorboard
                self.tensorboard_val_writer.add_scalar('Loss', loss.item(), batch_idx + 1)

                # We report loss that is accumulated across all of validation set
                loss_list.append(loss.item())

                # Accumulate predictions and targets for accuracy, AUC, and Kappa calculation
                predictions.extend(torch.argmax(output, dim=1).cpu().numpy())
                targets.extend(target.cpu().numpy())
                
                # log batch details only when verbose == 2
                log = f"[CV {self.fold}/{self.max_folds}]: Epoch: {self.epoch+1}, valid batch {batch_idx + 1}, loss {loss}"
                logger.info(log) if self.verbose == 2 else None
                log_to_file(log, Path(self.logs_path)) if self.verbose <= 1 else None

        # Step the learning rate scheduler based on the validation loss
        self.scheduler.step(np.mean(loss_list))

        # Calculate accuracy and AUC
        accuracy = accuracy_score(targets, predictions)
        auc = roc_auc_score(targets, predictions)
        kappa = cohen_kappa_score(targets, predictions)

        # Log accuracy and AUC
        self.tensorboard_val_writer.add_scalar('Accuracy', accuracy, self.epoch + 1)
        self.tensorboard_val_writer.add_scalar('AUC', auc, self.epoch + 1)
        self.tensorboard_val_writer.add_scalar('Kappa', kappa, self.epoch + 1)

        # average the losses
        epoch_loss = np.mean(loss_list)
        self.tensorboard_val_writer.add_scalar('Average Epoch Loss', epoch_loss, self.epoch+1)
        
        return epoch_loss, accuracy, auc, kappa
    
    def evaluation_metrics(self, all_predictions, all_targets, save_path=None, ensemble=False):
        if ensemble:
            logger.info(f"[CV {self.fold}/{self.max_folds}]: Evaluating ensemble predictions...")

        # Calculate various metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        auc_value = roc_auc_score(all_targets, all_predictions)
        kappa = cohen_kappa_score(all_targets, all_predictions)

        # Compute sensitivity, specificity, precision, and recall for each target class
        precision, recall, fscore, support = precision_recall_fscore_support(all_targets, all_predictions, labels=range(self.n_classes))

        # Generate a classification report
        class_report = classification_report(all_targets, all_predictions, target_names=[str(i) for i in range(self.n_classes)])

        # Create a 1x2 subplot for confusion matrix and ROC curve
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Generate confusion matrix and plot it
        conf_matrix = confusion_matrix(all_targets, all_predictions, labels=range(self.n_classes))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(self.n_classes), yticklabels=range(self.n_classes), ax=axes[0])
        axes[0].set_title('Confusion Matrix')

        # Generate ROC curve and plot it
        fpr, tpr, thresholds = roc_curve(all_targets, all_predictions)
        axes[1].plot(fpr, tpr, label=f'AUC = {auc_value:.2f}')
        axes[1].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend()

        # Compute sensitivity and specificity for each target class
        sensitivity = []
        specificity = []

        for i in range(self.n_classes):
            true_positive = conf_matrix[i, i]
            false_negative = conf_matrix[i, :].sum() - true_positive
            false_positive = conf_matrix[:, i].sum() - true_positive
            true_negative = conf_matrix.sum() - (true_positive + false_negative + false_positive)

            # Sensitivity (Recall)
            sensitivity_class = true_positive / (true_positive + false_negative)
            sensitivity.append(sensitivity_class)

            # Specificity
            specificity_class = true_negative / (true_negative + false_positive)
            specificity.append(specificity_class)

        # Save the subplot figure if save_path is provided
        if save_path:
            fig.savefig(os.path.join(save_path, 'confusion_matrix_and_roc_curve.png'))

        # Log and print metrics
        logger.info(f'[CV {self.fold}/{self.max_folds}]: Test Accuracy: {accuracy:.5f}')
        logger.info(f'[CV {self.fold}/{self.max_folds}]: Test AUC: {auc_value:.5f}')
        logger.info(f'[CV {self.fold}/{self.max_folds}]: Test Kappa: {kappa:.5f}')

        # Log and print sensitivity, specificity, precision, and recall for each target class
        for i in range(self.n_classes):
            logger.info(f'[CV {self.fold}/{self.max_folds}]: Target {i} - Precision: {precision[i]:.5f}, Recall: {recall[i]:.5f}, F-score: {fscore[i]:.5f}, Sensitivity: {sensitivity[i]}, Specificity: {specificity[i]}, Support: {support[i]}')

        # logger.info(f'\n[CV {self.fold}/{self.max_folds}]: Classification Report:\n' + class_report)

        # Convert NumPy arrays to Python lists before saving
        precision_list = precision.tolist()
        recall_list = recall.tolist()
        fscore_list = fscore.tolist()
        support_list = support.tolist()

        # Save metrics dictionary to a text file
        metrics_dict = {
            'accuracy': accuracy,
            'auc': auc_value,
            'kappa': kappa,
            'precision': precision_list,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'recall': recall_list,
            'fscore': fscore_list,
            'support': support_list,
            'classification_report': class_report
        }

        if save_path:
            with open(os.path.join(save_path, 'metrics.txt'), 'w') as file:
                json.dump(metrics_dict, file)

        return metrics_dict

    def run_test(self, test_loader, save_path=None):
        """
        This runs a test cycle on the test dataset.
        Computes various metrics and generates a confusion matrix and ROC curve.
        Optionally, saves the confusion matrix and ROC curve figures.

        Args:
            test_loader: DataLoader for the test dataset.
            save_path (str): Path to save the figures. If None, figures are not saved.

        Returns:
            dict: Dictionary containing various evaluation metrics.

        """
        logger.info(f"[CV {self.fold}/{self.max_folds}]: Testing...")

        # Set the model to evaluation mode
        self.model.eval()

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                output = self.model(data)

                # Accumulate predictions and targets
                predictions = torch.argmax(output, dim=1).cpu().numpy()
                targets = target.cpu().numpy()

                all_predictions.extend(predictions)
                all_targets.extend(targets)

        metrics_dict = self.evaluation_metrics(all_predictions, all_targets, save_path=save_path)

        logger.info(f"[CV {self.fold}/{self.max_folds}]: Testing complete.")

        return metrics_dict

    def load_model_parameters(self, checkpoint_file):
        """
        Loads model parameters (state_dict) from file_path.
        If optimizer is provided, loads state_dict of
        optimizer assuming it is present in checkpoint_file.
        """
        logger.info(f"Loading model parameters from {checkpoint_file}")
        # checkpoint = torch.load(checkpoint_file)
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # self.epoch = checkpoint['epoch']

        # logger.info(f"Loaded model parameters from {checkpoint_file}")

    def run(self):
        """
        Kicks off train cycle and writes model parameter file at the end
        """

        self._time_start = time.time()

        # Iterate over epochs
        for self.epoch in range(self.max_epochs):
            start_time = time.time()

            train_loss = self.train()
            valid_loss, accuracy, auc, kappa = self.validate() 

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            # get the new lr for logging
            after_lr = self.optimizer.param_groups[0]["lr"]

            # handle early stopping and saving model
            self.early_stopper(
                validation_loss=valid_loss, 
                epoch=self.epoch, 
                model=self.model, 
                optimizer=self.optimizer)
        
            logger.info(f'[CV {self.fold}/{self.max_folds}]: Epoch: {self.epoch+1:02}/{self.max_epochs} | epoch time: {epoch_mins}m {epoch_secs:.04}s | lr: {after_lr:.5e} | train/loss: {train_loss:.5f} | val/loss: {valid_loss:.5f} | val/accuracy: {accuracy:.5f} | val/AUC: {auc:.5f} | val/Kappa: {kappa:.5f}')

            if self.early_stopper.early_stop:
                logger.warning(f"[CV {self.fold}/{self.max_folds}]: Early stopping triggered at epoch {self.epoch+1}. Ending model training.")
                break

        self._time_end = time.time()
        logger.info(f"[CV {self.fold}/{self.max_folds}]: Run complete. Total time: {time.strftime('%H:%M:%S', time.gmtime(self._time_end - self._time_start))}")
