"""
This module represents an image classifier experiment and contains a class that handles
the experiment lifecycle
"""
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")


import torch
import time
import os
import numpy as np
from loguru import logger
from pathlib import Path

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils.utils import create_directory_if_not_exists, epoch_time
from utils.loggers import log_to_file

from callbacks.early_stopping import EarlyStopper

from torchsummary import summary

class ClassifierExperiment:
    """
    This class implements the basic life cycle for a classification task with different classifier architectures.
    The basic life cycle of a ClassifierExperiment is:

        run():
            for epoch in max_epochs:
                train()
                validate()
        test()
    """
    def __init__(self, args, train_loader, val_loader, n_classes, checkpoint_file, Network, output_path, c_weights=None):
        self.max_epochs         = args.max_epochs   # max epochs to iterate
        self.epoch              = 0                 # current epoch
        self._time_start        = ""
        self._time_end          = ""
        self.train_loader       = train_loader
        self.val_loader         = val_loader
        self.n_classes          = n_classes
        self.output_path        = os.path.join(output_path, f'{time.strftime("%Y-%m-%d_%H%M", time.gmtime())}_{args.network_name}')
        self.checkpoint_file    = os.path.join(self.output_path, checkpoint_file)
        self.logs_path          = os.path.join(self.output_path, 'experiment_logs.txt')
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
        create_directory_if_not_exists(self.output_path)

        # configure the model
        # self.model = DenseNetMel(num_classes=self.n_classes)
        # self.model = ResNetMel(num_classes=self.n_classes)
        self.model = Network(num_classes=self.n_classes)

        # self.model = ResNetMel(num_classes=self.n_classes, fine_tune = True, num_layers_to_unfreeze=2)
        # self.model = EfficientNetMel(num_classes=self.n_classes, weights_b=1, fine_tune = True, num_layers_to_unfreeze=20)
        # self.model = VGG16Mel(num_classes=self.n_classes)
        self.model.to(self.device)

        # Print the summary
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
        self.early_stopper = EarlyStopper(patience=args.patience, min_delta=0)

    def train(self):
        """
        This method is executed once per epoch and takes 
        care of model weight update cycle
        """
        if (self.verbose == 2):
            print(f"Training epoch {self.epoch+1}...")
            log_to_file(f"Training epoch {self.epoch+1}...", Path(self.logs_path))

        # Set the model to training mode
        self.model.train()
        epoch_loss = 0.0

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
            epoch_loss += loss.item()

            # log batch details only when verbose == 2
            if (self.verbose == 2) and (((batch_idx + 1) % 10) == 0):
                # Output to console on every 10th batch
                print(f"Epoch: {self.epoch+1} Train batch {batch_idx + 1} loss: {loss}, {100*(batch_idx+1)/len(self.train_loader):.1f}% complete")
                log_to_file(f"Epoch: {self.epoch+1} Train batch {batch_idx + 1} loss: {loss}, {100*(batch_idx+1)/len(self.train_loader):.1f}% complete", Path(self.logs_path))

        # average the losses
        epoch_loss = epoch_loss / len(self.train_loader)
        self.tensorboard_train_writer.add_scalar('Average Epoch Loss', epoch_loss, self.epoch+1)
        
        return epoch_loss

    def validate(self):
        """
        This method runs validation cycle, using same metrics as 
        Train method. Note that model needs to be switched to eval
        mode and no_grad needs to be called so that gradients do not 
        propagate
        """
        if (self.verbose == 2):
            print(f"Validating epoch {self.epoch+1}...")
            log_to_file(f"Validating epoch {self.epoch+1}...", Path(self.logs_path))

        # Set the model to evaluation mode
        self.model.eval()
        loss_list = []
        epoch_loss = 0.0

        # Disable gradient calculation
        with torch.no_grad():
            # Iterate over the validation data
            for batch_idx, (data, target) in enumerate(self.val_loader):
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                output = self.model(data)

                # Compute the loss
                loss = self.loss_function(output, target)

                # log batch details only when verbose == 2
                if (self.verbose == 2):
                    print(f"Batch {batch_idx + 1}. Data shape {data.shape} Loss {loss}")
                    log_to_file(f"Batch {batch_idx + 1}. Data shape {data.shape} Loss {loss}", Path(self.logs_path))

                # Log validation information to Tensorboard
                self.tensorboard_val_writer.add_scalar('Loss', loss.item(), batch_idx + 1)

                # We report loss that is accumulated across all of validation set
                loss_list.append(loss.item())

                # accumulate the epoch losses
                epoch_loss += loss.item()

        # Step the learning rate scheduler based on the validation loss
        self.scheduler.step(np.mean(loss_list))
        
        # average the losses
        epoch_loss = epoch_loss / len(self.val_loader)
        self.tensorboard_val_writer.add_scalar('Average Epoch Loss', epoch_loss, self.epoch+1)
        
        return epoch_loss

    def run_test(self):
        """
        This runs test cycle on the test dataset.
        Note that process and evaluations are quite different
        Here we are computing a lot more metrics and returning
        a dictionary that could later be persisted as JSON
        """
        logger.info("Testing...")


        logger.info("\nTesting complete.")

    def run(self):
        """
        Kicks off train cycle and writes model parameter file at the end
        """

        self._time_start = time.time()
        early_stopped = False  # Flag to track whether early stopping occurred

        logger.info("Experiment started.")

        # Iterate over epochs
        for self.epoch in range(self.max_epochs):
            start_time = time.time()

            train_loss = self.train()
            valid_loss = self.validate()

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            # get the new lr for logging
            after_lr = self.optimizer.param_groups[0]["lr"]

            print(f'Epoch: {self.epoch+1:02}/{self.max_epochs} | epoch time: {epoch_mins}m {epoch_secs:.04}s | lr: {after_lr:.5e} | train/loss: {train_loss:.5f} | val/loss: {valid_loss:.5f}')
            log_to_file(f'Epoch: {self.epoch+1:02}/{self.max_epochs} | epoch time: {epoch_mins}m {epoch_secs:.04}s | lr: {after_lr:.5e} | train/loss: {train_loss:.5f} | val/loss: {valid_loss:.5f}', 
                        Path(self.logs_path))

            # check if validation loss diverges
            if self.early_stopper.early_stop(valid_loss):
                # save model for inferencing
                self.early_stopper.save_checkpoint(self.epoch+1, self.model, valid_loss, self.optimizer, self.checkpoint_file)

                # set the flag to True to indicate early stopping occurred
                early_stopped = True

                # break the training loop
                break
        
        if not early_stopped:
            # Save the model after completing all epochs
            self.early_stopper.save_checkpoint(self.epoch+1, self.model, valid_loss, self.optimizer, self.checkpoint_file)

        self._time_end = time.time()
        logger.info(f"Run complete. Total time: {time.strftime('%H:%M:%S', time.gmtime(self._time_end - self._time_start))}")
