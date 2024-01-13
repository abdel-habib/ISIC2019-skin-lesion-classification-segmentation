
import os
os.environ["CUDA_VISIBLE_DEVICES"] = 'MIG-69a8ded4-a632-5ad1-8445-c2513c997b19' # this is your assigned UUID

import torch
import argparse
import sys
from loguru import logger
from data_prep.dataset import Dataset 
from data_prep.dataset_loader import LoadData
from glob import glob
import torchvision.utils as utils
from torchvision import transforms

# importing utilities
from utils.utils import seeding, load_model, pprint_objects
from utils.attention import visualize_attn
from networks.NetworkController import getNetwork
from networks.VGG16 import VGG16_BN_Attention
from experiments.ClassifierController import getExperiment

import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

import numpy as np
from scipy.stats import mode

from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import json
from metrics.utils import compute_multiclass_auc

from sklearn.metrics import accuracy_score, cohen_kappa_score

# Custom log format
fmt = "{message}"
config = {
    "handlers": [
        {"sink": sys.stderr, "format": fmt},
    ],
}
logger.configure(**config)

def calculate_model_accuracies(models, validation_dataloader, device):
    '''
    Calculates the accuracies of the models on the validation set to obtain the weights for Weighted Voting (Soft Voting).

    Args:
        models (list): List of models.
        validation_dataloader (torch.utils.data.DataLoader): Validation set dataloader.
        device (torch.device): Device to run the models on.

    Returns:
        accuracies (list): List of accuracies for each model.
    '''
    accuracies = []

    with torch.no_grad():
        for model in tqdm(models, desc="Calculating Accuracies"):            
            y_true = []
            y_pred = []

            for images, labels in validation_dataloader:
                images, labels = images.to(device), labels.to(device)
                
                output, _, _ = model(images)
                probs = F.softmax(output, dim=1)
                predictions = torch.argmax(probs, dim=1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())

            accuracy = accuracy_score(y_true, y_pred)
            accuracies.append(accuracy)

    return accuracies


if __name__ == "__main__":
    # optional arguments from the command line 
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_path', type=str, default='datasets/test', help='root dir for test data')
    parser.add_argument('--output', type=str, default='outputs', help="output dir for saving results")
    parser.add_argument('--experiment_name', type=str, default='exp0001', help='experiment name')
    parser.add_argument('--network_name', type=str, default='DenseNet', help='network name')
    parser.add_argument('--max_epochs', type=int, default=50, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size per gpu')
    parser.add_argument('--base_lr', type=float, default=0.0001, help='network learning rate')
    parser.add_argument('--img_size', type=int, default=224, help='input image size of network input')
    parser.add_argument('--seed', type=int, default=42, help='random seed value')
    parser.add_argument('--timeframe', type=str, help='date/time for training the model. It is similar in the model folder.')
    parser.add_argument('--verbose', type=int, default=1, help='verbose value [0:2]')
    parser.add_argument("--multi", action='store_true', help='if True, we use the 3 class labels for loading the data.')
    parser.add_argument("--report", action='store_true', help='if True, we evaluate the model on the available labels and save the report.')
    parser.add_argument("--ensemble", action='store_true', help='if True, we search in the folds to evaluate the ensemble.')
    parser.add_argument("--combination_strategy", type=str, default='majority_vote', help='combination strategy for the ensemble [majority_vote, weighted_voting]')
    parser.add_argument("--upscale_factor", type=int, default=8, help='upscale factor for the masks used in the training. Default=8')
    parser.add_argument("--gradcam", action='store_true', help='if True, we save gradcam results for the (first model if ensemble / base model).')

    # get cmd args from the parser 
    args = parser.parse_args()
    logger.info(f"Excuting inference script...")

    # set seed value
    seeding(args.seed)
    print(args)

    # get network
    network = getNetwork(args.network_name)

    # construct the experiment output directory
    output_path = \
        f'{args.output}/{args.experiment_name}_{args.img_size}_epo{args.max_epochs}_bs{args.batch_size}_lr{args.base_lr}_s{args.seed}/{args.timeframe}_{args.network_name}'
    logger.info(f"Constructed output path: {output_path}. Searching for models...")

    # get the models
    models_paths = sorted(glob(os.path.join(output_path, "***" if args.ensemble else '', "*.pth"), recursive=True))
    
    # check if the models are found
    if len(models_paths) == 0:
        logger.info(f"No models found in {output_path}. Exiting...")
        sys.exit(0)
    logger.info(f"Found {len(models_paths)} models.")

    # load the data
    if not "test" in args.test_path:
        if args.multi:
            logger.info(f"Loading data with 3 class labels from {args.test_path} path...")
            _labels = {'mel': 0, 'bcc': 1, 'scc': 2}
            _challenge_type = 'ch2'
        else:
            logger.info(f"Loading data with 2 class labels from {args.test_path} path...")
            _labels = {'nevus': 0, 'others': 1}
            _challenge_type = 'ch1'
    else:
        logger.info(f"Loading data with unknown class labels from {args.test_path} path...")
        # random labels to test
        # note as we count the labels based on the folder directory, we need to force this when the test data is passed to be based on the length
        # of the labels dictionary
        if args.multi:
            _labels = {'testX': -911, 'testY': -922, 'testZ': -933}
            _challenge_type = 'ch2'
        else:
            _labels = {'testX': -911, 'testY': -922}
            _challenge_type = 'ch1'
        

    logger.info(f"Dataset labels: {_labels} dictionary.")

    # load the data from the disk
    # check that the data is loaded by order, as we use sorted() inside here
    test_dataset_df, test_images, _, test_labels, n_classes = LoadData(
        dataset_path= args.test_path, 
        class_labels = _labels)
    # print(test_labels)

    if "test" in args.test_path:
        print("Forcing n_classes to be based on the length of the labels dictionary...")
        print(f"Before: {n_classes}")
        n_classes = len(_labels)
        print(f"After: {n_classes}")
    
    # create a dataset object with the loaded data
    test_dataset = Dataset(
        images_path=test_images, labels=test_labels, transform=True, split="test",
        input_size=(args.img_size,args.img_size))
    
    # create a dataloader object
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # log the length of the dataset
    logger.info(f"Dataset length: {len(test_dataset_df)}")

    # set the correct device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # check the combination strategy, if it is weighted voting, we need to load the validation data to obtain the weights
    if args.combination_strategy == 'weighted_voting':
        logger.info(f"Combination strategy is {args.combination_strategy}. Loading validation data to obtain the weights...")
        # construct the val path
        args.val_path = args.test_path.replace("test", "val")
        logger.info("args.val_path: ", args.val_path)

        if args.multi:
            logger.info(f"Loading data with 3 class labels from {args.val_path} path...")
            _val_labels = {'mel': 0, 'bcc': 1, 'scc': 2}
        else:
            logger.info(f"Loading data with 2 class labels from {args.val_path} path...")
            _val_labels = {'nevus': 0, 'others': 1}
        logger.info(f"Dataset _val_labels: {_val_labels} dictionary.")

        _, val_images, _, val_labels, val_n_classes = LoadData(
            dataset_path= args.val_path, 
            class_labels = _val_labels)

        val_dataset = Dataset(
            images_path=val_images, labels=val_labels, transform=True, split="val",
            input_size=(args.img_size,args.img_size))
        
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        # log the length of the dataset
        logger.info(f"Dataset `val_dataset` length: {len(val_dataset)}")
    
    # loading the models
    models = [network(num_classes=n_classes).to(device) for _ in range(len(models_paths))]
    for i, path in enumerate(models_paths):
        load_model(models[i], path)
    
    # Predictions placeholder for each model
    all_predictions = [[] for _ in range(len(models))]
    all_probabilities = [[] for _ in range(len(models))]

    # true labels placeholder for the loaded data
    true_labels = [] if args.report else None

    # filenames
    all_filenames = []

    # iterate over the test data
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_dataloader, desc="Inference")):
            images, labels = images.to(device), labels.to(labels)

            # get the true labels
            true_labels.extend(labels.numpy()) if args.report else None

            # iterate over the models
            for j, model in enumerate(models):
                # get the predictions
                output, a1, a2 = model(images)
                probs = F.softmax(output, dim=1)
                predictions = torch.argmax(probs, dim=1)

                all_predictions[j].extend(predictions.cpu().numpy())
                all_probabilities[j].extend(probs.cpu().numpy())

                # save the gradcam results for the first model only
                if args.gradcam:
                    I_val = utils.make_grid(images[0:16,:,:,:], nrow=4, normalize=True, scale_each=True)

                    attn1 = visualize_attn(I_val, a1[0:16,:,:,:], up_factor=args.upscale_factor, nrow=4)
                    # attn2 = visualize_attn(I_val, a2[0:16,:,:,:], up_factor=2*args.upscale_factor, nrow=4)

                    # Convert the result tensor to a PIL Image
                    attn1_image = transforms.ToPILImage()(attn1)
                    # attn2_image = transforms.ToPILImage()(attn2)

                    # Save the attention maps
                    att_output_dir = os.path.join(output_path, f'gradcam_{args.test_path.split("/")[-1]}')
                    os.makedirs(att_output_dir, exist_ok=True)

                    attn1_image.save(os.path.join(att_output_dir, f'attn1_model={j}_batch={batch_idx}.png'))
                    # attn2_image.save(os.path.join(att_output_dir, f'attn2_model={j}_batch={batch_idx}.png'))

            # get the images filenames
            images_filenames = test_dataset.images_path[batch_idx*args.batch_size:batch_idx*args.batch_size+args.batch_size]

            # get the filenames of the images from the paths
            images_filenames = [os.path.basename(file_path)  for file_path in images_filenames]

            # extend the filenames
            all_filenames += images_filenames

    # convert the predictions to numpy array
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)

    logger.info(f"all_predictions shape: {all_predictions.shape}")
    logger.info(f"all_probabilities shape: {all_probabilities.shape}")

    # convert the true labels to numpy array only for validation where we have the labels
    true_labels = np.array(true_labels) if args.report else None

    # check the combination strategy
    if args.combination_strategy == 'weighted_voting':
        logger.info(f"Combination strategy is {args.combination_strategy}. Computing the weighted voting...")
        # Calculate model accuracies on the validation set
        validation_accuracies = calculate_model_accuracies(models, val_dataloader, device)
        logger.info(f"Validation accuracies: {validation_accuracies}")

        # Define weights based on validation accuracies
        weights = [accuracy / sum(validation_accuracies) for accuracy in validation_accuracies]
        logger.info(f"Validation weights: {weights}")

        # Calculate weighted voting (soft voting)
        avg_probabilities  = np.average(all_probabilities, axis=0, weights=weights)
        
        # Get the class index with the maximum probability for each sample
        ensemble_pred = np.argmax(avg_probabilities, axis=1)

        # Calculate the weighted average of probabilities
        # avg_probabilities = np.average(all_probabilities, axis=0, weights=weights)

        # key to add into the filename
        strategy_key = 'weighted_voting'

    else:
        logger.info(f"Combination strategy is {args.combination_strategy}. Computing the majority vote...")
        # get the majority vote
        ensemble_pred, _ = mode(all_predictions, axis=0)

        # get the average probabilities
        avg_probabilities = np.mean(all_probabilities, axis=0)
        logger.info(f"Average probabilities: {avg_probabilities}")

        # key to add into the filename
        strategy_key = 'majority_vote'

    # export the results into a csv file
    result_df = pd.DataFrame({'ensemble_pred': ensemble_pred, 'filenames': all_filenames})
    output_csv_path = os.path.join(output_path, f'{_challenge_type}_{strategy_key}_{args.test_path.split("/")[-1]}_{args.experiment_name}_{args.img_size}_epo{args.max_epochs}_bs{args.batch_size}_lr{args.base_lr}_s{args.seed}_{args.timeframe}_{args.network_name}.csv')
    result_df.to_csv(output_csv_path, index=False, header=False)
    logger.info(f"Results exported to: {output_csv_path}")

    # report the results only if the report flag is True
    if args.report:
        # calculate the metrics
        acc = accuracy_score(true_labels, ensemble_pred)
        kappa = cohen_kappa_score(true_labels, ensemble_pred)

        # Compute sensitivity, specificity, precision, and recall for each target class
        precision, recall, fscore, support = precision_recall_fscore_support(true_labels, ensemble_pred, labels=range(n_classes))

        # Generate a classification report
        class_report = classification_report(true_labels, ensemble_pred, target_names=[str(i) for i in range(n_classes)])

        # Create a 1x2 subplot for confusion matrix and ROC curve
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Generate confusion matrix and plot it
        conf_matrix = confusion_matrix(true_labels, ensemble_pred, labels=range(n_classes))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(n_classes), yticklabels=range(n_classes), ax=axes[0])
        axes[0].set_title('Confusion Matrix')

        if args.multi:
            logger.info(f"Computing AUC scores for each class...")

            # Convert lists to numpy arrays
            true_labels = np.array(true_labels)
            all_probabilities = np.array(avg_probabilities)

            auc_scores = compute_multiclass_auc(true_labels, all_probabilities)

            # Generate ROC curve and plot it using One-vs-Rest (OvR) strategy
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(true_labels == i, all_probabilities[:, i])
                auc_value = auc_scores[i]
                axes[1].plot(fpr, tpr, label=f'Class {i} (AUC = {auc_value:.2f}')

            # set the auc value to the mean of all classes auc scores
            auc_value = np.mean(auc_scores)
        else:
            auc_value = roc_auc_score(true_labels, ensemble_pred)
            # Generate ROC curve and plot it
            fpr, tpr, thresholds = roc_curve(true_labels, ensemble_pred)
            axes[1].plot(fpr, tpr, label=f'AUC = {auc_value:.2f}')

        axes[1].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend()

        # Compute sensitivity and specificity for each target class
        sensitivity = []
        specificity = []

        for i in range(n_classes):
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
        fig.savefig(os.path.join(output_path, 'confusion_matrix_and_roc_curve.png'))

        # Display the plots in the notebook
        plt.show()

        # Log and print metrics
        logger.info(f'Majority vote AUC: {auc_value:.5f}')
        logger.info(f'Majority vote AUC Scores: {auc_scores}') if args.multi else None
        logger.info(f"Majority vote accuracy: {acc}")
        logger.info(f"Majority vote kappa: {kappa}")

        # Log and print sensitivity, specificity, precision, and recall for each target class
        for i in range(n_classes):
            logger.info(f'Target {i} - Precision: {precision[i]:.5f}, Recall: {recall[i]:.5f}, F-score: {fscore[i]:.5f}, Sensitivity: {sensitivity[i]}, Specificity: {specificity[i]}, Support: {support[i]}')

        # logger.info(f'\nClassification Report:\n' + class_report)

        # Convert NumPy arrays to Python lists before saving
        precision_list = precision.tolist()
        recall_list = recall.tolist()
        fscore_list = fscore.tolist()
        support_list = support.tolist()

        # Save metrics dictionary to a text file
        metrics_dict = {
            'accuracy': acc,
            'auc': auc_value,
            'kappa': kappa,
            'precision': precision_list,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'recall': recall_list,
            'fscore': fscore_list,
            'support': support_list,
            # 'classification_report': class_report
        }

        if args.multi:
            metrics_dict['auc_scores'] = auc_scores.tolist()

        pprint_objects(metrics_dict)

        with open(os.path.join(output_path, 'metrics.txt'), 'w') as file:
            json.dump(metrics_dict, file)


