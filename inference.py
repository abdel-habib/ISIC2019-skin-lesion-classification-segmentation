
import os
os.environ["CUDA_VISIBLE_DEVICES"] = 'MIG-69a8ded4-a632-5ad1-8445-c2513c997b19' # this is your assigned UUID

import torch
import argparse
import sys
from loguru import logger
from data_prep.dataset import Dataset 
from data_prep.dataset_loader import LoadData
from glob import glob

# importing utilities
from utils.utils import seeding, load_model, pprint_objects
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
    logger.info(f"Found {len(models_paths)} models. Starting loading the models.")

    # load the data
    if not "test" in args.test_path:
        if args.multi:
            logger.info(f"Loading data with 3 class labels from {args.test_path} path...")
            _labels = {'mel': 0, 'bcc': 1, 'scc': 2}
        else:
            logger.info(f"Loading data with 2 class labels from {args.test_path} path...")
            _labels = {'nevus': 0, 'others': 1}
    else:
        logger.info(f"Loading data with unknown class labels from {args.test_path} path...")
        # random labels to test
        # note as we count the labels based on the folder directory, we need to force this when the test data is passed to be based on the length
        # of the labels dictionary
        if args.multi:
            _labels = {'testX': -911, 'testY': -922, 'testZ': -933}
        else:
            _labels = {'testX': -911, 'testY': -922}

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
    
    # loading the models
    models = [network(num_classes=n_classes).to(device) for _ in range(len(models_paths))]
    for i, path in enumerate(models_paths):
        load_model(models[i], path)
    
    # Predictions placeholder for each model
    all_predictions = [[] for _ in range(len(models))]
    all_probabilities = [[] for _ in range(len(models))]

    # true labels placeholder for the loaded data
    true_labels = [] if args.report else None

    # iterate over the test data
    with torch.no_grad():
        for images, labels in tqdm(test_dataloader, desc="Inference"):
            images, labels = images.to(device), labels.to(labels)

            # get the true labels
            true_labels.extend(labels.numpy()) if args.report else None

            # iterate over the models
            for j, model in enumerate(models):
                # get the predictions
                output, _, _ = model(images)
                probs = F.softmax(output, dim=1)
                predictions = torch.argmax(probs, dim=1)

                all_predictions[j].extend(predictions.cpu().numpy())
                all_probabilities[j].extend(probs.cpu().numpy())
                
    # convert the predictions to numpy array
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)

    # convert the true labels to numpy array only for validation where we have the labels
    true_labels = np.array(true_labels) if args.report else None

    # get the majority vote
    majority_vote, _ = mode(all_predictions, axis=0)

    # get the average probabilities
    avg_probabilities = np.mean(all_probabilities, axis=0)

    # export the results into a csv file
    result_df = pd.DataFrame({'Majority_Vote': majority_vote})
    output_csv_path = os.path.join(output_path, f'{args.test_path.split("/")[-1]}_{args.experiment_name}_{args.img_size}_epo{args.max_epochs}_bs{args.batch_size}_lr{args.base_lr}_s{args.seed}_{args.timeframe}_{args.network_name}.csv')
    result_df.to_csv(output_csv_path, index=False, header=False)
    logger.info(f"Results exported to: {output_csv_path}")

    # report the results only if the report flag is True
    if args.report:
        # calculate the metrics
        acc = accuracy_score(true_labels, majority_vote)
        kappa = cohen_kappa_score(true_labels, majority_vote)

        # Compute sensitivity, specificity, precision, and recall for each target class
        precision, recall, fscore, support = precision_recall_fscore_support(true_labels, majority_vote, labels=range(n_classes))

        # Generate a classification report
        class_report = classification_report(true_labels, majority_vote, target_names=[str(i) for i in range(n_classes)])

        # Create a 1x2 subplot for confusion matrix and ROC curve
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Generate confusion matrix and plot it
        conf_matrix = confusion_matrix(true_labels, majority_vote, labels=range(n_classes))
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
            auc_value = roc_auc_score(true_labels, majority_vote)
            # Generate ROC curve and plot it
            fpr, tpr, thresholds = roc_curve(true_labels, majority_vote)
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


