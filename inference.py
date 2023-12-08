import torch
import argparse
import os
import sys
from loguru import logger
from data_prep.dataset import Dataset 
from data_prep.dataset_loader import LoadData
from glob import glob

# importing utilities
from utils.utils import seeding, load_model
from networks.NetworkController import getNetwork
from networks.VGG16 import VGG16_BN_Attention
from experiments.ClassifierController import getExperiment

import torch.nn.functional as F
from tqdm import tqdm

import numpy as np
from scipy.stats import mode

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

    # get cmd args from the parser 
    args = parser.parse_args()
    logger.info(f"Excuting inference script...")

    # set seed value
    seeding(args.seed)

    # get network
    network = getNetwork(args.network_name)

    # construct the experiment output directory
    output_path = \
        f'{args.output}/{args.experiment_name}_{args.img_size}_epo{args.max_epochs}_bs{args.batch_size}_lr{args.base_lr}_s{args.seed}/{args.timeframe}_{args.network_name}'
    logger.info(f"Constructed output path: {output_path}. Searching for models...")

    # get the models
    models_paths = sorted(glob(os.path.join(output_path, "***", "*.pth"), recursive=True))
    
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
        _labels = {'testX': -1}

    logger.info(f"Dataset labels: {_labels} dictionary.")

    # load the data from the disk
    test_dataset_df, test_images, test_labels, n_classes = LoadData(
        dataset_path= args.test_path, 
        class_labels = _labels)
    # print(test_labels)
    
    # create a dataset object with the loaded data
    test_dataset = Dataset(
        images_path=test_images, labels=test_labels, transform=True, split="test",
        input_size=(args.img_size,args.img_size))
    
    # create a dataloader object
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # log the length of the dataset
    logger.info(f"Dataset length: {len(test_dataset_df)}")
    
    # loading the models
    models = [network(num_classes=n_classes) for _ in range(len(models_paths))]
    for i, path in enumerate(models_paths):
        load_model(models[i], path)
    
    # Predictions placeholder for each model
    all_predictions = [[] for _ in range(len(models))]

    # true labels placeholder for the loaded data
    true_labels = [] if args.report else None

    # iterate over the test data
    with torch.no_grad():
        for images, labels in tqdm(test_dataloader, desc="Inference"):
            # get the true labels
            true_labels.extend(labels.numpy()) if args.report else None

            # iterate over the models
            for j, model in enumerate(models):
                # get the predictions
                output = model(images)
                probs = F.softmax(output, dim=1)
                predictions = torch.argmax(probs, dim=1)
                all_predictions[j].extend(predictions.numpy())
    
    # convert the predictions to numpy array
    all_predictions = np.array(all_predictions)
    true_labels = np.array(true_labels) if args.report else None

    # get the majority vote
    majority_vote, _ = mode(all_predictions, axis=0)

    # report the results only if the report flag is True
    if args.report:
        # calculate the metrics
        acc = accuracy_score(true_labels, majority_vote)
        kappa = cohen_kappa_score(true_labels, majority_vote)

        # print the results
        logger.info(f"Majority vote accuracy: {acc}")
        logger.info(f"Majority vote kappa: {kappa}")