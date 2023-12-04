import torch
import argparse
import os
import sys
import time
from glob import glob 
from loguru import logger
import numpy as np
from scipy.stats import mode
from data_prep.dataset import Dataset 
from data_prep.dataset_loader import LoadData

# importing utilities
from utils.utils import seeding
from networks.NetworkController import getNetwork
from networks.VGG16 import VGG16_BN_Attention
from experiments.ClassifierController import getExperiment
from sklearn.model_selection import StratifiedKFold

from utils.loggers import log_to_file

# Custom log format
fmt = "{message}"
config = {
    "handlers": [
        {"sink": sys.stderr, "format": fmt},
    ],
}
logger.configure(**config)

# importing experiments
# from experiments.ClassifierExperiment import ClassifierExperiment

if __name__ == "__main__":
    # optional arguments from the command line 
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', type=str, default='datasets/train', help='root dir for training data')
    parser.add_argument('--valid_path', type=str, default='datasets/val', help='root dir for validation data')
    parser.add_argument('--output', type=str, default='outputs', help="output dir for saving results")
    parser.add_argument('--experiment_name', type=str, default='exp0001', help='experiment name')
    parser.add_argument('--network_name', type=str, default='DenseNet', help='network name')
    parser.add_argument('--max_epochs', type=int, default=50, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size per gpu')
    parser.add_argument('--base_lr', type=float, default=0.0001, help='network learning rate')
    parser.add_argument('--patience', type=int, default=5, help='patience for lr and early stopping scheduler')
    parser.add_argument('--img_size', type=int, default=224, help='input image size of network input')
    parser.add_argument('--seed', type=int, default=42, help='random seed value')
    parser.add_argument('--verbose', type=int, default=1, help='verbose value [0:2]')
    parser.add_argument("--normalize_attn", action='store_true', help='if True, attention map is normalized by softmax; otherwise use sigmoid. This is only for certain networks.')
    parser.add_argument('--num_folds', type=int, default=5, help='number of folds for cross-validation. This will build a model for each fold.')

    # get cmd args from the parser 
    args = parser.parse_args()
    logger.info(f"Excuting training pipeline with {args.max_epochs} epochs and {args.num_folds} folds.")

    # set paths and dirs
    args.exp = args.experiment_name + '_' + str(args.img_size)
    output_path = os.path.join(os.getcwd(), args.output, "{}".format(args.exp))      # 'outputs/exp0001_224'
    snapshot_path = output_path + '_epo' + str(args.max_epochs)
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr)
    snapshot_path = snapshot_path + '_s' + str(args.seed)               # 'outputs/exp0001_224_epo50_bs32_lr0.0001_s42'
    
    checkpoint_file = args.exp + '_' + args.network_name + '_epo' + str(args.max_epochs)
    checkpoint_file = checkpoint_file + '_bs' + str(args.batch_size)
    checkpoint_file = checkpoint_file + '_lr' + str(args.base_lr)
    checkpoint_file = checkpoint_file + '_seed' + str(args.seed)

    output_path        = os.path.join(snapshot_path, f'{time.strftime("%Y-%m-%d_%H%M", time.gmtime())}_{args.network_name}')

    # set seed value
    seeding(args.seed)

    # load the data from the disk
    # ch1 -> class_labels = {'nevus': 0, 'others': 1}
    # ch2 -> class_labels = {'mel': 0, 'bcc': 1, 'scc': 2}
    _, train_images, train_labels, n_classes = LoadData(
        dataset_path= args.train_path, 
        class_labels = {'nevus': 0, 'others': 1})
    
    _, val_images, val_labels, n_classes = LoadData(
        dataset_path= args.valid_path, 
        class_labels = {'nevus': 0, 'others': 1})
    
    # Use StratifiedKFold for cross-validation
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)

    # Concatenate the training and validation datasets
    all_images = train_images + val_images
    all_labels = train_labels + val_labels

    # Create a new instance of the experiment
    experiment = getExperiment(args.experiment_name)
    network = getNetwork(args.network_name)

    # args.normalize_attn is only possible when the network is VGG16_BN_Attention
    assert not (args.normalize_attn and network != VGG16_BN_Attention), "normalize_att is expected to be used with args.network_name='VGG16_BN_Attention' only."

    # Initialize lists to store predictions from each fold
    all_fold_metrics = []
    all_val_targets = []

    # Iterate over folds
    for fold, (train_index, val_index) in enumerate(skf.split(all_images, all_labels)):
        # Create dataset and loaders for current fold
        fold_train_dataset = Dataset(
            images_path=[all_images[i] for i in train_index], 
            labels=[all_labels[i] for i in train_index], 
            transform=True, 
            split="train",
            input_size=(args.img_size,args.img_size))
        
        fold_val_dataset = Dataset(
            images_path=[all_images[i] for i in val_index], 
            labels=[all_labels[i] for i in val_index], 
            transform=True, 
            split="val", 
            input_size=(args.img_size,args.img_size))
        
        fold_train_loader = torch.utils.data.DataLoader(
            fold_train_dataset, batch_size=args.batch_size, shuffle=True)
        
        fold_val_loader = torch.utils.data.DataLoader(
            fold_val_dataset, batch_size=args.batch_size, shuffle=True)

        # Add fold number to the checkpoint file name
        fold_ckp_file = checkpoint_file + f'_fold{fold+1}' + '.pth'

        # Create a new directory for the current fold
        fold_path          = os.path.join(output_path, f'fold_{fold+1}')
        
        # Initialize the experiment 
        exp = experiment(
            args, 
            fold_train_loader, 
            fold_val_loader, 
            n_classes, 
            checkpoint_file = fold_ckp_file,
            Network = network,
            fold_path = fold_path, 
            fold_num = fold+1,
            c_weights = fold_train_dataset.get_class_weight())

        # run training and validation for current fold
        exp.run()
        
        # run the val/test report generation
        metrics_dict = exp.run_test(test_loader=fold_val_loader, save_path=fold_path)

        # append the metrics to the list
        all_fold_metrics.append(metrics_dict)
    
    # compute the average metrics across all folds 'accuracy', 'auc', 'kappa'
    all_fold_metrics = np.array(all_fold_metrics)
    avg_accuracy = np.mean([fmet['accuracy'] for fmet in all_fold_metrics])
    avg_auc = np.mean([fmet['auc'] for fmet in all_fold_metrics])
    avg_kappa = np.mean([fmet['kappa'] for fmet in all_fold_metrics])

    logger.info(f"Average accuracy: {avg_accuracy:.5f}. Average AUC: {avg_auc:.5f}. Average Kappa: {avg_kappa:.5f}")




