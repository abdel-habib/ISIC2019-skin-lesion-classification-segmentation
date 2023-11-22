import torch
import argparse
import os
from glob import glob 
from loguru import logger
from data_prep.dataset import Dataset 
from data_prep.dataset_loader import LoadData

# importing utilities
from utils.utils import seeding

# importing experiments
from experiments.ClassifierExperiment import ClassifierExperiment

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

    # get cmd args from the parser 
    args = parser.parse_args()
    logger.info(f"Excuting training pipeline with {args.max_epochs} epochs...")

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
    checkpoint_file = checkpoint_file + '.pth'

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
    
    # create a dataset object with the loaded data
    train_dataset = Dataset(
        images_path=train_images, labels=train_labels, transform=True, split="train",
        input_size=(args.img_size,args.img_size)
        )
    
    val_dataset = Dataset(
        images_path=val_images, labels=val_labels, transform=True, split="val", 
        input_size=(args.img_size,args.img_size)
        )
    
    # create train and val data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True)

    # creating an instance of a classifier experiment
    exp = ClassifierExperiment(
        args, 
        train_loader, 
        val_loader, 
        n_classes, 
        checkpoint_file,
        output_path = snapshot_path, 
        c_weights = train_dataset.get_class_weight())

    # run training
    exp.run()

