from glob import glob 
import os
import numpy as np
import pandas as pd
from loguru import logger
import sys

# Custom log format
fmt = "{message}"
config = {
    "handlers": [
        {"sink": sys.stderr, "format": fmt},
    ],
}
logger.configure(**config)


def LoadData(dataset_path='../datasets/train', class_labels = None):
        """
            Get image paths and corresponding labels from a dataset.

            Args:
                dataset_path ('str'): Path to the dataset directory.
                class_labels ('dict'): Dictionary to map class names to numerical labels.

            Returns:
                tuple: A tuple containing:
                    - DataFrame: Pandas DataFrame with 'Image_Path' and 'Label' columns.
                    - list: List of image paths.
                    - list: List of corresponding labels.
        """
        if class_labels is None:
            raise ValueError("Missing class_labels dictionary")
        
        logger.info(f"Loading the data from {dataset_path}")
            
        # Get paths of all images in the train directory
        train_images_paths = sorted(glob(os.path.join(os.getcwd(), dataset_path, '*', '*.jpg')))

        # Lists to store image paths and corresponding labels
        images = []
        labels = []

        # Iterate over the image paths
        for image_path in train_images_paths:
            # Extract class name from the directory
            class_name = os.path.basename(os.path.dirname(image_path))
            
            # Assign the label based on the class folder name
            label = class_labels[class_name]
            
            # Append the image path and its corresponding label to the lists
            images.append(image_path)
            labels.append(label)

        n_classes = len(np.unique(labels))

        # Create a DataFrame
        dataset_df = pd.DataFrame({
            'Image_Path': images,
            'Label': labels
        })

        return dataset_df, images, labels, n_classes