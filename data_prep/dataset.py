import sys
sys.path.append('../')

import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import albumentations as A
import numpy as np
import utils.preprocessor as preprocessor
import pandas as pd

from loguru import logger


class Dataset(data.Dataset):
    """Custom data.Dataset class compatible with data.DataLoader."""
    def __init__(self, 
                 images_path:list,
                 labels: list, 
                 transform:bool = False,
                 split:str = "train",
                 input_size:tuple = (224, 224)
                 ):
        
        self.images_path    = images_path
        self.labels         = labels
        self.transform      = transform
        self.split          = split
        self.input_size     = input_size

        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std  = [0.229, 0.224, 0.225]

        logger.info(f"Creating a Dataset instance for {self.split} split.")

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image = Image.open(self.images_path[index])
        label = self.labels[index]

        if self.transform:
            image = self._transform(image)

        # convert to channel-first format (C, H, W) 
        image = image.transpose(2, 0, 1)

        # convert the image to tensor instead of using ToTensorV2
        image = torch.tensor(image).float()
        
        return image, label

    def __len__(self):
        """
        This method is called by PyTorch DataLoader class to return number of samples in the dataset

        Returns:
            len(self.images_path) ('int'): 
                Number of classes found in the dataset directory.
        """
        return len(self.images_path)

    def _transform(self, image):
        '''If you have N original training images and you use a batch size of B, you will process 
        N/B batches during each training epoch. For each batch, you apply the specified data 
        augmentation transforms to B images. Therefore, you effectively generate N/B * B 
        augmented images in each epoch.

        For example, if you have 1,000 original training images and use a batch size of 32, 
        you'll generate 1,000 / 32 * 32 = 1,000 augmented images during each training epoch.

        Note that these augmented images are generated on-the-fly during training and are not 
        saved as separate image files. The data augmentation occurs during the forward pass of 
        the training loop, providing a diverse set of training examples for the model to learn from.
        '''

        # if self.split == "train":
        #     transform = transforms.Compose([
        #         transforms.RandomResizedCrop(size=self.input_size[0], scale=(0.7, 1.0)),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.RandomVerticalFlip(),
        #         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        #         transforms.RandomRotation(degrees=30),
        #         transforms.RandomCrop(size=self.input_size, padding=4),
        #         transforms.GaussianBlur(kernel_size=3),
        #         transforms.ToTensor(), # uses channel-first format (C, H, W)
        #         transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std),
        #     ])
        
        # elif self.split in ["val", "test"]:
        #     transform = transforms.Compose([
        #         transforms.Resize(size=self.input_size),
        #         transforms.ToTensor(), # uses channel-first format (C, H, W)
        #         transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std),
        #     ])

        # image = transform(image)

        if self.split == "train":
            transform = A.Compose([
                preprocessor.CropFrame(),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                A.Rotate(limit=30),
                # A.RandomCrop(self.input_size[0], self.input_size[1], p=0.5),  # Added probability for RandomCrop
                A.RandomBrightnessContrast(),
                A.OneOf([
                    A.MotionBlur(blur_limit=5),
                    A.MedianBlur(blur_limit=5),
                    A.GaussianBlur(blur_limit=5),
                    A.GaussNoise(var_limit=(5.0, 30.0)),
                ], p=0.7),
                A.CLAHE(clip_limit=4.0, p=0.7),
                A.Transpose(),
                A.HueSaturationValue(hue_shift_limit=(10, 10), val_shift_limit=(10, 10), sat_shift_limit=(20, 20)),
                A.Resize(self.input_size[0], self.input_size[0]),
                A.Normalize(),
            ])
        elif self.split in ["val", "test"]:
            transform = A.Compose([
                preprocessor.CropFrame(),
                A.Resize(self.input_size[0], self.input_size[1]),
                A.Normalize(),
            ])

        # Convert PIL image to numpy array
        image_np = np.array(image)
        
        # Apply transformations
        transformed = transform(image=image_np)
        
        # Extract the transformed image
        image_transformed = transformed['image']

        return image_transformed
    
    def get_class_weight(self):
        n_classes = len(np.unique(self.labels))
        value_count = pd.DataFrame(self.labels).value_counts()

        w0 = value_count[0] / len(self.labels)
        w1 = value_count[1] / len(self.labels)
        w2 = (value_count[2] / len(self.labels)) if n_classes > 2 else None     

        if n_classes > 2:
            return w0, w1, w2
        
        return w0, w1
        

