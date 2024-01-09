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
from sklearn.utils.class_weight import compute_class_weight

from loguru import logger

# Custom log format
fmt = "{message}"
config = {
    "handlers": [
        {"sink": sys.stderr, "format": fmt},
    ],
}
logger.configure(**config)


class Dataset(data.Dataset):
    """Custom data.Dataset class compatible with data.DataLoader."""
    def __init__(self, 
                 images_path:list,
                 labels: list, 
                 masks_path:list = None,
                 transform:bool = False,
                 split:str = "train",
                 input_size:tuple = (224, 224),
                 ):
        
        self.images_path    = images_path
        self.masks_path      = masks_path
        self.labels         = labels
        self.transform      = transform
        self.split          = split
        self.input_size     = input_size

        # logger.info(f"Creating a Dataset instance for {self.split} split.")

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image = Image.open(self.images_path[index])
        label = self.labels[index]

        mask = None
        if self.masks_path:
            mask = Image.open(self.masks_path[index])
            mask = np.array(mask)
            mask = np.where(mask > 0, 255, 0)
            mask = mask.astype(np.uint8)
            mask = Image.fromarray(mask)

        if self.transform:
            image, mask = self._transform(image, mask)

        # convert to channel-first format (C, H, W) 
        image = image.transpose(2, 0, 1)

        # convert the image to tensor instead of using ToTensorV2
        image = torch.tensor(image).float()
        
        if mask is not None:
            mask = np.array(mask)
            mask = torch.tensor(mask, dtype=torch.float32)
            return image, mask, label
        
        return image, label

    def __len__(self):
        """
        This method is called by PyTorch DataLoader class to return number of samples in the dataset

        Returns:
            len(self.images_path) ('int'): 
                Number of classes found in the dataset directory.
        """
        return len(self.images_path)

    def _transform(self, image, mask=None):
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
        #     transform = A.Compose([
        #         preprocessor.CropFrame(),
        #         preprocessor.AdvancedHairAugmentation(hairs_folder='../datasets/melanoma_hairs/'),
        #         A.HorizontalFlip(),
        #         A.VerticalFlip(),
        #         A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        #         A.Rotate(limit=30),
        #         A.RandomBrightnessContrast(),
        #         A.OneOf([
        #             A.MotionBlur(blur_limit=5),
        #             A.MedianBlur(blur_limit=5),
        #             A.GaussianBlur(blur_limit=5),
        #             A.GaussNoise(var_limit=(5.0, 30.0)),
        #         ], p=0.7),
        #         A.CLAHE(clip_limit=4.0, p=0.7),
        #         A.Transpose(),
        #         A.HueSaturationValue(hue_shift_limit=(10, 10), val_shift_limit=(10, 10), sat_shift_limit=(20, 20)),
        #         A.Resize(self.input_size[0], self.input_size[0]),
        #         A.Normalize(),
        #     ])
        # elif self.split in ["val", "test"]:
        #     transform = A.Compose([
        #         preprocessor.CropFrame(),
        #         A.Resize(self.input_size[0], self.input_size[1]),
        #         A.Normalize(),
        #     ])

        if self.split == "train":
            transform = A.Compose([
                preprocessor.CropFrame(),
                preprocessor.AdvancedHairAugmentation(hairs_folder='../datasets/skin_hairs/'),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                A.Rotate(limit=30),
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
            ], is_check_shapes=False if mask is not None else True)   
            # the masks were already preprocessed using preprocessor.CropFrame() during the segmentation, 
            # making some of the images shapes not equal to the cropped mask shape, thus we set is_check_shapes=False
            # false only for the mask, not the image

        elif self.split in ["val", "test"]:
            transform = A.Compose([
                preprocessor.CropFrame(),
                A.Resize(self.input_size[0], self.input_size[1]),
                A.Normalize(),
            ])

        # Apply transformations
        if mask is not None:
            transformed_image, transformed_mask = transform(image=np.array(image), mask=np.array(mask) if mask is not None else None).values()
            return transformed_image, transformed_mask

        transformed_image = transform(image=np.array(image))['image']

        return transformed_image, mask # mask here is none

    
    def get_class_weight(self):
        class_labels = np.unique(self.labels)
        class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=self.labels)
        return class_weights
        

