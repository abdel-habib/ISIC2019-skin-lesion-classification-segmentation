import sys
sys.path.append('./')

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

from utils.preprocessor import denormalize, min_max_normalization

def display_PIL_image(image_array):
    """
    Display an image using Pillow.

    Args:
        image_array ('np.array'): 
            NumPy array representing the image.
    """
    # Convert the NumPy array to a Pillow image
    image = Image.fromarray(np.uint8(image_array))

    # Display the image
    image.show()

def display_CV2_image(img):
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def visualize_batch(images, labels, dataset_type, mask=False):
    """
        Visualize a batch of images with their labels.

        Args:
            images (list of numpy.ndarray): List of images in CHW format.
            labels (list): List of labels corresponding to the images.
            dataset_type (str): Type of dataset (e.g., 'train', 'val', 'test').

        Returns:
            None
    """
    print(f"Visualizing a batch from '{dataset_type}' dataset type.\n")
    # get the batch size (batch_size, channels, height, width)
    batch_size = len(images)
    
    figure = plt.figure(figsize=(batch_size, batch_size))

    # iterate on the batch images
    for i in range(0, batch_size):
        image = images[i]
        if not mask:
            # get the image and transpose it to (H, W, C)
            image = np.transpose(image, (1,2,0))

            # denormalize from the augmentation
            image = denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            # standerdize to [0, 255]
            image = min_max_normalization(image, 255).astype('uint8')
            
        # Display the image
        subplot = figure.add_subplot(8, batch_size//4, i + 1)
        subplot.axis('off')
        subplot.set_title(labels[i].item())
        plt.imshow(image, cmap='gray' if mask else None)

    plt.show()
    
