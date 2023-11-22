import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2
import numpy as np
import copy

class Resize(A.ImageOnlyTransform):
    def __init__(self, new_size=(227, 227), preserve_ratio=False, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.new_size = new_size
        self.preserve_ratio = preserve_ratio

    def apply(self, img, **params):
        return resize_images(img, self.new_size, self.preserve_ratio)

class ExtractHair(A.ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        return extract_hair(img)

class CropFrame(A.ImageOnlyTransform):
    def __init__(self, threshold=0.2, debug=False, margin=0.31, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.threshold = threshold
        self.debug = debug
        self.margin = margin

    def apply(self, img, **params):
        return crop_frame(img, self.threshold, self.debug, self.margin)[0]

class MinMaxNormalization(A.ImageOnlyTransform):
    def __init__(self, max_value, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.max_value = max_value

    def apply(self, img, **params):
        return min_max_normalization(img, self.max_value)



def resize_images(image, new_size=(227, 227), preserve_ratio = False):
        '''Resizing function to handle image sizes from different datasets. It can resize to a fixed 
        ratio, or preserve the ratio as HAM10000 dataset aspect ratio.
        '''
        height, width       = image.shape[:2]

        # print(f"i/p image shape (h, w, c): {image.shape}")

        # Option to preserve the ratio of all images, so resizing all images' longer 
        # side to 600 pixels while preserving the aspect ratio.
        if(preserve_ratio):
            # if HAM10000, don't resize
            if (height, width) == (450, 600):
                return image        
            
            # HAM10000 aspect ratio is 600 / 450 = 1.33333333 
            image_aspect_ratio = float(600) / 450

            # print(f"Resizing as HAM10000 ratio with the longest side = 600 : {image_aspect_ratio}")

            if width > height:
                new_width = 600
                new_height = int(new_width / image_aspect_ratio)
            else:
                new_height = 600
                new_width = int(new_height * image_aspect_ratio)

            resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        else:
            # For consistency, all the images are resized to 227×227×3.
            resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)

        return resized_image

def extract_hair(img):

    # Convert RGB to grayscale
    img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clip_limit = 1.0 # 10.0
    tile_size = 10 # 6
    CLAHE = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size,tile_size))
    img_CLAHE = CLAHE.apply(img_grayscale)

    # Apply Gaussian filter
    filter_size = 5
    filtered_image = cv2.GaussianBlur(img_CLAHE, (filter_size, filter_size), 0)

    # Canny edge detection
    # Calculate the median pixel value of the image
    median_value = np.median(filtered_image)
    # Define high and low thresholds for Canny
    high_threshold = median_value + 75
    low_threshold = median_value - 75
    # Apply Canny edge detection with the selected thresholds
    edges = cv2.Canny(filtered_image, low_threshold, high_threshold)

    # Dilation
    # Define the size of the dilation kernel (structuring element)
    kernel_size = (9, 9)  # Adjust the size as needed
    # Perform dilation
    dilated_image = cv2.dilate(edges, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size))

    inpainted_img = cv2.inpaint(img, dilated_image, inpaintRadius=5, flags=cv2.INPAINT_TELEA) # INPAINT_NS

    return inpainted_img  
    
def crop_frame(image, threshold = 0.2, debug=False, margin=0.31):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        
    # Gaussian blur and binarize
    blurred = cv2.GaussianBlur(gray, (0, 0), 2)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find connected components and their properties
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Cropping flag for identifying cropped images
    CROP_FLAG = False

    if contours:        
        # Calculate the center and diameter of the largest component
        largest_contour = max(contours, key=cv2.contourArea)

        # finds a circle of the minimum area enclosing an area
        (x , y), radius = cv2.minEnclosingCircle(largest_contour) # returns the center and radius

        # casting from float to int for precision, necessary for visualizing 
        center = (int(x), int(y)) #(x, y)
        radius = int(radius)

        if debug:
            # for debugging we can visualize the contour of the minimum enclosing circle using its radius
            print(center, radius)
            cv2.circle(image, center, 5, (255, 0, 0), -1)   # viz the center as a dot (small radius)
            cv2.circle(image, center, radius, (255,0,0), 3) # viz the center as a crcle
        
        # Define the cropping box
        x_min = int(center[0] - radius)
        x_max = int(center[0] + radius)
        y_min = int(center[1] - radius)
        y_max = int(center[1] + radius)

        # check if we need to crop the images based on their intensities mean
        cropped_image = image.copy()[y_min:y_max, x_min:x_max]

        # Calculate mean values inside and outside the cropping box
        mean_inside = np.mean(cropped_image)

        # Calculate mean values outside the cropping box
        mean_above = np.mean(image[:y_min, :])
        mean_below = np.mean(image[y_max:, :])
        mean_left = np.mean(image[y_min:y_max, :x_min])
        mean_right = np.mean(image[y_min:y_max, x_max:])

        # Calculate mean_outside as the average of the four regions
        mean_outside = (mean_above + mean_below + mean_left + mean_right) / 4

        if mean_outside / mean_inside < threshold:
            # Define the cropping box with the required margin (TO EXCLUDE THE MARGIN FROM AFFECTING THIS VALIDATION)
            ret =  image.copy()[
                int(center[1] - radius + margin * radius):int(center[1] + radius - margin * radius), 
                int(center[0] - radius + margin * radius):int(center[0] + radius - margin * radius)]
            CROP_FLAG = True
        else: 
            # print("Doesn't required cropping")
            ret = image
    else:
        # print("No contours found, doesn't required cropping")
        ret = image

    return ret, CROP_FLAG

def min_max_normalization(image, max_value):
    # Ensure the image is a NumPy array for efficient calculations
    image = np.array(image)
    
    # Calculate the minimum and maximum pixel values
    min_value = np.min(image)
    max_actual = np.max(image)
    
    # Perform min-max normalization
    normalized_image = (image - min_value) / (max_actual - min_value) * max_value
    
    return normalized_image


def denormalize(array, mean, std):
    """
    Denormalize a NumPy array by reversing the normalization process.

    Args:
        array (numpy.ndarray): Input array.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.

    Returns:
        numpy.ndarray: Denormalized array.
    """
    # Copy the array to avoid modifying the original
    denormalized_array = copy.deepcopy(array)
    
    for i in range(len(mean)):
        denormalized_array[..., i] = denormalized_array[..., i] * std[i] + mean[i]
    
    return denormalized_array