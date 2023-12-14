import cv2
import numpy as np

from .preprocessor import crop_frame, extract_hair

class SegmentMelISIC():
    def __init__(self):
        pass

    def gaussian_blur(self, image, kernel_size = (5, 5)):
        # Apply Gaussian blur to smooth the image
        return cv2.GaussianBlur(image, kernel_size, 0)
    
    def contrast_enhancment(self, image):
        # Convert the image to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Split the LAB image into L, A, and B channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to the L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge the CLAHE-enhanced L channel with the original A and B channels
        limg = cv2.merge((cl, a, b))
        
        # Convert the LAB image back to BGR color space
        result = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
        return result

    def kmeans(self, input_img, k=2):
        try:
            # Reshape the image to a 2D array of pixels (rows) and color channels (columns)
            data = input_img.reshape((-1, 3)).astype(np.float32)
    
            # Perform k-means clustering
            _, labels, centers = cv2.kmeans(
                data, k, None,
                criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0),
                attempts=15,
                flags=cv2.KMEANS_PP_CENTERS
            )
    
            # Replace pixel values with their center value for each channel
            # for i in range(data.shape[0]):
            #     center_id = labels[i]
            #     data[i] = centers[center_id]
    
            # Reshape the data back to the original image shape
            segmented_img = data.reshape(input_img.shape).astype(np.uint8)
    
            # Convert the segmented image to grayscale
            segmented_gray = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
    
            # Apply adaptive thresholding to obtain a binary image
            _, thresholded_img = cv2.threshold(segmented_gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    
            return thresholded_img
    
        except Exception as e:
            print(str(e))

    def remove_circular_borders(self, image):
        # Find connected components and their properties
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate the center and diameter of the largest component
        largest_contour = max(contours, key=cv2.contourArea)
    
        # finds a circle of the minimum area enclosing an area
        (x , y), radius = cv2.minEnclosingCircle(largest_contour) # returns the center and radius

        # to avoid cropping the lesion if it is large
        if radius < max(image.shape)//2:
            return image
    
        center = (int(x), int(y)) #(x, y)
        radius = int(radius)
        
        # Calculate border size based on a percentage of the circle radius
        border_percentage = (radius / min(image.shape[0], image.shape[1])) * 100
        if border_percentage > 25:
            border_percentage = 20
            
        border_size = int(radius * border_percentage / 100)
        
        # Create a new white image with the same shape as the original image
        result_image = np.zeros_like(image)
    
        # Crop the image to focus on the center
        y_min = border_size
        y_max = image.shape[0] - border_size
        x_min = border_size
        x_max = image.shape[1] - border_size
    
        # Copy the cropped tumor region into the new image
        result_image[y_min:y_max, x_min:x_max] = image[y_min:y_max, x_min:x_max]
    
        return result_image

    def remove_lines(self, image, kernel_size = (5, 5), iterations=2):
        # Apply erosion to remove lines (adjust the kernel size as needed)
        return cv2.erode(image, kernel=np.ones(kernel_size, np.uint8), iterations=iterations)

    def keep_largest_region_center(self, segmentation):
        # Find contours in the binary mask
        contours, _ = cv2.findContours(segmentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        if not contours:
            return segmentation
        
        # Find the center of the image
        center_x, center_y = segmentation.shape[1] // 2, segmentation.shape[0] // 2 # //3
    
        # Filter contours based on their proximity to the center
        distance_to_center = lambda c: np.linalg.norm(np.mean(c, axis=0) - np.array([center_x, center_y]))
        contours_centered = sorted(contours, key=distance_to_center)
    
        # Select the contour closest to the center (largest in the center)
        largest_contour_centered = contours_centered[0]
    
        # Create an empty mask to draw the largest contour
        result_mask = np.zeros_like(segmentation)
    
        # Draw the largest contour on the empty mask
        cv2.drawContours(result_mask, [largest_contour_centered], 0, 255, thickness=cv2.FILLED)
    
        return result_mask

    def predict(self, image_path):
        # read the image
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        # crop microscopic frame if it exists
        cropped_image = crop_frame(image)[0]

        # remove hair from the image
        hair_rem_cropped = extract_hair(cropped_image)

        # apply gaussian blur
        blurred = self.gaussian_blur(hair_rem_cropped)

        # enhance contrast
        contrast_enhanced = self.contrast_enhancment(blurred)

        # Apply mean shift filter using OpenCV
        filtered_CE = cv2.pyrMeanShiftFiltering(contrast_enhanced, 5, 30, maxLevel=4) # 25, 20)

        # cluster using kmeans
        clusterd_filtered = self.kmeans(filtered_CE, k=2)

        # remove circular borders 
        initial_seg_clustered = self.remove_circular_borders(clusterd_filtered)

        # remove lines
        eroded_lines_seg = self.remove_lines(initial_seg_clustered, iterations=1)

        # remove any surrounding region
        final_seg = self.keep_largest_region_center(eroded_lines_seg)

        return final_seg