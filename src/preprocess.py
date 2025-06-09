import os
from PIL import Image, ImageChops
import cv2
import numpy as np
from numpy import asarray

class ImagePreprocessor:
    def __init__(self, input_dir='input', output_dir='processed'):
        """
        Initialize the ImagePreprocessor with input and output directories.
        
        Args:
            input_dir (str): Directory containing input images
            output_dir (str): Directory to save processed images
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    @staticmethod
    def trim(im):
        """Trim whitespace from PIL Image."""
        bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
        diff = ImageChops.difference(im, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        if bbox:
            return im.crop(bbox)
        return im

    @staticmethod
    def crop_image(img, tol=0):
        """Crop numpy array image based on tolerance threshold."""
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]

    def get_image_names(self, extensions=[".jpg"]):
        """Get names of all images in directory with specified extensions."""
        names = []
        for root, dirnames, filenames in os.walk(self.input_dir):
            for filename in filenames:
                _, ext = os.path.splitext(filename)
                if ext.lower() in extensions:
                    names.append(filename)
        return names

    def process_image(self, filename, method='trim'):
        """
        Process a single image using the specified method.
        
        Args:
            filename (str): Name of the image file to process
            method (str): Processing method ('trim' or 'crop')            
        Returns:
            str: Path to the processed image
        """
        input_path = os.path.join(self.input_dir, filename)
        output_path = os.path.join(self.output_dir, filename)
        
        if method.lower() == 'trim':
            img = Image.open(input_path)
            trimmed = self.trim(img)
            numpydata = asarray(trimmed)
            final_image = Image.fromarray(numpydata)
            final_image.save(output_path)
        elif method.lower() == 'crop':
            img = cv2.imread(input_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cropped = self.crop_image(gray)
            cv2.imwrite(output_path, cropped)
        else:
            raise ValueError(f"Unknown processing method: {method}")
            
        return output_path

    def process_all_images(self, method='trim'):
        """
        Process all images in the input directory.
        
        Args:
            method (str): Processing method ('trim' or 'crop')
            
        Returns:
            list: Paths to all processed images
        """
        names = self.get_image_names()
        processed_paths = []
        
        for name in names:
            try:
                output_path = self.process_image(name, method)
                processed_paths.append(output_path)
            except Exception as e:
                print(f"Error processing {name}: {str(e)}")
                
        return processed_paths

