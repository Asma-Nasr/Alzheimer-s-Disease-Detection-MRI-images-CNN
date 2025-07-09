import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def get_class_names(image_path):
    """
    Detect class names from subdirectories.
    
    Args:
        image_path (str): Path to the directory containing class folders
        
    Returns:
        list: Sorted list of class names
    """
    # Get all directories in the image_path
    cls_names = [d for d in os.listdir(image_path) 
                if os.path.isdir(os.path.join(image_path, d))]
    # Sort to maintain consistency
    cls_names.sort()
    return cls_names


def load_and_preprocess_data(base_path, dataset_folder, label_name=None, image_size=150):
    """
    Load images, resize them, and create corresponding labels.
    
    Args:
        base_path (str): Base directory path
        dataset_folder (str): Name of dataset folder (e.g., 'tt')
        label_name (list, optional): List of class labels. If None, will auto-detect.
        image_size (int): Target size for resizing images
        
    Returns:
        tuple: (image_data, label_data, label_names) as numpy arrays
    """
    image_data = []
    label_data = []
    data_path = os.path.join(base_path, dataset_folder)
    
    if label_name is None:
        label_name = get_class_names(data_path)
    
    for class_name in label_name:
        class_path = os.path.join(data_path, class_name)
        for m in tqdm(os.listdir(class_path), desc=f"Loading {class_name}"):
            image = cv2.imread(os.path.join(class_path, m))
            if image is not None:  # Only process if image was properly loaded
                image = cv2.resize(image, (image_size, image_size))
                image_data.append(image)
                label_data.append(class_name)
            
    return np.array(image_data), np.array(label_data), label_name

def prepare_train_test_data(image_data, label_data, label_name=None, test_size=0.2, random_state=42):
    """
    Prepare train/test split and convert labels to categorical.
    
    Args:
        image_data (np.array): Array of images
        label_data (np.array): Array of string labels
        label_name (list, optional): List of class labels. If None, will auto-detect.
        test_size (float): Proportion for test split
        random_state (int): Random seed
        
    Returns:
        tuple: (X_train, X_test, Y_train, Y_test, label_names)
    """
    if label_name is None:
        label_name = sorted(list(set(label_data)))  # Get unique classes from labels
    
    # Shuffle data
    image_data, label_data = shuffle(image_data, label_data, random_state=random_state)
    
    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        image_data, label_data, test_size=test_size, random_state=random_state)
    
    # Convert string labels to categorical
    def convert_labels(labels, label_list):
        numeric_labels = [label_list.index(n) for n in labels]
        return to_categorical(numeric_labels, num_classes=len(label_list))
    
    Y_train = convert_labels(Y_train, label_name)
    Y_test = convert_labels(Y_test, label_name)
    
    return X_train, X_test, Y_train, Y_test, label_name
