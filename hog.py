import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

from skimage.io import imread
from skimage.color import rgb2gray
from skimage import data, exposure
from skimage.feature import hog
from skimage.transform import resize

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer

IMG_PATH = 'CAPTCHA-10k/'

def get_letters_from_img(img):
    # These widths were set by hand, and tested. Results were satisfactory enough.
    window_widths = [(0, 40), (30, 70), (60, 100), (90, 130), (120, 160), (140, 180)]

    quadrants = []
    for (start_col, end_col) in window_widths:
        quadrant = img[:, start_col:end_col]
        quadrants.append(quadrant)
    quadrants = [resize(q, (128*4, 64*4)) for q in quadrants]

    return quadrants

def hog_on_letters(letters:list) -> list:
    descriptors = []
    
    for l in letters:
        fd = hog(
			l,
			orientations=8,
			pixels_per_cell=(8, 8),
			cells_per_block=(2, 2),
		)
        descriptors.append(fd)
        
    return descriptors

def get_descriptors_and_labels(captcha_dir, label_dir):
    descriptors = []
    labels = []

    for img_filename in os.listdir(captcha_dir):
        if img_filename.endswith('.jpg'):
            print(f"Running on {img_filename}")
            img_path = os.path.join(captcha_dir, img_filename)
            img = rgb2gray(imread(img_path))
            
            # Get the letters from the image
            letters = get_letters_from_img(img)
            # Read the corresponding label
            label_path = os.path.join(label_dir, img_filename.replace('.jpg', '.txt'))
            with open(label_path, 'r') as label_file:
                label = label_file.read().strip()
            
            # Ensure we have the same number of letters and label characters
            if len(letters) != len(label):
                print(f'Warning: Mismatch between number of letters and label length for {img_filename}')
                continue
            
            # Computing hog descriptors for each letter
            hog_descriptors = hog_on_letters(letters)
            for descriptor, letter_label in zip(hog_descriptors, label):
                descriptors.append(descriptor)
                labels.append(letter_label)

    # Convert lists to numpy arrays
    np_descriptors = np.array(descriptors)
    np_labels = np.array(labels)

    return np_descriptors, np_labels

# Necessary Directories
training_dir = os.path.join(IMG_PATH, 'treinamento/')
validation_dir = os.path.join(IMG_PATH, 'validacao/')
test_dir = os.path.join(IMG_PATH, 'teste/')
label_dir = os.path.join(IMG_PATH, 'labels10k/')

# Define paths for pickled files
PKL_PATH = 'saved_datasets/'
train_data_path = os.path.join(PKL_PATH, 'train_data.pkl')
validation_data_path = os.path.join(PKL_PATH, 'validation_data.pkl')
test_data_path = os.path.join(PKL_PATH, 'test_data.pkl')

# Function to save data to pickle files
def save_data():
    with open(train_data_path, 'wb') as f:
        pickle.dump((train_X, train_y), f)
    
    with open(validation_data_path, 'wb') as f:
        pickle.dump((validation_X, validation_y), f)
    
    with open(test_data_path, 'wb') as f:
        pickle.dump((test_X, test_y), f)
    
    print("Data has been pickled and saved.")

# Function to load data from pickle files
def load_data():
    global train_X, train_y, validation_X, validation_y, test_X, test_y

    with open(train_data_path, 'rb') as f:
        train_X, train_y = pickle.load(f)
    
    with open(validation_data_path, 'rb') as f:
        validation_X, validation_y = pickle.load(f)
    
    with open(test_data_path, 'rb') as f:
        test_X, test_y = pickle.load(f)

    print("Data has been loaded from pickle files.")

# Check if pickle files exist
if all(os.path.exists(p) for p in [train_data_path, validation_data_path, test_data_path]):
    print("Pickle files found. Loading data...")
    load_data()
else:
    print("Pickle files not found. Generating data...")
    print("Generating training dataset...")
    train_X, train_y = get_descriptors_and_labels(training_dir, label_dir)
    print("Generating validation dataset...")
    validation_X, validation_y = get_descriptors_and_labels(validation_dir, label_dir)
    print("Generating testing dataset...")
    test_X, test_y = get_descriptors_and_labels(test_dir, label_dir)
    save_data()