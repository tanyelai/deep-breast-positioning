# debug_crop_pad_resize.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from utils import read_dicom_image

def load_npy_image(path):
    """Load and return a .npy image."""
    return np.load(path)

def plot_landmarks(image, landmarks, color='r', ax=None):
    """Plot landmarks on an image. If ax is provided, use it for plotting."""
    if ax is None:
        ax = plt
    if 'pectoral_line' in landmarks:
        ax.plot([landmarks['pectoral_line']['x1'], landmarks['pectoral_line']['x2']],
                 [landmarks['pectoral_line']['y1'], landmarks['pectoral_line']['y2']], color=color, marker='o', linestyle='-')
    if 'nipple' in landmarks:
        ax.scatter(landmarks['nipple']['x'], landmarks['nipple']['y'], color=color)

def visualize_images(dicom_path, processed_path, original_landmarks, processed_landmarks, save_path):
    """Visualize original and processed images with their respective landmarks."""
    # Load images
    original_img = read_dicom_image(dicom_path)
    processed_img = load_npy_image(processed_path)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image with original landmarks
    axs[0].imshow(original_img, cmap='gray')
    plot_landmarks(original_img, original_landmarks, color='yellow', ax=axs[0])
    axs[0].set_title('Original Image with Original Landmarks')
    
    # Processed image with processed landmarks
    axs[1].imshow(processed_img, cmap='gray')
    plot_landmarks(processed_img, processed_landmarks, color='blue', ax=axs[1])
    axs[1].set_title('Processed Image with Processed Landmarks')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Configuration of paths
dicom_base_path = '../quality'
processed_image_dir = '../data/images'
fig_save_path = '../figures/debug_preprocessing'
transformation_details_path = '../data/transformation_details.csv'

# Ensure the directory exists
os.makedirs(fig_save_path, exist_ok=True)

# Load the transformation details and unified_df for DICOM paths
transformation_df = pd.read_csv(transformation_details_path)
unified_df = pd.read_csv('../annotations/unified_data.csv')

# SOPInstanceUIDs for debugging
sop_instance_uids = [
    'ac0f58c8690f16fe4dcd300bb3e7ee0b',
    '3c5b8c9aa17dd132db42747ad3d4f10e',
    '47c8858666bcce92bcbd57974b5ce522',
    'b664cf1e7c968896144a3a2005cd3eb4',
    '552a67df6d68b8f9498db7cadd23b0e0',
    'e7ad5afd2fb520aca0409c734315e385',
    '3944b33b99c98e763763b9da54b4f41c',
    '20ff52d0631b27a5387abcd5920b3e5d',
    '0e31855d02eadf8670ffaeeaeddbf229',
    'f9e67324d12f697ff0c00f327fc4fca9',
    'e42cba584142f92dc42ef3fb95b5878d',
    'cfba6992d0d0da02d516039af80a0549',
    'c72f340f53c8a8acd3566cf239af8463'
]


for sop_instance_uid in sop_instance_uids:
    # Find the row in unified_df to get the DICOM path
    unified_row = unified_df[unified_df['SOPInstanceUID'] == sop_instance_uid].iloc[0]
    study_instance_uid = unified_row['StudyInstanceUID']
    series_instance_uid = unified_row['SeriesInstanceUID']
    dicom_path = os.path.join(dicom_base_path, study_instance_uid, series_instance_uid, f"{sop_instance_uid}.dcm")

    # Get the transformation row for landmarks
    transformation_row = transformation_df[transformation_df['SOPInstanceUID'] == sop_instance_uid].iloc[0]
    processed_path = os.path.join(processed_image_dir, f"{sop_instance_uid}.npy")
    
    # Load and parse landmarks
    original_landmarks = json.loads(transformation_row['original_landmarks'])
    processed_landmarks = json.loads(transformation_row['adjusted_landmarks'])

    # Visualize and save
    save_path = os.path.join(fig_save_path, f"{sop_instance_uid}.png")
    visualize_images(dicom_path, processed_path, original_landmarks, processed_landmarks, save_path)