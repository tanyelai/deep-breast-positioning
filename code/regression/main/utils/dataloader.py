import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import json

class Dataset(Dataset):
    def __init__(self, dataframe, base_image_dir, target_task='all'):
        """
        Args:
            dataframe (DataFrame): DataFrame containing the data paths and adjusted landmarks.
            base_image_dir (string): Directory with all the images as .npy files.
        """
        self.dataframe = dataframe
        self.base_image_dir = base_image_dir
        self.target_task = target_task

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['SOPInstanceUID'] + '.npy'
        img_path = os.path.join(self.base_image_dir, img_name)
        image = np.load(img_path)
        image_height, image_width = image.shape[:2]
        adjusted_landmarks_str = self.dataframe.iloc[idx]['adjusted_landmarks']
        adjusted_landmarks = json.loads(adjusted_landmarks_str)
        
        # Depending on the task, parse landmarks differently
        coordinates = self.parse_coordinates(adjusted_landmarks, image_width, image_height)
        coordinates = torch.tensor(coordinates, dtype=torch.float)
        image = torch.from_numpy(image).float().unsqueeze(0)  # Add channel dimension if grayscale
        return image, coordinates

    def parse_coordinates(self, landmarks, width, height):
        if self.target_task == 'pectoralis':
            return [landmarks["pectoral_line"]["x1"] / width, landmarks["pectoral_line"]["y1"] / height,
                    landmarks["pectoral_line"]["x2"] / width, landmarks["pectoral_line"]["y2"] / height]
        elif self.target_task == 'nipple':
            return [landmarks["nipple"]["x"] / width, landmarks["nipple"]["y"] / height]
        elif self.target_task == 'all':
            return [landmarks["pectoral_line"]["x1"] / width, landmarks["pectoral_line"]["y1"] / height,
                    landmarks["pectoral_line"]["x2"] / width, landmarks["pectoral_line"]["y2"] / height,
                    landmarks["nipple"]["x"] / width, landmarks["nipple"]["y"] / height]
        else:
            raise ValueError("Invalid target_config. Must be one of ['pectoralis', 'nipple', 'all']")


def create_dataloaders(unified_df, config, return_test_df=False):
    df = unified_df[unified_df['labelName'] == 'Pectoralis']
    datasets = {}
    test_df = None
    for split in ['Train', 'Validation', 'Test']:
        split_df = df[df['Split'] == split]
        if split == 'Test':
            test_df = split_df
        datasets[split] = Dataset(split_df, config['base_image_dir'], config['target_task'])

    dataloaders = {x: DataLoader(datasets[x], batch_size=config['batch_size'], shuffle=True if x == 'Train' else False) for x in datasets.keys()}

    # Return both if the flag is set, otherwise return just dataloaders
    if return_test_df:
        return dataloaders, test_df
    else:
        return dataloaders



def preprocess_data(config):
    split_df = pd.read_csv(config['split_file'])
    details_df = pd.read_csv(config['details_file'])

    # Merge the DataFrames on the 'SOPInstanceUID' column
    unified_df = pd.merge(split_df, details_df, on='SOPInstanceUID', how='left')

    return unified_df