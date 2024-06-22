import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class QualityDataset(Dataset):
    def __init__(self, image_dir, annotations_file, split_type, label_type='manual'):
        """
        Args:
            image_dir (str): Path to the image directory.
            annotations_file (str): Path to the annotations CSV file.
            split_type (str): One of 'Train', 'Validation', or 'Test'.
            label_type (str): 'manual' for manual_quality or 'automated' for automated_quality.
        """
        self.image_dir = image_dir
        self.df = pd.read_csv(annotations_file)
        self.df = self.df[self.df['Split'] == split_type]
        self.label_type = label_type
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert numpy arrays to torch tensors
            #transforms.Normalize(mean=[0.485], std=[0.229])  # Normalize grayscale images
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sop_instance_uid = row['SOPInstanceUID']
        image_path = os.path.join(self.image_dir, f"{sop_instance_uid}.npy")
        image = np.load(image_path)
        
        if self.label_type == 'manual':
            label = 0 if row['manual_quality'] == 'Bad' else 1
        elif self.label_type == 'automated':
            label = 0 if row['automated_quality'] == 'Bad' else 1

        image = self.transform(image)
        return image, label

def get_dataloader(image_dir, annotations_file, split_type, label_type='manual', batch_size=8):
    dataset = QualityDataset(image_dir, annotations_file, split_type, label_type)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split_type == 'Train'), num_workers=4)
