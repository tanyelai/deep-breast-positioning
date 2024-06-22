# crop_pad_resize_dataset.py

import os
import cv2
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from skimage import measure, morphology
from scipy import ndimage
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

# Configuration class for centralizing settings
class Config:
    TARGET_SIZE = (512, 512)

def load_dicom(path):
    """Loads a DICOM image, applies VOI LUT if available, and returns it as a normalized numpy array."""
    dicom = pydicom.dcmread(path)
    data = apply_voi_lut(dicom.pixel_array, dicom)
    data = data.astype(np.float32)
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.max(data) - data
    data -= np.min(data)
    data /= np.max(data)
    return data

def load_landmarks(json_path):
    """Loads landmark data from a JSON file."""
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

def adjust_landmarks(original_landmarks, crop_coords, pad_coords, original_shape, series_description):
    """
    Adjusts landmark coordinates based on image processing steps: cropping, padding, and resizing.
    
    :param original_landmarks: Dictionary of landmarks with their coordinates.
                               e.g., {"pectoral_line": {"x1": 647.5, "y1": 92.4, "x2": 46.1, "y2": 1292.1},
                                      "nipple": {"x": 869.9, "y": 1931.9}}.
    :param crop_coords: Tuple of crop coordinates (rmin, rmax, cmin, cmax).
    :param pad_coords: Tuple of padding values (pad_left, pad_right, pad_top, pad_bottom).
    :param original_shape: Tuple of the original image shape (height, width).
    :param series_description: Description of the image series for potential specific adjustments.
    
    :return: Dictionary of adjusted landmarks.
    """
    rmin, rmax, cmin, cmax = crop_coords
    pad_left, pad_right, pad_top, pad_bottom = pad_coords
    original_height, original_width = original_shape[:2]

    # Calculate scale factors for resizing
    target_height, target_width = Config.TARGET_SIZE
    scale_y = target_height / (rmax - rmin + 1 + pad_top + pad_bottom)
    scale_x = target_width / (cmax - cmin + 1 + pad_left + pad_right)
    
    adjusted_landmarks = {}
    for landmark, points in original_landmarks.items():
        adjusted_points = {}
        for key, value in points.items():
            if 'x' in key:
                # Adjust for cropping, padding, and resizing
                adjusted_x = (value - cmin + pad_left) * scale_x
                adjusted_points[key] = adjusted_x
            elif 'y' in key:
                adjusted_y = (value - rmin + pad_top) * scale_y
                adjusted_points[key] = adjusted_y
        adjusted_landmarks[landmark] = adjusted_points

    return adjusted_landmarks


class MammogramDataset:
    def __init__(self, annotations_file, base_image_dir, base_landmark_dir, output_dir):
        self.dataframe = pd.read_csv(annotations_file)
        self.base_image_dir = base_image_dir
        self.base_landmark_dir = base_landmark_dir
        self.output_dir = output_dir
        self.output_image_dir = os.path.join(output_dir, 'images')
        self.transformation_details = []

        os.makedirs(self.output_image_dir, exist_ok=True)

    def _find_largest_rectangle(self, img):
        """Finds and returns the bounding box of the largest rectangle in the image."""
        thresh_val = img.mean()
        binary_image = img > thresh_val
        cleaned_image = morphology.opening(binary_image, morphology.disk(3))
        labeled_image, _ = ndimage.label(cleaned_image)
        regions = measure.regionprops(labeled_image, intensity_image=img)
        largest_region = max(regions, key=lambda x: x.area)
        minr, minc, maxr, maxc = largest_region.bbox
        return minr, maxr, minc, maxc

    def _crop_image(self, img):
        """Crops the image based on the largest rectangle found."""
        rmin, rmax, cmin, cmax = self._find_largest_rectangle(img)
        cropped_img = img[rmin:rmax+1, cmin:cmax+1]
        return cropped_img, (rmin, rmax, cmin, cmax)

    def _pad_and_resize(self, img, series_description):
        """Pads and resizes the image to a target size."""
        target_size = Config.TARGET_SIZE
        height, width = img.shape[:2]
        max_side = max(height, width)
        total_padding = max_side - width
        pad_top = pad_bottom = (max_side - height) // 2

        # Determine padding based on series description for L-MLO and R-MLO
        if "L-MLO" in series_description:
            pad_left = 0
            pad_right = total_padding
        elif "R-MLO" in series_description:
            pad_left = total_padding
            pad_right = 0
        else:  # Default case, can be adjusted as needed
            pad_left = pad_right = total_padding // 2

        img_padded = MammogramDataset.pad_image(img, pad_left, pad_right, pad_top, pad_bottom, constant_values=0)
        img_resized = MammogramDataset.resize_image(img_padded, target_size, is_mask=False)
        return img_padded, img_resized, (pad_left, pad_right, pad_top, pad_bottom)

    @staticmethod
    def pad_image(img, pad_left, pad_right, pad_top, pad_bottom, constant_values=0):
        """Pads the image with specified padding."""
        return np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant', constant_values=constant_values)

    @staticmethod
    def resize_image(img, target_size, is_mask):
        """Resizes the image to the target size."""
        interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        return cv2.resize(img, target_size, interpolation=interpolation)
    
    def adjust_pixel_spacing(self, original_spacing, scale_x, scale_y):
        """
        Adjusts the pixel spacing based on the scaling factors applied during resizing.
        :param original_spacing: Tuple or string representing the original pixel spacing (e.g., "0.085\\0.085").
        :param scale_x: Horizontal scaling factor.
        :param scale_y: Vertical scaling factor.
        :return: Adjusted pixel spacing as a tuple.
        """
        # Ensure original_spacing is a string and split it correctly
        if isinstance(original_spacing, str):
            spacing_parts = original_spacing.split('\\')
            if len(spacing_parts) != 2:
                raise ValueError(f"Unexpected format of pixel spacing: {original_spacing}")
            spacing_x, spacing_y = map(float, spacing_parts)
        elif isinstance(original_spacing, (list, tuple)) and len(original_spacing) == 2:
            spacing_x, spacing_y = map(float, original_spacing)
        else:
            raise TypeError("original_spacing must be a string or a tuple/list with two elements")

        # Adjust pixel spacing by the inverse of scaling factors
        new_spacing_x = spacing_x / scale_x
        new_spacing_y = spacing_y / scale_y
        return new_spacing_x, new_spacing_y


    def process_and_save_images(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = os.path.join(self.base_image_dir, row['StudyInstanceUID'], row['SeriesInstanceUID'], f"{row['SOPInstanceUID']}.dcm")
        json_path = os.path.join(self.base_landmark_dir, f"{row['SOPInstanceUID']}.json")

        img = load_dicom(img_path)
        original_landmarks = load_landmarks(json_path) if os.path.exists(json_path) else {}
        cropped_img, crop_coords = self._crop_image(img)
        padded_img, resized_img, pad_coords = self._pad_and_resize(cropped_img, row['SeriesDescription'])

        # Handle string splitting properly
        original_pixel_spacing = row['ImagerPixelSpacing'].split('\\\\')  # Correct handling of backslashes
        if len(original_pixel_spacing) != 2:
            print(f"Split parts: {original_pixel_spacing}")  # Debug print to see what is being split
            raise ValueError(f"Pixel spacing data in an unexpected format: {row['ImagerPixelSpacing']}")

        # Convert to floats
        original_pixel_spacing = tuple(map(float, original_pixel_spacing))

        # Calculate scaling factors based on the resized image dimensions
        scale_x = resized_img.shape[1] / padded_img.shape[1]
        scale_y = resized_img.shape[0] / padded_img.shape[0]
        adjusted_pixel_spacing = self.adjust_pixel_spacing(original_pixel_spacing, scale_x, scale_y)  # Adjust pixel spacing based on scaling

        adjusted_landmarks = adjust_landmarks(original_landmarks, crop_coords, pad_coords, img.shape, row['SeriesDescription'])

        np.save(os.path.join(self.output_image_dir, f"{row['SOPInstanceUID']}.npy"), resized_img)

        self.transformation_details.append({
            'SOPInstanceUID': row['SOPInstanceUID'],
            'original_shape': img.shape,
            'crop_coords': crop_coords,
            'pad_coords': pad_coords,
            'original_landmarks': json.dumps(original_landmarks),
            'adjusted_landmarks': json.dumps(adjusted_landmarks),
            'original_pixel_spacing': row['ImagerPixelSpacing'],
            'adjusted_pixel_spacing': f"{adjusted_pixel_spacing[0]}\\{adjusted_pixel_spacing[1]}",
        })


    def finalize(self):
        df_details = pd.DataFrame(self.transformation_details)
        df_details.to_csv(os.path.join(self.output_dir, "transformation_details.csv"), index=False)


# change paths to your own
def main():
    annotations_file = '../annotations/unified_data.csv'
    base_image_dir = '../quality'
    base_landmark_dir = '../landmark_coords' 
    output_dir = '../data'

    dataset = MammogramDataset(annotations_file, base_image_dir, base_landmark_dir, output_dir)
    filtered_indices = dataset.dataframe[dataset.dataframe['labelName'] == 'Pectoralis'].index.tolist()

    for idx in tqdm(filtered_indices, desc="Processing Images"):
        dataset.process_and_save_images(idx)

    dataset.finalize()

if __name__ == "__main__":
    main()