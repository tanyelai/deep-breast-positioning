import json
from math import atan2, degrees
import numpy as np
import pandas as pd
import torch
from utils.models import UNet, RAUNet, CRAUNet, ResNeXt50


def load_config(file_path):
    """Load configuration from a JSON file."""
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

def load_model(model_type, model_path, device):
    """Load and return the specified model type onto the device."""
    if model_type == "UNet":
        model = UNet(in_channels=1, out_features=6).to(device)
    elif model_type == "RAUNet":
        model = RAUNet(in_channels=1, out_features=6).to(device)
    elif model_type == "CRAUNet":
        model = CRAUNet(in_channels=1, out_features=6).to(device)
    elif model_type == "ResNeXt50":
        model = ResNeXt50(in_channels=1, out_features=6).to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess_data(config):
    split_df = pd.read_csv(config['split_file'])
    details_df = pd.read_csv(config['details_file'])
    unified_df = pd.merge(split_df, details_df, on='SOPInstanceUID', how='left')
    return unified_df

def calculate_perpendicular_endpoint(x1, y1, x2, y2, xn, yn):
    """
    Calculate the endpoint of the perpendicular line from (xn, yn) to the line
    defined by (x1, y1) and (x2, y2).
    """
    if x1 == x2:  # Vertical line case
        return x1, yn
    elif y1 == y2:  # Horizontal line case
        return xn, y1
    else:
        # Slope of the original line
        m = (y2 - y1) / (x2 - x1)
        # Intercept of the original line
        c = y1 - m * x1
        # Slope and intercept of the perpendicular line
        m_perp = -1 / m
        c_perp = yn - m_perp * xn
        # Calculate intersection
        x_intersect = (c_perp - c) / (m - m_perp)
        y_intersect = m * x_intersect + c
        return x_intersect, y_intersect

def is_point_inside_image(image_shape, point):
    """
    Check if a point is within the image boundaries.
    """
    x, y = point
    h, w = image_shape[:2]
    return 0 <= x < w and 0 <= y < h

def is_point_inside_image_with_threshold(image_shape, point, threshold_percentage=1):
    """
    Check if a point is within the image boundaries, allowing for a small margin outside
    the image as defined by a threshold based on the longest dimension of the image.
    
    Args:
    - image_shape: Tuple of (height, width) of the image.
    - point: Tuple of (x, y) coordinates of the point.
    - threshold_percentage: Percentage of the longest image dimension to allow as a margin.
    
    Returns:
    - A boolean indicating whether the point is considered inside the image (including the margin).
    """
    x, y = point
    h, w = image_shape[:2]
    
    # Determine the longest dimension and calculate the threshold in pixels
    longest_dimension = max(h, w)
    threshold_in_pixels = (threshold_percentage / 100.0) * longest_dimension
    
    # Adjust the image bounds by the threshold
    return (-threshold_in_pixels <= x < w + threshold_in_pixels) and (-threshold_in_pixels <= y < h + threshold_in_pixels)


def euclidean_distance(x1, y1, x2, y2):
    """
    Calculate the Euclidean distance between two points (x1, y1) and (x2, y2).
    """
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def calculate_mm_distance(px, py, df_row):
    """
    Calculate the millimeter distance between predicted and original landmarks using adjusted pixel spacing.
    Assumes that pixel spacings in the DataFrame are stored as 'adjusted_pixel_spacing' in the format "spacingX\\spacingY".
    """
    # Parse the adjusted pixel spacing
    spacing_x, spacing_y = parse_pixel_spacing(str(df_row['adjusted_pixel_spacing']))

    # Calculate the distance in pixels
    pixel_distance = np.sqrt((px[0] - py[0])**2 + (px[1] - py[1])**2)

    # Convert pixel distance to millimeters by averaging the spacings if they differ
    mm_distance = pixel_distance * ((spacing_x + spacing_y) / 2)
    return mm_distance


def parse_pixel_spacing(spacing_str):
    """
    Parse the pixel spacing string from DICOM metadata.
    Assumes that the input string is formatted like '0.123\\0.456'.
    Adjust the split function according to actual delimiter used in your data.
    """
    parts = spacing_str.split('\\')  # Ensure this is the correct delimiter
    return tuple(map(float, parts))


def calculate_side_specific_angle(x1, y1, x2, y2, orientation):
    """
    Calculate the angle of a line with respect to the vertical line extending from the side of the breast.
    The angle is calculated such that it measures how much the line deviates from being parallel to the breast side's vertical.

    Args:
        x1, y1: Coordinates of the first point on the line.
        x2, y2: Coordinates of the second point on the line.
        orientation: 'L-MLO' for left and 'R-MLO' for right, indicating the breast's position.

    Returns:
        The angle in degrees between the line and the nearest vertical side, within a range of 0 to 180 degrees.
    """
    # Calculate the angle of the line with respect to the horizontal axis
    angle_rad = atan2(y2 - y1, x2 - x1)
    angle_deg = degrees(angle_rad)

    # Adjust the angle to be relative to the vertical line from the breast's side
    if orientation == 'L-MLO':
        # Calculate angle from left vertical: The line's angle + 90 degrees
        angle_from_vertical = (angle_deg + 90) % 360
    elif orientation == 'R-MLO':
        # Calculate angle from right vertical: 90 degrees - the line's angle
        angle_from_vertical = (90 - angle_deg) % 360

    # Normalize the angle to ensure it is within 0 to 180 degrees
    if angle_from_vertical > 180:
        angle_from_vertical = 360 - angle_from_vertical
    
    if angle_from_vertical > 90:
        angle_from_vertical = 180 - angle_from_vertical

    return angle_from_vertical




####

# def calculate_sensitivity_specificity(predictions, truths):
#     """
#     Calculate sensitivity and specificity based on binary classification outcomes.
#     """
#     tp = np.sum((predictions == 1) & (truths == 1))
#     fn = np.sum((predictions == 0) & (truths == 1))
#     tn = np.sum((predictions == 0) & (truths == 0))
#     fp = np.sum((predictions == 1) & (truths == 0))

#     sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
#     specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
#     return sensitivity, specificity

def calculate_sensitivity_specificity(predictions, truths):
    """
    Calculate sensitivity and specificity based on binary classification outcomes.
    Sensitivity (Recall) is calculated for the 'Bad' cases (label 0).
    Specificity is calculated for the 'Good' cases (label 1).
    """
    tp = np.sum((predictions == 0) & (truths == 0))  # True Positives for 'Bad' cases
    fn = np.sum((predictions == 1) & (truths == 0))  # False Negatives for 'Bad' cases
    tn = np.sum((predictions == 1) & (truths == 1))  # True Negatives for 'Good' cases
    fp = np.sum((predictions == 0) & (truths == 1))  # False Positives for 'Good' cases

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return sensitivity, specificity
