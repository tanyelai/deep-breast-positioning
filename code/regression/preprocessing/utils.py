import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import pandas as pd
import numpy as np
from PIL import Image
import json 
import os
import matplotlib.pyplot as plt
from PIL import ImageDraw
import csv

def read_dicom_image(path):
    """Load and return a DICOM image as a PIL Image."""
    dicom = pydicom.dcmread(path)
    image = apply_voi_lut(dicom.pixel_array, dicom)
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        image = np.max(image) - image
    image = image - np.min(image)
    image = image / np.max(image)
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image)

def load_csv_data(csv_path):
    """Load and return CSV data."""
    return pd.read_csv(csv_path)

def extract_line_data(row):
    """Extract and return line data from a CSV row."""
    data = eval(row['data'])
    vertices = data['vertices']
    return vertices

def determine_side(series_description):
    """Determine and return the side (L or R) based on the SeriesDescription."""
    if "L-MLO" in series_description:
        return "L"
    elif "R-MLO" in series_description:
        return "R"
    else:
        return None

def midpoint_from_bbox(bbox):
    """Calculate and return the midpoint of a bounding box."""
    midpoint = (bbox['x'] + bbox['width'] / 2, bbox['y'] + bbox['height'] / 2)
    return midpoint

def save_landmarks_as_json(pectoral_line_coords, nipple_coord, path):
    """
    Save the pectoral muscle line coordinates and nipple coordinate to a JSON file.
    Parameters:
        pectoral_line_coords (tuple): A tuple of tuples ((x1, y1), (x2, y2)) representing the pectoral muscle line.
        nipple_coord (tuple): The (x, y) coordinates of the nipple.
        path (str): File path where the JSON will be saved.
    """
    data = {
        "pectoral_line": {
            "x1": pectoral_line_coords[0][0],
            "y1": pectoral_line_coords[0][1],
            "x2": pectoral_line_coords[1][0],
            "y2": pectoral_line_coords[1][1]
        },
        "nipple": {
            "x": nipple_coord[0],
            "y": nipple_coord[1]
        }
    }

    with open(path, 'w') as json_file:
        json.dump(data, json_file)


def reorder_landmarks(landmarks):
    """
    Ensure that the landmark with the lower y-value is first.
    This also ensures the x-values are associated with the correct y-values.
    """
    (x1, y1), (x2, y2) = landmarks

    if y1 > y2:
        return [(x2, y2), (x1, y1)]
    else:
        return [(x1, y1), (x2, y2)]
    

def adjust_pectoralis_line(pectoralis_line, image_width, image_height, side, TOP_OFFSET=10):
    """Adjust the endpoints of the pectoralis line to standard distances from the image boundaries."""

    # Unpack the original line endpoints
    (x1, y1), (x2, y2) = pectoralis_line

    # Calculate the slope and angle
    slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')

    # Adjust the upper endpoint y-coordinate
    if y1 > y2:  # Ensuring y1 is always the upper point
        y1, y2 = y2, y1
        x1, x2 = x2, x1

    y1_new = TOP_OFFSET
    x1_new = x1 + (y1_new - y1) / np.tan(np.arctan(slope)) if slope != float('inf') else x1

    # Adjust x or y for the lower endpoint, maintaining the line's angle
    if side == 'L':
        # If it's the left side, attempt to extend x2 to the left edge while maintaining the line's angle
        x2_new = 0 + TOP_OFFSET  # Adjust for the left side with some offset if needed
    else:
        # If it's the right side, extend x2 to the right edge while maintaining the line's angle
        x2_new = image_width - TOP_OFFSET  # Adjust for the right side

    # Calculate the new y2 based on x2_new and maintaining the line angle
    y2_new = slope * (x2_new - x1_new) + y1_new

    # Ensure the new coordinates are within image boundaries
    x1_new, x2_new = max(0, min(x1_new, image_width)), max(0, min(x2_new, image_width))
    y1_new, y2_new = max(0, min(y1_new, image_height)), max(0, min(y2_new, image_height))

    return [(x1_new, y1_new), (x2_new, y2_new)]


def verify_landmarks(dicom_paths, landmark_paths, fig_save_path, sop_instance_uids):
    """Visualize and verify landmarks on DICOM images with uniform and clearly visible red dots and lines for key points."""
    for dicom_path, landmark_path, sop_instance_uid in zip(dicom_paths, landmark_paths, sop_instance_uids):
        # Load DICOM image as PIL Image
        dicom_image = read_dicom_image(dicom_path)

        # Convert the image to RGB to support colored drawings
        dicom_image_rgb = dicom_image.convert("RGB")

        # Load landmark data from JSON
        with open(landmark_path, 'r') as json_file:
            landmarks = json.load(json_file)
        
        # Draw landmarks on the RGB image
        draw = ImageDraw.Draw(dicom_image_rgb)
        pectoral_line = landmarks['pectoral_line']
        nipple = landmarks['nipple']

        # Uniform dot size for all key points
        dot_radius = 10 

        # Use red color for high visibility
        dot_color = 'red'

        # Draw red dots for the start and end points of the pectoral line
        draw.ellipse([(pectoral_line['x1'] - dot_radius, pectoral_line['y1'] - dot_radius), 
                      (pectoral_line['x1'] + dot_radius, pectoral_line['y1'] + dot_radius)], fill=dot_color)
        draw.ellipse([(pectoral_line['x2'] - dot_radius, pectoral_line['y2'] - dot_radius), 
                      (pectoral_line['x2'] + dot_radius, pectoral_line['y2'] + dot_radius)], fill=dot_color)

        # Draw a line between the start and end points of the pectoral line
        line_color = 'red'  # Using red for the line as well for consistency
        draw.line([pectoral_line['x1'], pectoral_line['y1'], pectoral_line['x2'], pectoral_line['y2']], fill=line_color, width=3)
        
        # Draw a red dot for the nipple with the same size
        draw.ellipse([(nipple['x'] - dot_radius, nipple['y'] - dot_radius), 
                      (nipple['x'] + dot_radius, nipple['y'] + dot_radius)], fill=dot_color)

        # Save or display the annotated RGB image
        save_fig_path = os.path.join(fig_save_path, f"{sop_instance_uid}_annotated.png")
        if not os.path.exists(fig_save_path):
            os.makedirs(fig_save_path)
        dicom_image_rgb.save(save_fig_path)
        print(f"Saved annotated image for {sop_instance_uid} at {save_fig_path}")
