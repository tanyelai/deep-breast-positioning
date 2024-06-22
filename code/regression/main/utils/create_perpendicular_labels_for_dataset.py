import numpy as np
import pandas as pd
from math import atan2, degrees
import json

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

def is_point_inside_image_with_threshold(image_shape, point, threshold_percentage=0):
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

def preprocess_data(split_file, details_file):
    split_df = pd.read_csv(split_file)
    details_df = pd.read_csv(details_file)
    unified_df = pd.merge(split_df, details_df, on='SOPInstanceUID', how='left')
    df = unified_df[unified_df['labelName'] == 'Pectoralis']
    return df

def evaluate_and_label(unified_df):
    results = []

    for idx, row in unified_df.iterrows():
        # Parse the adjusted_landmarks from JSON string to dictionary
        adjusted_landmarks_str = row['adjusted_landmarks']
        adjusted_landmarks = json.loads(adjusted_landmarks_str)

        # Extract the x1, y1, x2, y2 coordinates
        x1, y1 = adjusted_landmarks["pectoral_line"]["x1"], adjusted_landmarks["pectoral_line"]["y1"]
        x2, y2 = adjusted_landmarks["pectoral_line"]["x2"], adjusted_landmarks["pectoral_line"]["y2"]
        xn, yn = adjusted_landmarks["nipple"]["x"], adjusted_landmarks["nipple"]["y"]
        img_height, img_width = 512, 512

        # Calculate the perpendicular endpoint from the extracted coordinates
        orig_endpoint = calculate_perpendicular_endpoint(x1, y1, x2, y2, xn, yn)
        orig_quality = is_point_inside_image_with_threshold((img_height, img_width),
                                                             orig_endpoint, threshold_percentage=0)
        

        results.append({
            "StudyInstanceUID": row['StudyInstanceUID'],
            "SOPInstanceUID": row['SOPInstanceUID'],
            "Side": row['SeriesDescription'],
            "endpoint_x": orig_endpoint[0],
            "endpoint_y": orig_endpoint[1],
            "automated_quality": "Good" if orig_quality else "Bad",
            "manual_quality": row['qualitativeLabel']
        })

    return pd.DataFrame(results)


# change paths to your own
def main():
    split_file = '../annotations/unified_data_with_splits.csv'
    details_file = '../data/transformation_details.csv'
    
    unified_df = preprocess_data(split_file, details_file)
    results_df = evaluate_and_label(unified_df)
    
    # Save the results to a CSV file
    results_df.to_csv('perpendicular_coordinates_quality.csv', index=False)
    print("Results saved to perpendicular_coordinates_quality.csv")

if __name__ == "__main__":
    main()