from utils import (read_dicom_image, load_csv_data,
                    extract_line_data, midpoint_from_bbox,
                      save_landmarks_as_json, reorder_landmarks,
                      adjust_pectoralis_line, determine_side)
import os
import pandas as pd
from tqdm import tqdm

def process_images(csv_path, dicom_base_path, landmarks_save_path):
    df = load_csv_data(csv_path)
    
    pectoralis_df = df[df['labelName'] == 'Pectoralis']
    nipple_df = df[df['labelName'] == 'Nipple']

    for _, row in tqdm(pectoralis_df.iterrows(), total=pectoralis_df.shape[0], desc="Processing Images"):
        dicom_path = os.path.join(dicom_base_path, row['StudyInstanceUID'], row['SeriesInstanceUID'], f"{row['SOPInstanceUID']}.dcm")
        json_path = os.path.join(landmarks_save_path, f"{row['SOPInstanceUID']}.json")

        os.makedirs(os.path.dirname(json_path), exist_ok=True)

        if row['SOPInstanceUID'] not in nipple_df['SOPInstanceUID'].values:
            continue

        image = read_dicom_image(dicom_path)
        image_width, image_height = image.size

        vertices = extract_line_data(row)
        pectoralis_line = reorder_landmarks([(float(vertices[0][0]), float(vertices[0][1])), (float(vertices[1][0]), float(vertices[1][1]))])
        side = determine_side(row['SeriesDescription'])

        adjusted_pectoralis_line = adjust_pectoralis_line(pectoralis_line, image_width, image_height, side)

        nipple_record = nipple_df[nipple_df['SOPInstanceUID'] == row['SOPInstanceUID']].iloc[0]
        bbox_data = eval(nipple_record['data'])
        nipple_point = midpoint_from_bbox(bbox_data)

        save_landmarks_as_json(adjusted_pectoralis_line, nipple_point, json_path)
        print(f"Processed and saved landmarks for {row['SOPInstanceUID']}")

csv_path = '../annotations/unified_data.csv'
dicom_base_path = '../quality'
landmarks_save_path = '../landmark_coords'

process_images(csv_path, dicom_base_path, landmarks_save_path)