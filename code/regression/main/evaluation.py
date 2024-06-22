# evaluation.py

import torch
import numpy as np
import pandas as pd
import argparse
from utils.dataloader import create_dataloaders
from utils.evaluation_utils import load_config, preprocess_data, calculate_perpendicular_endpoint,\
                                 is_point_inside_image_with_threshold, calculate_side_specific_angle,\
                                      calculate_mm_distance, euclidean_distance, calculate_sensitivity_specificity,\
                                        load_model
                


def evaluate_and_label(config, model, test_loader, unified_df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    evaluation_results = []
    perpendicular_distances = []
    pec1_distances = []
    pec2_distances = []
    nipple_distances = []
    angular_distances = []
    mm_distances_pec1 = []
    mm_distances_pec2 = []
    mm_distances_nipple = []
    mm_distances_perpendicular = []
    correct_automated_predictions = 0
    correct_manual_predictions = 0
    pred = []
    automated_truth = []
    manual_truth = []
    
    thresholds = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
    threshold_metrics = {t: {
        'predictions': [], 'truths': [], 
        'total_good_pred': 0, 'total_bad_pred': 0,
        'total_good_truth': 0, 'total_bad_truth': 0,
        'correct_good_count': 0, 'correct_bad_count': 0
    } for t in thresholds}

    with torch.no_grad():
        for i, (image, original_landmarks) in enumerate(test_loader):
            image = image.to(device)
            predictions = model(image).cpu().numpy().flatten()
            original_landmarks = original_landmarks.numpy().flatten()

            # Get the original image dimensions for rescaling
            img_height, img_width = image.shape[2], image.shape[3]

            # Rescale predictions and original landmarks to original dimensions
            predictions_rescaled = np.reshape(predictions, (-1, 2)) * [img_width, img_height]
            original_landmarks_rescaled = np.reshape(original_landmarks, (-1, 2)) * [img_width, img_height]

            predictions_rescaled = predictions_rescaled.flatten()
            original_landmarks_rescaled = original_landmarks_rescaled.flatten()

            # Calculate distances between landmarks
            pec1 = euclidean_distance(original_landmarks_rescaled[0], original_landmarks_rescaled[1], predictions_rescaled[0], predictions_rescaled[1])
            pec2 = euclidean_distance(original_landmarks_rescaled[2], original_landmarks_rescaled[3], predictions_rescaled[2], predictions_rescaled[3])
            pec1_distances.append(pec1); pec2_distances.append(pec2)
            
            nipple_distance = euclidean_distance(original_landmarks_rescaled[4], original_landmarks_rescaled[5], predictions_rescaled[4], predictions_rescaled[5])
            nipple_distances.append(nipple_distance)

            perp_orig_endpoint = calculate_perpendicular_endpoint(*original_landmarks_rescaled[:6])
            perp_pred_endpoint = calculate_perpendicular_endpoint(*predictions_rescaled[:6])
            perp_distance = euclidean_distance(perp_orig_endpoint[0], perp_orig_endpoint[1], perp_pred_endpoint[0], perp_pred_endpoint[1])
            perpendicular_distances.append(perp_distance)

            # Calculate millimeter distances
            mm_pec1 = calculate_mm_distance(original_landmarks_rescaled[:2], predictions_rescaled[:2], unified_df.iloc[i])
            mm_pec2 = calculate_mm_distance(original_landmarks_rescaled[2:4], predictions_rescaled[2:4], unified_df.iloc[i])
            mm_nipple = calculate_mm_distance(original_landmarks_rescaled[4:6], predictions_rescaled[4:6], unified_df.iloc[i])
            mm_perpendicular = calculate_mm_distance(perp_orig_endpoint, perp_pred_endpoint, unified_df.iloc[i])
            mm_distances_pec1.append(mm_pec1); mm_distances_pec2.append(mm_pec2)
            mm_distances_nipple.append(mm_nipple); mm_distances_perpendicular.append(mm_perpendicular)

            image_shape = (img_height, img_width)

            orig_automated_quality = is_point_inside_image_with_threshold(image_shape, perp_orig_endpoint, threshold_percentage=0)
            orig_manual_quality = unified_df.iloc[i]['qualitativeLabel'] == "Good"
            pred_quality = is_point_inside_image_with_threshold(image_shape, perp_pred_endpoint, threshold_percentage=0)
            automated_accuracy = "accurate" if orig_automated_quality == pred_quality else "inaccurate"
            manual_accuracy = "accurate" if orig_manual_quality == pred_quality else "inaccurate"
            correct_automated_predictions += 1 if orig_automated_quality == pred_quality else 0
            correct_manual_predictions += 1 if orig_manual_quality == pred_quality else 0


            # store the prediction and automated_truth for each image
            pred.append(pred_quality)
            automated_truth.append(orig_automated_quality)
            manual_truth.append(orig_manual_quality)

            # Determine image orientation for angle calculation
            orientation = unified_df.iloc[i]['SeriesDescription']

            # Calculate angles for original and predicted points
            original_angle = calculate_side_specific_angle(original_landmarks_rescaled[0], original_landmarks_rescaled[1],
                                                              original_landmarks_rescaled[2], original_landmarks_rescaled[3],
                                                              orientation)
            predicted_angle = calculate_side_specific_angle(predictions_rescaled[0], predictions_rescaled[1],
                                                               predictions_rescaled[2], predictions_rescaled[3],
                                                               orientation)
            angular_distance = abs(original_angle - predicted_angle)
            angular_distances.append(angular_distance)
        
            
            evaluation_results.append({
                "image_index": i+1,
                "study_uid": unified_df.iloc[i]['StudyInstanceUID'],
                "sop_uid": unified_df.iloc[i]['SOPInstanceUID'],
                "automated_accuracy": automated_accuracy,
                "manual_accuracy": manual_accuracy,
                "automated_label": "good" if orig_automated_quality else "bad",
                "manual_label": "good" if orig_manual_quality else "bad",
                "prediction_label": "good" if pred_quality else "bad",
                "perpendicular_distance": perp_distance,
                "pec1_distance": pec1,
                "pec2_distance": pec2,
                "nipple_distance": nipple_distance,
                "angular_distance": angular_distance,
                "mm_pec1": mm_pec1,
                "mm_pec2": mm_pec2,
                "mm_nipple": mm_nipple,
                "mm_perpendicular": mm_perpendicular,
                "original_angle": original_angle,
                "predicted_angle": predicted_angle,
                "perp_org_x": perp_orig_endpoint[0],
                "perp_org_y": perp_orig_endpoint[1],
                "perp_pred_x": perp_pred_endpoint[0],
                "perp_pred_y": perp_pred_endpoint[1],
                "pec1_org_x": original_landmarks_rescaled[0],
                "pec1_org_y": original_landmarks_rescaled[1],
                "pec1_pred_x": predictions_rescaled[0],
                "pec1_pred_y": predictions_rescaled[1],
                "pec2_org_x": original_landmarks_rescaled[2],
                "pec2_org_y": original_landmarks_rescaled[3],
                "pec2_pred_x": predictions_rescaled[2],
                "pec2_pred_y": predictions_rescaled[3],
            })


            for threshold in thresholds:
                # Assuming you define a function to calculate based on threshold:
                orig_automated_quality = is_point_inside_image_with_threshold(image_shape, perp_orig_endpoint,
                                                                     threshold_percentage=threshold)
                pred_quality = is_point_inside_image_with_threshold(image_shape, perp_pred_endpoint,
                                                                     threshold_percentage=threshold)

                correct_classification = int(orig_automated_quality == pred_quality)

                # Store prediction and automated_truth quality for later analysis
                threshold_metrics[threshold]['predictions'].append(pred_quality)
                threshold_metrics[threshold]['truths'].append(orig_automated_quality)

                # Update counts based on prediction and automated_truth classifications
                if pred_quality:
                    threshold_metrics[threshold]['total_good_pred'] += 1
                else:
                    threshold_metrics[threshold]['total_bad_pred'] += 1
                
                if orig_automated_quality:
                    threshold_metrics[threshold]['total_good_truth'] += 1
                else:
                    threshold_metrics[threshold]['total_bad_truth'] += 1

                # Update counts for correct classifications
                if correct_classification:
                    if orig_automated_quality:
                        threshold_metrics[threshold]['correct_good_count'] += 1
                    else:
                        threshold_metrics[threshold]['correct_bad_count'] += 1

    # Calculate overall automated_accuracy and distances statistics
    total_predictions = len(evaluation_results)
    overall_automated_accuracy = (correct_automated_predictions / total_predictions) * 100
    overall_manual_accuracy = (correct_manual_predictions / total_predictions) * 100

    perp_distance_mean_std = (np.mean(perpendicular_distances), np.std(perpendicular_distances))
    pec1_distance_mean_std = (np.mean(pec1_distances), np.std(pec1_distances))
    pec2_distance_mean_std = (np.mean(pec2_distances), np.std(pec2_distances))
    nipple_distance_mean_std = (np.mean(nipple_distances), np.std(nipple_distances))
    angular_distance_mean_std = (np.mean(angular_distances), np.std(angular_distances), np.median(angular_distances))

    # calc mm distances
    mm_pec1_mean_std = (np.mean(mm_distances_pec1), np.std(mm_distances_pec1), np.median(mm_distances_pec1))
    mm_pec2_mean_std = (np.mean(mm_distances_pec2), np.std(mm_distances_pec2), np.median(mm_distances_pec2))
    mm_nipple_mean_std = (np.mean(mm_distances_nipple), np.std(mm_distances_nipple), np.median(mm_distances_nipple))
    mm_perpendicular_mean_std = (np.mean(mm_distances_perpendicular), np.std(mm_distances_perpendicular), np.median(mm_distances_perpendicular))

    a_sensitivity, a_specificity = calculate_sensitivity_specificity(np.array(pred), np.array(automated_truth))
    m_sensitivity, m_specificity = calculate_sensitivity_specificity(np.array(pred), np.array(manual_truth))

    # Prepare summary statistics for saving
    summary_stats = pd.DataFrame({
        "Metric": ["Overall Automated Accuracy", "Overall Manual Accuracy", "mm_perpendicular Mean (Std)",
                    "mm_pec1 Mean (Std)", "mm_pec2 Mean (Std)", "mm_nipple Mean (Std)", "Angular Distance Mean (Std)",
                    "Perpendicular Distance Mean (Std)", "Pec1 Distance Mean (Std)",
                    "Pec2 Distance Mean (Std)", "Nipple Distance Mean (Std)"],
        "Value": [f"{overall_automated_accuracy:.2f}% (Sensitivity: {a_sensitivity* 100:.2f}, Specificity: {a_specificity* 100:.2f})",
                    f"{overall_manual_accuracy:.2f}% (Sensitivity: {m_sensitivity* 100:.2f}, Specificity: {m_specificity* 100:.2f})",
                    f"{mm_perpendicular_mean_std[0]:.2f} ({mm_perpendicular_mean_std[1]:.2f}), Median: {mm_perpendicular_mean_std[2]:.2f}",
                    f"{mm_pec1_mean_std[0]:.2f} ({mm_pec1_mean_std[1]:.2f}), Median: {mm_pec1_mean_std[2]:.2f}",
                    f"{mm_pec2_mean_std[0]:.2f} ({mm_pec2_mean_std[1]:.2f}), Median: {mm_pec2_mean_std[2]:.2f}",
                    f"{mm_nipple_mean_std[0]:.2f} ({mm_nipple_mean_std[1]:.2f}), Median: {mm_nipple_mean_std[2]:.2f}",
                    f"{angular_distance_mean_std[0]:.2f} ({angular_distance_mean_std[1]:.2f}), Median: {angular_distance_mean_std[2]:.2f}",
                  f"{perp_distance_mean_std[0]:.2f} ({perp_distance_mean_std[1]:.2f})",
                  f"{pec1_distance_mean_std[0]:.2f} ({pec1_distance_mean_std[1]:.2f})",
                  f"{pec2_distance_mean_std[0]:.2f} ({pec2_distance_mean_std[1]:.2f})",
                  f"{nipple_distance_mean_std[0]:.2f} ({nipple_distance_mean_std[1]:.2f})"]
                  
    })
    summary_stats.to_csv('evaluation_summary_stats.csv', index=False)

    # Convert evaluation results to DataFrame and save to CSV
    results_df = pd.DataFrame(evaluation_results)
    cols = ["image_index", "study_uid", "sop_uid", "automated_accuracy", "manual_accuracy",
            "automated_label", "manual_label","prediction_label",
            "perpendicular_distance", "mm_perpendicular",
            "pec1_distance", "mm_pec1", "pec2_distance", "mm_pec2", "nipple_distance", "mm_nipple",
            "angular_distance", "original_angle", "predicted_angle", "perp_org_x", "perp_org_y",
            "perp_pred_x", "perp_pred_y",
            "pec1_org_x", "pec1_org_y", "pec1_pred_x", "pec1_pred_y",
            "pec2_org_x", "pec2_org_y", "pec2_pred_x", "pec2_pred_y"]  
    results_df = results_df[cols]
    results_df.to_csv('evaluation_results_with_accuracy_and_distances.csv', index=False)
    
    print("Evaluation complete. Results saved.")


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    unified_df = preprocess_data(config)
    dataloaders, test_df = create_dataloaders(unified_df, config, return_test_df=True)  # Set this to True if needed
    test_loader = dataloaders['Test']

    model = load_model(config['model_type'], config['best_model_path'], device)
    evaluate_and_label(config, model, test_loader, test_df)  # Pass test_df when evaluating

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Model Evaluation Script")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration JSON file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    main(config)
