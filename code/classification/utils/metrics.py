# metrics.py

import os
import csv
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt

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

def evaluate_model(model, loader, device, num_classes, metrics_csv, results_csv, annotation_path):
    all_preds, all_labels = [], []
    model.eval()

    # Load annotations for UID mapping and filter by "Test" split
    annotations = pd.read_csv(annotation_path)
    test_annotations = annotations[annotations['Split'] == 'Test']
    study_uids = test_annotations['StudyInstanceUID'].tolist()
    sop_uids = test_annotations['SOPInstanceUID'].tolist()
    ground_truths = test_annotations['manual_quality'].tolist()

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy, f1, sensitivity, specificity, auc = compute_metrics(np.array(all_preds), np.array(all_labels), num_classes)
    save_metrics_csv([accuracy, f1, sensitivity, specificity, auc], metrics_csv)
    save_predictions_csv(all_preds, all_labels, study_uids, sop_uids, results_csv)

def compute_metrics(predictions, truths, num_classes):
    accuracy = accuracy_score(truths, predictions) * 100
    f1 = f1_score(truths, predictions, average='macro') * 100
    sensitivity, specificity = calculate_sensitivity_specificity(predictions, truths)
    sensitivity, specificity = sensitivity * 100, specificity * 100
    auc = roc_auc_score(truths, predictions) * 100 if num_classes == 2 else 'N/A'
    return accuracy, f1, sensitivity, specificity, auc

def save_metrics_csv(metrics, file_path):
    fieldnames = ["Accuracy", "F1 Score", "Sensitivity", "Specificity", "AUC"]
    with open(file_path, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "Accuracy": f"{metrics[0]:.2f}",
            "F1 Score": f"{metrics[1]:.2f}",
            "Sensitivity": f"{metrics[2]:.2f}",
            "Specificity": f"{metrics[3]:.2f}",
            "AUC": f"{metrics[4]:.2f}" if metrics[4] != 'N/A' else metrics[4]
        })
    print(f"Metrics saved to {file_path}")

def save_predictions_csv(predictions, truths, study_uids, sop_uids, file_path):
    rows = [
        {"index": i+1, "StudyInstanceUID": study_uids[i], "SOPInstanceUID": sop_uids[i], "Ground Truth": truths[i], "Prediction": predictions[i]}
        for i in range(len(predictions))
    ]
    with open(file_path, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["index", "StudyInstanceUID", "SOPInstanceUID", "Ground Truth", "Prediction"])
        writer.writeheader()
        writer.writerows(rows)  # Write all rows in one go
    print(f"Predictions saved to {file_path}")
    

def compute_metrics(predictions, truths, num_classes):
    accuracy = accuracy_score(truths, predictions) * 100
    f1 = f1_score(truths, predictions, average='macro') * 100
    sensitivity, specificity = calculate_sensitivity_specificity(predictions, truths)
    sensitivity, specificity = sensitivity * 100, specificity * 100
    auc = roc_auc_score(truths, predictions) * 100 if num_classes == 2 else 'N/A'
    return accuracy, f1, sensitivity, specificity, auc

def save_metrics_csv(metrics, file_path):
    fieldnames = ["Accuracy", "F1 Score", "Sensitivity", "Specificity", "AUC"]
    with open(file_path, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "Accuracy": f"{metrics[0]:.2f}",
            "F1 Score": f"{metrics[1]:.2f}",
            "Sensitivity": f"{metrics[2]:.2f}",
            "Specificity": f"{metrics[3]:.2f}",
            "AUC": f"{metrics[4]:.2f}" if metrics[4] != 'N/A' else metrics[4]
        })
    print(f"Metrics saved to {file_path}")


def gradcam_visualization(model, loader, device, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define the target layer you want to visualize
    target_layer = model.layer4[-1]

    # Setup Grad-CAM
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=device.type == 'cuda')

    for batch_idx, (images, _) in enumerate(loader):
        for idx, img_tensor in enumerate(images):
            img_tensor = img_tensor.to(device)

            # Normalize image for visualization
            img = img_tensor.cpu().numpy()
            img = img - img.min()
            img = img / img.max()
            
            # Ensure img is three-channel RGB
            if img.shape[0] == 1:  # If grayscale, convert to RGB by repeating the single channel
                img = np.tile(img, (3, 1, 1))
            img = np.transpose(img, (1, 2, 0))  # Change to HWC for visualization

            # Prepare input tensor for CAM
            input_tensor = img_tensor.unsqueeze(0)

            # Generate the CAM mask
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)
            grayscale_cam = grayscale_cam[0, :]  # Use the first CAM (corresponding to the first image in the batch)

            # Convert grayscale CAM to heatmap and overlay it on the image
            cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

            # Convert from RGB (matplotlib) to BGR (OpenCV)
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

            # Save the result
            output_path = os.path.join(output_dir, f"gradcam_{batch_idx * loader.batch_size + idx + 1}.png")
            cv2.imwrite(output_path, cam_image)

            if (batch_idx * loader.batch_size + idx) >= 199:  # Process only the first 20 images
                break
        if (batch_idx * loader.batch_size + idx) >= 199:
            break

    print(f"GradCAM visualizations saved to {output_dir}")