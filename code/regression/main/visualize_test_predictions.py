# visualize_test_predictions.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.models import UNet, RAUNet, CRAUNet, ResNeXt50
from utils.dataloader import create_dataloaders
import os
import pandas as pd
import matplotlib.patches as patches

def load_model(model_path, device):
    model = UNet(in_channels=1, out_features=6).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def visualize_prediction(image, predicted_landmarks, original_landmarks, draw=True,
                          title="Prediction vs Original", threshold_percentage=0):
    """
    Visualizes a single image with both predicted and original landmarks. Optionally, draws lines between
    two line landmarks and a bounding box around the nipple landmark for both prediction and ground truth.
    Also visualizes the threshold-based safe zone around the image.
    """
    img_height, img_width = image.shape[2:]  # Assuming image is in (C, H, W) format
    longest_dimension = max(img_height, img_width)
    margin = (threshold_percentage / 100.0) * longest_dimension  # Calculate the safe zone margin

    
    # Correctly rescale the landmarks
    # Reshape the landmarks for easier manipulation
    predicted_landmarks = np.reshape(predicted_landmarks, (-1, 2))  # Reshape to (N, 2) where N is number of landmarks
    original_landmarks = np.reshape(original_landmarks, (-1, 2))
    
    # Scale the x coordinates
    predicted_landmarks[:, 0] *= img_width
    original_landmarks[:, 0] *= img_width
    
    # Scale the y coordinates
    predicted_landmarks[:, 1] *= img_height
    original_landmarks[:, 1] *= img_height
    
    # Flatten back if needed for the rest of your function
    predicted_landmarks = predicted_landmarks.flatten()
    original_landmarks = original_landmarks.flatten()

    plt.imshow(image.squeeze(), cmap='gray')
    plt.scatter(predicted_landmarks[::2], predicted_landmarks[1::2], s=10, marker='.', c='r', label='Predicted')
    plt.scatter(original_landmarks[::2], original_landmarks[1::2], s=10, marker='x', c='b', label='Original')
    
    if draw:
        # Draw lines between the first two landmarks
        plt.plot(predicted_landmarks[:4:2], predicted_landmarks[1:4:2], 'r-', linewidth=2)
        plt.plot(original_landmarks[:4:2], original_landmarks[1:4:2], 'b-', linewidth=2)
        
        # Bounding boxes around nipple landmarks
        bbox_size = 10
        for landmarks, color in [(predicted_landmarks, 'r'), (original_landmarks, 'b')]:
            nipple_box = patches.Rectangle((landmarks[4]-bbox_size/2, landmarks[5]-bbox_size/2),
                                           bbox_size, bbox_size, linewidth=1, edgecolor=color, facecolor='none')
            plt.gca().add_patch(nipple_box)
            
            # Calculate and draw the vertical line from nipple to the line between first two landmarks
            x1, y1, x2, y2, xn, yn = landmarks[:6]
            if x2 != x1:  # Avoid division by zero for non-vertical lines
                m = (y2 - y1) / (x2 - x1)  # Slope of the original line
                c = y1 - m * x1  # Intercept of the original line
                perp_m = -1 / m  # Slope of the perpendicular line
                
                # Since the perpendicular line goes through (xn, yn), we can calculate its intercept (perp_c)
                perp_c = yn - perp_m * xn
                
                # Find intersection point (inter_x, inter_y) between the original and perpendicular lines
                inter_x = (perp_c - c) / (m - perp_m)
                inter_y = m * inter_x + c
                
                plt.plot([xn, inter_x], [yn, inter_y], color+'--', linewidth=1)
            else:
                # If the original line is vertical, draw a horizontal line from the nipple coordinate
                plt.plot([xn, xn], [yn, y1], color+'--', linewidth=1)  # Use y1 or y2 since both are on the vertical line

    # Draw the safe zone boundary. Adjust the coordinates and dimensions to account for the margin
    #safe_zone = patches.Rectangle((-margin, -margin), img_width + 2*margin, img_height + 2*margin, linewidth=1, edgecolor='green', facecolor='none', linestyle='--', label='Safe Zone')
    #plt.gca().add_patch(safe_zone)

    #plt.title(title)
    plt.legend()
    plt.axis('off')
    plt.tight_layout()

    predictions_dir = 'predictions'
    os.makedirs(predictions_dir, exist_ok=True)
    plt.savefig(os.path.join(predictions_dir, f"{title}.png"), dpi=300, bbox_inches='tight')
    plt.close()



def preprocess_data(config):
    # Assuming this function prepares your data and it's correctly implemented in your project.
    split_df = pd.read_csv(config['split_file'])
    details_df = pd.read_csv(config['details_file'])
    unified_df = pd.merge(split_df, details_df, on='SOPInstanceUID', how='left')
    return unified_df

def test_model(config, model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for i, (image, original_landmarks) in enumerate(test_loader):
            image = image.to(device)
            predictions = model(image).cpu().numpy().flatten()
            original_landmarks = original_landmarks.numpy().flatten()

            if i < 200:  # Adjust this number to control how many predictions you want to visualize
                visualize_prediction(image.cpu().numpy(), predictions, original_landmarks, title=f"Test Sample {i+1}")
            else:
                break

def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Preprocess and merge data
    unified_df = preprocess_data(config)

    # Create dataloaders
    dataloaders = create_dataloaders(unified_df, config)
    test_loader = dataloaders['Test']

    # Load the model
    model = load_model(config['best_model_path'], device)
    
    # Test the model
    test_model(config, model, test_loader)


# change paths to your own
if __name__ == "__main__":
    config = {
        'split_file': '../unified_data_with_splits.csv',
        'details_file': '../transformation_details.csv',
        'base_image_dir': '../images',
        'best_model_path': '../UNet_baseline.pth',
        'batch_size': 1, 
        'target_task': 'all'
    }
    main(config)
