import torch
from utils.dataloader import get_dataloader
from torchvision import models
import torch.nn as nn
from utils.metrics import gradcam_visualization, evaluate_model

# change paths to your own
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        'device': device,
        'image_dir': '../data/images',
        'annotations_file': '../classification_data.csv',
        'batch_size': 8,
        'model_name': 'resnext50',
        'num_classes': 2,
        'label_type': 'automated' # automated, manual
    }

    # Use the 'Test' split
    test_loader = get_dataloader(config['image_dir'], config['annotations_file'], 'Test', config['label_type'], config['batch_size'])

    # Initialize the model with pretrained=False since we're loading your own weights
    model = models.resnext50_32x4d(pretrained=False)
    
    # Change the first layer to accept single channel images
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Change the final fully connected layer to match the number of classes you have
    model.fc = torch.nn.Linear(model.fc.in_features, config["num_classes"])
    
    # Move the model to the specified device
    model = model.to(device)

    # Load the model weights from a pre-saved checkpoint
    model.load_state_dict(torch.load('../classification/results/best_model.pth', map_location=device))

    # Perform GradCAM Visualization
    gradcam_visualization(model, test_loader, device, output_dir="gradcam_outs")

    # Evaluate and save metrics
    evaluate_model(model, test_loader, device, config['num_classes'], metrics_csv="test_metrics.csv",
                    results_csv="predictions.csv", annotation_path=config['annotations_file'])

if __name__ == "__main__":
    main()