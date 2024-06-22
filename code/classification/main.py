# main.py

"""
The `classification_data.csv` file is manually generated adhering to the PNL (Perpendicular Normative Line) rule,
which utilizes perpendicular coordinates combined with image-level quality labels.

Columns in the CSV are as follows:
StudyInstanceUID, SOPInstanceUID, Side, endpoint_x, endpoint_y, automated_quality, manual_quality, Split, adjusted_landmarks

Users need to create this file using the labels we provide.
- `automated_quality` denotes labels automatically derived from the radiologist's annotations, applying PNL criteria.
- `manual_quality` refers to additional qualitative evaluations by radiologists on the overall image quality.

Feel free to modify these variable names as needed.
"""

import torch
import os
import pandas as pd
import numpy as np
from utils.dataloader import get_dataloader
from utils.train import Trainer
from utils.validate import Validator
from utils.models import get_model

def main(config):
    print(f"Using device: {config['device']}")

    # Prepare data loaders
    image_dir = config['image_dir']
    annotations_file = config['annotations_file']
    train_loader = get_dataloader(image_dir, annotations_file, 'Train', label_type=config['label_type'], batch_size=config['batch_size'])
    val_loader = get_dataloader(image_dir, annotations_file, 'Validation', label_type=config['label_type'], batch_size=config['batch_size'])

    # Initialize model
    model = get_model(config['model_name'], config['num_classes']).to(config['device'])
    
    # Initialize training and validation handlers
    trainer = Trainer(config, train_loader, model)
    validator = Validator(config, val_loader, model)

    metrics = []
    best_model_state = None
    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(config['num_epochs']):
        # Unpack all returned metrics from the trainer and validator
        train_loss, train_f1, train_accuracy, train_precision, train_sensitivity, train_specificity, train_auc = trainer.train(epoch)
        validation_loss, val_f1, val_accuracy, val_precision, val_sensitivity, val_specificity, val_auc, is_best = validator.validate()

        metrics.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'validation_loss': validation_loss,
            'train_f1': train_f1,
            'val_f1': val_f1,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'train_precision': train_precision,
            'val_precision': val_precision,
            'train_sensitivity': train_sensitivity,
            'val_sensitivity': val_sensitivity,
            'train_specificity': train_specificity,
            'val_specificity': val_specificity,
            'train_auc': train_auc,
            'val_auc': val_auc
        })

        print(f"Epoch {epoch + 1}/{config['num_epochs']}, Train Loss: {train_loss}, Validation Loss: {validation_loss}")
        print(f"Train Metrics - F1: {train_f1}, Accuracy: {train_accuracy}, Precision: {train_precision}, "
              f"Sensitivity: {train_sensitivity}, Specificity: {train_specificity}, AUC: {train_auc}")
        print(f"Validation Metrics - F1: {val_f1}, Accuracy: {val_accuracy}, Precision: {val_precision}, "
              f"Sensitivity: {val_sensitivity}, Specificity: {val_specificity}, AUC: {val_auc}")

        if is_best:
            best_model_state = model.state_dict()
            best_val_loss = validation_loss
            print(f"Best model found at epoch {epoch + 1}, Validation Loss: {best_val_loss} and F1 score: {val_f1}")
            best_epoch = epoch + 1

    if best_model_state:
        os.makedirs(os.path.dirname(config['best_model_path']), exist_ok=True)
        torch.save(best_model_state, config['best_model_path'])
        print(f"Best model saved to {config['best_model_path']}, Validation Loss: {best_val_loss}")
        print(f"Best model found at epoch {best_epoch}")
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv('training_metrics.csv', index=False)



# change paths to your own

if __name__ == "__main__":
    config = {
        'model_name': 'resnext50',  # or 'resnet50', 'efficientnetv2', etc.
        'label_type': 'automated',  # 'manual or 'automated'
        'num_classes': 2,
        'batch_size': 8,
        'num_epochs': 30,
        'base_lr': 1e-5,
        'max_lr': 5e-4,
        'step_size_down': 10,
        'learning_rate': 1e-4,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'image_dir': '../data/images',
        'annotations_file': '../annotations/classification_data.csv',
        'best_model_path': '../best_model.pth'
    }
    main(config)