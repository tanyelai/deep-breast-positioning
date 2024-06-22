# main.py
import argparse
import json
import pandas as pd
import torch
import os

from utils.dataloader import create_dataloaders, preprocess_data
from utils.train import Trainer
from utils.validate import Validator
from utils.models import UNet, RAUNet, CRAUNet, GuptaModel, ResNeXt50

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    config['device'] = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    return config

def main(config):
    print(f"Using device: {config['device']}")

    unified_df = preprocess_data(config)
    dataloaders = create_dataloaders(unified_df, config)

    train_loader, val_loader = dataloaders['Train'], dataloaders['Validation']

    # Determine the number of output features based on the target task
    out_features_map = {
        'pectoralis': 4,
        'nipple': 2,
        'all': 6
    }
    if config['target_task'] not in out_features_map:
        raise ValueError("Invalid target configuration")
    out_features = out_features_map[config['target_task']]

    # Instantiate the model based on the configuration
    if config['model_type'] == "UNet": # Vanilla UNet
        model = UNet(in_channels=1, out_features=out_features)
    elif config['model_type'] == "RAUNet": # Attention UNet
        model = RAUNet(in_channels=1, out_features=out_features)
    elif config['model_type'] == "CRAUNet": # CoordAttention UNet
        model = CRAUNet(in_channels=1, out_features=out_features)
    elif config['model_type'] == "ResNeXt50":
        model = ResNeXt50(in_channels=1, out_features=out_features)
    else:
        raise ValueError("Invalid model type specified")
    
    model = model.to(config['device'])
    
    trainer = Trainer(config, train_loader, model)
    validator = Validator(config, val_loader, model)

    best_model_state = None
    best_val_loss = float('inf')
    metrics = []

    for epoch in range(config['num_epochs']):
        train_loss = trainer.train(epoch)
        validation_loss, is_best = validator.validate()
        current_lr = trainer.scheduler.get_last_lr()[0]

        metrics.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'validation_loss': validation_loss,
            'learning_rate': current_lr
        }) 

        if is_best:
            best_model_state = model.state_dict()
            best_val_loss = validation_loss

    if best_model_state:
        os.makedirs(os.path.dirname(config['best_model_path']), exist_ok=True)
        torch.save(best_model_state, config['best_model_path'])
        print(f"Best model saved to {config['best_model_path']}, Validation Loss: {best_val_loss}")

    # Save the metrics to a CSV file
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv('training_metrics.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training experiments with different configurations")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration JSON file')
    
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)