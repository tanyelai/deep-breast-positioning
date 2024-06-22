# validate.py

import numpy as np
import torch
from tqdm import tqdm
from .loss import CategoricalCrossEntropyLoss
from sklearn.metrics import f1_score, precision_score, roc_auc_score, accuracy_score
from utils.metrics import calculate_sensitivity_specificity

class Validator:
    def __init__(self, config, val_loader, model):
        self.config = config
        self.val_loader = val_loader
        self.model = model.to(config['device'])
        self.device = config['device']
        self.criterion = CategoricalCrossEntropyLoss().to(self.device)
        self.best_val_f1 = 0.0

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        all_targets, all_outputs = [], []
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc='Validation')
            for images, targets in progress_bar:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                # Convert logits to predicted class indices
                predicted_classes = torch.argmax(outputs, dim=1)
                all_targets.extend(targets.cpu().numpy())
                all_outputs.extend(predicted_classes.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)

        # Calculate other metrics
        f1 = f1_score(all_targets, all_outputs, average='weighted')
        precision = precision_score(all_targets, all_outputs, average='weighted')
        accuracy = accuracy_score(all_targets, all_outputs)


        # Calculate sensitivity and specificity using the new function
        sensitivity, specificity = calculate_sensitivity_specificity(np.array(all_outputs), np.array(all_targets))


        try:
            auc = roc_auc_score(all_targets, all_outputs, multi_class='ovo')
        except ValueError:
            auc = float('nan')  # If only one class is present in y_true

        # Determine if this validation result is the best so far
        is_best = f1 > self.best_val_f1
        if is_best:
            self.best_val_f1 = f1

        return avg_loss, f1, accuracy, precision, sensitivity, specificity, auc, is_best
