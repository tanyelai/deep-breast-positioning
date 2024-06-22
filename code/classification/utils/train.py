# train.py

import numpy as np
import torch
from tqdm import tqdm
from utils.loss import CategoricalCrossEntropyLoss
from torch.optim.lr_scheduler import CyclicLR
from sklearn.metrics import f1_score, precision_score, roc_auc_score, accuracy_score
from utils.metrics import calculate_sensitivity_specificity

class Trainer:
    def __init__(self, config, train_loader, model):
        self.config = config
        self.train_loader = train_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = CategoricalCrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.scheduler = CyclicLR(self.optimizer, base_lr=config['base_lr'], max_lr=config['max_lr'],
                                  step_size_down=config['step_size_down'], mode='triangular', cycle_momentum=False)

    def train(self, epoch):
        self.model.train()
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.config["num_epochs"]}')
        total_loss = 0.0
        all_targets, all_outputs = [], []
        for images, targets in progress_bar:
            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()  # Update the learning rate
            total_loss += loss.item()

            # Use argmax to extract the predicted class indices
            predicted_classes = torch.argmax(outputs, dim=1)
            all_targets.extend(targets.detach().cpu().numpy())
            all_outputs.extend(predicted_classes.detach().cpu().numpy())
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(self.train_loader)

        # Calculate other metrics
        f1 = f1_score(all_targets, all_outputs, average='weighted')
        precision = precision_score(all_targets, all_outputs, average='weighted')
        accuracy = accuracy_score(all_targets, all_outputs)

        # Calculate sensitivity and specificity using the new function
        sensitivity, specificity = calculate_sensitivity_specificity(np.array(all_outputs), np.array(all_targets))

        try:
            auc = roc_auc_score(all_targets, all_outputs, multi_class='ovo')
        except ValueError:
            auc = float('nan')  # If only one class present in y_true

        return avg_loss, f1, accuracy, precision, sensitivity, specificity, auc
