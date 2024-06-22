# train.py

import torch
from tqdm import tqdm
from .loss import MultifacetedLoss
from torch.optim.lr_scheduler import CyclicLR

class Trainer:
    def __init__(self, config, train_loader, model=None):
        self.config = config
        self.train_loader = train_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.criterion = MultifacetedLoss(
            w=config['w'],
            epsilon=config['epsilon'],
            alpha=config['alpha'],
            beta=config['beta'],
            gamma=config['gamma'],
            delta=config['delta']
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.scheduler = CyclicLR(self.optimizer, 
                                  base_lr=config['base_lr'], 
                                  max_lr=config['max_lr'],
                                  step_size_down=config['step_size_down'], 
                                  mode='triangular',
                                  cycle_momentum=False)

    def train(self, epoch):
        self.model.train()
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["num_epochs"]}')
        total_loss = 0.0
        for images, landmarks in progress_bar:
            images, landmarks = images.to(self.device), landmarks.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, landmarks)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()  # Step the scheduler at the end of each batch 
            total_loss += loss.item()

            progress_bar.set_postfix(loss=loss.item())
        avg_loss = total_loss / len(self.train_loader)
        self.scheduler.step()  # Update the learning rate
        return avg_loss  # Return the average loss for this epoch