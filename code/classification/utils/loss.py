import torch.nn as nn

class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.loss = nn.BCELoss()

    def forward(self, outputs, targets):
        return self.loss(outputs, targets)
    
class CategoricalCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CategoricalCrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        # Ensure targets are of type long since CrossEntropyLoss expects class indices
        targets = targets.long()
        return self.loss(outputs, targets)