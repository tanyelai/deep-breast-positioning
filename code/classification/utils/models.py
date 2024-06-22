# models.py

import torch.nn as nn
import torch
from torchvision import models

def get_model(model_name, num_classes, pretrained=True):
    """
    Fetch a model and adapt its first convolutional layer to accept grayscale images.

    Args:
    model_name (str): The name of the model to retrieve.
    num_classes (int): The number of output classes for the model.
    pretrained (bool): Whether to load pretrained weights.

    Returns:
    torch.nn.Module: The modified model.
    """
    if "resnext50" in model_name:
        model = models.resnext50_32x4d(pretrained=pretrained)
        # Adjust the first convolutional layer for 1-channel input
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError("Unsupported model name")
    
    return model
