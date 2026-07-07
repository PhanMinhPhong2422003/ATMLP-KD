import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, width, height, depth, classes,feature_dim=128):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=depth, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.flatten = nn.Flatten()
        final_width = width // 4
        final_height = height // 4
        self.linear_input_size = 64 * final_width * final_height  
        self.fc1 = nn.Linear(self.linear_input_size, feature_dim)
        self.bn3 = nn.BatchNorm1d(feature_dim) 
        self.dropout = nn.Dropout(0.5)   
        self.fc2 = nn.Linear(feature_dim, classes)
    def forward(self, x, return_features=False):
        
        # Block 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        
        # Block 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        
        # Flatten
        x = self.flatten(x)
        
        # --- Feature Extraction Part ---
        x = self.fc1(x)
        x = F.relu(x)
        features = self.bn3(x) 

        x_drop = self.dropout(features)
        logits = self.fc2(x_drop)
        
        if return_features:
            return logits, features
        return logits



