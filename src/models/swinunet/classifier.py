import os, sys
sys.path.append(os.path.abspath(os.curdir))

import torch 
import torch.nn as nn

class SwinUnetClassifier(nn.Module):
    def __init__(self, encoder, num_classes=4, feature_dim=768):
        super().__init__()
        self.encoder = encoder
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )
        
        self.image_projection = nn.Linear(768, feature_dim)
        
    def forward(self, x):
        bottleneck, _ = self.encoder(x)
        bottleneck = bottleneck.permute(0, 3, 1, 2)
        # print("Bottleneck shape:", bottleneck.shape)  # Nên là (16, 8, 8, 768)
        logits = self.classifier(bottleneck)
        # print("Logits shape:", logits.shape)  # Nên là (16, 3)
        
        features = nn.functional.adaptive_avg_pool2d(bottleneck, 1).flatten(1)
        # print("Features shape:", features.shape)  # Nên là (16, 768)
        image_features = self.image_projection(features)
        # print("Image features shape:", image_features.shape)  # Nên là (16, 768)
        
        return logits, image_features