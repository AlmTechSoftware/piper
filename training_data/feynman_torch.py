import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from torch.utils.data import DataLoader

from dataset_handler import COCODataset


class FeynmanModel(nn.Module):
    def __init__(self, num_classes: int = 3):
        super(self.__class__, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Middle
        self.middle = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        
    def forward(self, x):
        # Encoder
        x1 = self.encoder(x)
        
        # Middle
        x2 = self.middle(x1)
        
        # Decoder
        x3 = self.decoder(x2)
        
        # Upsample to original size
        x3 = nn.functional.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        
        return x3

    @staticmethod
    def load_data(dataset_dir: str):
        # Load the COCO dataset
        dataset = COCODataset(dataset_dir=dataset_dir)
        return dataset

    def train(self, dataset_dir: str, epochs=10, batch_size=32):
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        dataset = self.load_data(dataset_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for i, (batch_images, batch_masks) in enumerate(dataloader):
                optimizer.zero_grad()
                outputs = self.forward(batch_images)
                loss = criterion(outputs, batch_masks)
                loss.backward()
                optimizer.step()

                print(
                    f"Epoch [{epoch + 1}/{epochs}], Batch [{i + 1}/{len(dataloader)}], Loss: {loss.item()}"
                )


# Usage:
model = FeynmanModel(3)
model.train(dataset_dir="./dataset/train/", epochs=10, batch_size=32)
