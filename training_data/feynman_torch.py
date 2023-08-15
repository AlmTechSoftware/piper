import logging
import torch.nn as nn
import torch.optim as optim

from torch import Tensor

# import numpy as np

from torch.utils.data import DataLoader

from dataset_handler import COCODataset
from segnet import *


# Model definition. We use a SegNet-Basic model with some minor tweaks.
# Our input images are 128x128.
class FeynmanModel(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.out_channels = 3
        self.bn_input = nn.BatchNorm2d(3)

        # Encoder
        self.dc1 = DownConv2(3, 64, kernel_size=kernel_size)
        self.dc2 = DownConv2(64, 128, kernel_size=kernel_size)
        self.dc3 = DownConv3(128, 256, kernel_size=kernel_size)
        self.dc4 = DownConv3(256, 512, kernel_size=kernel_size)

        # Decoder
        self.uc4 = UpConv3(512, 256, kernel_size=kernel_size)
        self.uc3 = UpConv3(256, 128, kernel_size=kernel_size)
        self.uc2 = UpConv2(128, 64, kernel_size=kernel_size)
        self.uc1 = UpConv2(64, 3, kernel_size=kernel_size)

    def forward(self, batch: Tensor):
        x = self.bn_input(batch)
        x, mp1_indices, shape1 = self.dc1(x)
        x, mp2_indices, shape2 = self.dc2(x)
        x, mp3_indices, shape3 = self.dc3(x)
        x, mp4_indices, shape4 = self.dc4(x)
        # x, mp5_indices, shape5 = self.dc5(x)

        # SegNet Decoder
        # x = self.uc5(x, mp5_indices, output_size=shape5)
        x = self.uc4(x, mp4_indices, output_size=shape4)
        x = self.uc3(x, mp3_indices, output_size=shape3)
        x = self.uc2(x, mp2_indices, output_size=shape2)
        x = self.uc1(x, mp1_indices, output_size=shape1)

        return x

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
                logging.debug(f"{epoch+1} - {i+1}")
                optimizer.zero_grad()
                outputs = self.forward(batch_images)
                loss = criterion(outputs, batch_masks)
                loss.backward()
                optimizer.step()

                logging.debug(
                    f"\tEpoch [{epoch + 1}/{epochs}], Batch [{i + 1}/{len(dataloader)}], Loss: {loss.item()}"
                )
