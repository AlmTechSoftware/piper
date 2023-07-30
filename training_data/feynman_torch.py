import torch
import torch.nn as nn
import torch.nn as nn
import torch.optim as optim

import numpy as np

from torch.utils.data import DataLoader

from dataset_handler import COCODataset


class FeynmanModel(nn.Module):
    def __init__(self):
        super(FeynmanModel, self).__init__()
        self.model = self.build_model()

    def build_model(self):
        # Encoder
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # Decoder
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv9 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv11 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv13 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv14 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Output
        self.output = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = nn.ReLU()(self.conv1(x))
        x2 = nn.ReLU()(self.conv2(x1))
        pool1 = nn.MaxPool2d(kernel_size=2, stride=2)(x2)

        x3 = nn.ReLU()(self.conv3(pool1))
        x4 = nn.ReLU()(self.conv4(x3))
        pool2 = nn.MaxPool2d(kernel_size=2, stride=2)(x4)

        x5 = nn.ReLU()(self.conv5(pool2))
        x6 = nn.ReLU()(self.conv6(x5))
        pool3 = nn.MaxPool2d(kernel_size=2, stride=2)(x6)

        # Bottleneck
        x7 = nn.ReLU()(self.conv7(pool3))
        x8 = nn.ReLU()(self.conv8(x7))

        # Decoder
        up1 = self.up1(x8)
        concat1 = torch.cat((up1, x6), dim=1)
        x9 = nn.ReLU()(self.conv9(concat1))
        x10 = nn.ReLU()(self.conv10(x9))

        up2 = self.up2(x10)
        concat2 = torch.cat((up2, x4), dim=1)
        x11 = nn.ReLU()(self.conv11(concat2))
        x12 = nn.ReLU()(self.conv12(x11))

        up3 = self.up3(x12)
        concat3 = torch.cat((up3, x2), dim=1)
        x13 = nn.ReLU()(self.conv13(concat3))
        x14 = nn.ReLU()(self.conv14(x13))

        # Output
        output = nn.Sigmoid()(self.output(x14))

        return output

    def load_data(self, dataset_dir: str):
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
model = FeynmanModel()
model.train(dataset_dir="./dataset/train/", epochs=10, batch_size=32)
