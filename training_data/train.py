#!/usr/bin/env python

import logging
from torch.utils.data import DataLoader
from feynman_torch import FeynmanModel

from colored import Fore, Back, Style

import torch
import torch.nn as nn
import torch.optim as optim

from dataset_handler import COCODataset


logging.basicConfig(
    level=logging.DEBUG,
    format=f"{Back.rgb(80, 200, 80)} FeynMAN {Style.reset} [{Fore.rgb(255, 240, 240)}%(levelname)s{Style.reset}] %(message)s",
)


def load_data(dataset_dir: str):
    # Load the COCO dataset
    dataset = COCODataset(dataset_dir=dataset_dir)
    return dataset


def train_model(
    model: nn.Module,
    dataset_dir: str,
    epochs: int = 10,
    batch_size: int = 32,
):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    dataset = load_data(dataset_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, masks) in enumerate(dataloader):
            print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(dataloader)}]")
            # images, masks = images.to(device), masks.to(device).float()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks.squeeze(1))

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            print(f" - Loss: {running_loss/10:.4f}")
            running_loss = 0.0


def main():
    # Usage:
    logging.info("Creating model...")
    model = FeynmanModel(3).cuda()

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    # device_type = "cpu"
    logging.debug(f"Doing training on device type '{device_type}'!")
    device = torch.device(device_type)

    model = model.to(device)

    logging.info("START TRAINING!")
    train_model(model, dataset_dir="./dataset/train/", epochs=10, batch_size=2)
    logging.info("END TRAINING!")


if __name__ == "__main__":
    main()
