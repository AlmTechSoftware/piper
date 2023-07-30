import torch
from pycocotools import coco
import torch.nn as nn
import os
import json
import numpy as np
from PIL import Image


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
        # Load COCO format annotations
        coco_file_path = os.path.join(dataset_dir, "_annotations.coco.json")
        coco_data = json.load(open(coco_file_path, "r"))
        coco_data = coco.COCO(coco_file_path)

        image_ids = list(coco_data.imgs.keys())
        num_images = len(image_ids)

        images = []
        masks = []

        for idx in range(num_images):
            image_info = coco_data.loadImgs(image_ids[idx])[0]
            image_path = os.path.join(dataset_dir, "train", image_info["file_name"])

            # Load and preprocess the image
            image = Image.open(image_path)
            image = image.resize((self.input_shape[1], self.input_shape[2]))
            image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
            images.append(image)

            # Load and preprocess the segmentation mask
            ann_ids = coco_data.getAnnIds(imgIds=image_ids[idx], iscrowd=None)
            anns = coco_data.loadAnns(ann_ids)
            mask = coco_data.annToMask(anns[0])
            for i in range(1, len(anns)):
                mask += coco_data.annToMask(anns[i])
            mask = Image.fromarray(mask.astype(np.uint8))
            mask = mask.resize((self.input_shape[1], self.input_shape[2]))
            mask = torch.tensor(np.array(mask)).unsqueeze(0).float() / 255.0
            masks.append(mask)

        images = torch.stack(images)
        masks = torch.stack(masks)

        return images, masks

    def train(self, dataset_dir, epochs=10, batch_size=32):
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        images, masks = self.load_data(dataset_dir)

        for epoch in range(epochs):
            for i in range(0, len(images), batch_size):
                batch_images = images[i : i + batch_size]
                batch_masks = masks[i : i + batch_size]

                optimizer.zero_grad()
                outputs = self.forward(batch_images)
                loss = criterion(outputs, batch_masks)
                loss.backward()
                optimizer.step()

                print(
                    f"Epoch [{epoch + 1}/{epochs}], Batch [{i//batch_size + 1}/{len(images)//batch_size}], Loss: {loss.item()}"
                )


# Usage:
model = FeynmanModel()
model.train(dataset_dir="./dataset/train/", epochs=10, batch_size=32)
