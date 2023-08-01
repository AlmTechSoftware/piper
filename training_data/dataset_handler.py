from torch.utils.data import Dataset
from torchvision import transforms
from pycocotools import coco
import numpy as np
import os
import json
from PIL import Image
import torch


class COCODataset(Dataset):
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    # NOTE: ImageNet
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.coco_data = self.load_coco_data()

    def load_coco_data(self):
        coco_file_path = os.path.join(self.dataset_dir, "_annotations.coco.json")
        # coco_data = json.load(open(coco_file_path, "r"))
        return coco.COCO(coco_file_path)

    def __len__(self):
        return len(self.coco_data.imgs)

    def __getitem__(self, idx):
        image_info = list(self.coco_data.imgs.values())[idx]
        image_path = os.path.join(self.dataset_dir, image_info["file_name"])

        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # Load and preprocess the segmentation mask
        ann_ids = self.coco_data.getAnnIds(imgIds=image_info["id"], iscrowd=None)
        mask = np.zeros((image_info["height"], image_info["width"]))
        for ann_id in ann_ids:
            mask += self.coco_data.annToMask(self.coco_data.anns[ann_id])
        mask = torch.tensor(mask).unsqueeze(0).float() / 255.0

        return image, mask
