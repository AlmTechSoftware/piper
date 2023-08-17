import os
from typing import Tuple
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pycocotools.coco as coco


class COCODataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        img_size: Tuple[int, int] = (640, 640),
    ):
        self.dataset_dir = dataset_dir
        self.transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    # NOTE: ImageNet
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.trafos = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
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
        mask = self.trafos(mask.numpy())

        print("MASK", mask, mask.size())

        return image, mask
