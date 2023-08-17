import torch.nn as nn

import torchvision.models.segmentation as models


class FeynmanModel(nn.Module):
    def __init__(self, num_classes):
        super(__class__, self).__init__()
        self.backbone = models.fcn_resnet50(pretrained=False, progress=True)
        self.backbone.classifier[4] = nn.Conv2d(
            512, num_classes, kernel_size=(1, 1), stride=(1, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        res = self.backbone(x)["out"]
        res = self.sigmoid(res)
        return res
