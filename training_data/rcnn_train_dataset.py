#!/usr/bin/env python

import argparse
from rcnn_model import FeynmanModel

NUM_CLASSES = 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", help="Data directory", default="dataset/train/images/"
    )
    parser.add_argument(
        "--labels", help="Labels directory", default="dataset/train/labels/"
    )
    parser.add_argument("--epochs", help="Number of training epochs", default=10)
    args = parser.parse_args()

    print("\n" * 4)
    print("TRAINING BEGIN")

    model = FeynmanModel(num_classes=NUM_CLASSES)
    model.build((640, 640, 3))
    model.train(
        images_dir=args.data,
        labels_dir=args.labels,
        num_epochs=int(args.epochs),
    )
