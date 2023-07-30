#!/usr/bin/env python

import argparse
import tensorflow as tf
import numpy as np
import cv2
import os

from fcn_model import FeynmanModel
from training_data.dataset_handler import COCOJsonUtility, load_coco_json

# from dataset_handler import *

ANNOTATIONS_FILE_NAME = "_annotations.coco.json"


def train_model(
    model: FeynmanModel,
    coco_data,
    epochs=100,
    batch_size=32,
):
    # Images that will be used to train the model
    train_images = [image.file_name for image in coco_data.images]

    # load dataset annotations
    segments = []
    segments = COCOJsonUtility.get_annotations_by_image_path(
        coco_data=coco_data, image_path=EXAMPLE_IMAGE_NAME
    )
    ground_truth = COCOJsonUtility.annotations2detections(annotations=annotations)

    # NOTE: fix
    ground_truth.class_id = ground_truth.class_id - 1

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix, save_weights_only=True, save_best_only=True
    )

    model.fit(
        train_images,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint_callback],
        verbose=1,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        help="Training dataset path",
        default="dataset/train/",
    )
    parser.add_argument(
        "--epochs", help="Number of training epochs", default=10, type=int
    )
    parser.add_argument("--batch", help="Batch size", default=64, type=int)
    args = parser.parse_args()

    print("\n" * 4)
    print("TRAINING BEGIN")
    print("\n" * 4)

    # train_images_path = os.path.join(dataset_path, "train/")
    annotations_file = os.path.join(dataset_path, ANNOTATIONS_FILE_NAME)
    coco_data = load_coco_json(json_file=annotations_file)

    classes = [
        category.name
        for category in coco_data.categories
        if category.supercategory != "none"
    ]

    model = FeynmanModel(num_classes=len(classes))
    train_model(model, coco_data, epochs=args.epochs)
