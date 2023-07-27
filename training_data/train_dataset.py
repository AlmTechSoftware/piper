#!/usr/bin/env python

import argparse
import tensorflow as tf
import pycocotools.coco as coco
import numpy as np
import cv2
import os

from fcn_model import FeynmanModel

NUM_CLASSES = 3


def preprocess_coco_image(image_path):
    image = cv2.imread(image_path)
    processed_image = FeynmanModel.preprocess_image(image)
    return processed_image


def preprocess_coco_mask(mask):
    return mask.astype(np.float32) / 255.0


def data_generator(
    coco_dataset: coco.COCO,
    image_paths: list[str],
    image_ids: list[int],
    batch_size: int = 32,
):
    num_samples = len(image_paths)
    while True:
        batch_indexes = np.random.choice(num_samples, batch_size)
        batch_images, batch_masks = [], []
        for index in batch_indexes:
            image_path = image_paths[index]
            mask = coco_dataset.annToMask(
                coco_dataset.loadAnns(coco_dataset.getAnnIds(image_ids[index]))
            )
            processed_image = preprocess_coco_image(image_path)
            processed_mask = preprocess_coco_mask(mask)
            batch_images.append(processed_image)
            batch_masks.append(processed_mask)

        yield np.array(batch_images), np.array(batch_masks)


# Define the Mean Squared Error loss function for segmentation
def segmentation_loss(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)


def train_model(
    model: FeynmanModel,
    train_file: str,
    valid_file: str,
    test_file: str,
    num_epochs=100,
    batch_size=32,
):
    coco_dataset = coco.COCO(train_file)
    # Load image file paths from the dataset
    image_ids = coco_dataset.getImgIds()
    image_paths = [
        coco_dataset.loadImgs(image_id)[0]["file_name"] for image_id in image_ids
    ]

    model.compile(
        optimizer="adam",
        loss=segmentation_loss,
        metrics=["accuracy"],
    )

    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix, save_weights_only=True, save_best_only=True
    )

    model.fit(
        data_generator(coco_dataset, image_paths, image_ids, batch_size),
        epochs=num_epochs,
        callbacks=[checkpoint_callback],
        verbose=1,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        help="Training annotations file",
        default="dataset/train/_annotations.coco.json",
    )
    parser.add_argument(
        "--valid",
        help="Validations annotations file",
        default="dataset/valid/_annotations.coco.json",
    )
    parser.add_argument(
        "--test",
        help="Testing annotations file",
        default="dataset/test/_annotations.coco.json",
    )
    parser.add_argument(
        "--epochs", help="Number of training epochs", default=10, type=int
    )
    args = parser.parse_args()

    print("\n" * 4)
    print("TRAINING BEGIN")
    print("\n" * 4)

    model = FeynmanModel(num_classes=NUM_CLASSES)
    train_model(model, args.train, args.valid, args.test, num_epochs=args.epochs)
