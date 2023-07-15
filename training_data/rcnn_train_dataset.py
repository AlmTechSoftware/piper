#!/usr/bin/env python

import argparse
from typing import List

# from tensorflow.python.ops.gen_dataset_ops import ZipDataset
# from tensorflow.python.keras.models import Model

# from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from rcnn_model import FeynmanModel

import os
import tensorflow as tf
import numpy as np

NUM_CLASSES = 2
BATCH_SIZE = 10
NUM_EPOCHS = 10


def load_dataset(dataset_dir: str):
    image_dir = os.path.join(dataset_dir, "images")
    label_dir = os.path.join(dataset_dir, "labels")

    image_filenames = os.listdir(image_dir)
    dataset = []

    for filename in image_filenames:
        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, f"{os.path.splitext(filename)[0]}.txt")

        # Load the image
        image = tf.io.decode_jpeg(tf.io.read_file(image_path), channels=3)
        image = tf.cast(image, tf.float32) / 255.0

        # Load the label
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                label_values = [float(value) for value in f.read().split()]
            label = np.array(label_values, dtype=np.float32)
        else:
            label = None

        dataset.append((image, label))

    return dataset


def train(
    model: FeynmanModel,
    data_dir: str,
    labels_dir: str,
    batch_size: int,
    num_epochs: int,
    validation_split: float = 0.2,
):
    # Get the list of image filenames from the data directory
    image_filenames = sorted(os.listdir(data_dir))

    # Split the data into training and validation sets
    num_validation_samples = int(validation_split * len(image_filenames))
    train_image_filenames = image_filenames[:-num_validation_samples]
    val_image_filenames = image_filenames[-num_validation_samples:]

    # Prepare the training and validation datasets
    # train_dataset = create_dataset(
    #     train_image_filenames, data_dir, labels_dir, batch_size
    # )
    # val_dataset = create_dataset(val_image_filenames, data_dir, labels_dir, batch_size)

    model.compile_model()  # Compile the model

    # Set up callbacks for saving model checkpoints
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix, save_weights_only=True, save_best_only=True
    )

    # Start training
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=num_epochs,
        callbacks=[checkpoint_callback],
        verbose=1,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", help="Data directory", default="dataset/train/images/"
    )
    parser.add_argument(
        "--labels", help="Labels directory", default="dataset/train/labels/"
    )
    args = parser.parse_args()

    model = FeynmanModel(num_classes=NUM_CLASSES)
    train(
        model,
        data_dir=args.data,
        labels_dir=args.labels,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
    )
