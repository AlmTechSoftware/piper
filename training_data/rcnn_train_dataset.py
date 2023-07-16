#!/usr/bin/env python

import os
import tensorflow as tf
import numpy as np
import argparse

from rcnn_model import FeynmanModel


NUM_CLASSES = 2
NUM_EPOCHS = 10e10


def load_dataset(image_filenames: list[str], image_dir: str, labels_dir: str):
    dataset = []

    for filename in image_filenames:
        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(labels_dir, f"{os.path.splitext(filename)[0]}.txt")

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
    images_dir: str,
    labels_dir: str,
    num_epochs: int,
    validation_split: float = 0.2,
):
    # Get the list of image filenames from the data directory
    image_filenames = sorted(os.listdir(images_dir))

    # Split the data into training and validation sets
    num_validation_samples = int(validation_split * len(image_filenames))
    train_image_filenames = image_filenames[:-num_validation_samples]
    val_image_filenames = image_filenames[-num_validation_samples:]

    # Setup the datasets
    train_dataset = load_dataset(train_image_filenames, images_dir, labels_dir)
    val_dataset = load_dataset(val_image_filenames, images_dir, labels_dir)

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
        images_dir=args.data,
        labels_dir=args.labels,
        num_epochs=NUM_EPOCHS,
    )
