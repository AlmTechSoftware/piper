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


def load_image_and_label(image_filename: str, data_dir: str, labels_dir: str):
    try:
        image_path = os.path.join(data_dir, image_filename)
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32)

        label_filename = image_filename + ".txt"
        label_path = os.path.join(labels_dir, label_filename)
        label = np.loadtxt(label_path, delimiter=" ")
        label = np.expand_dims(label, axis=2)
        label = tf.convert_to_tensor(label, dtype=tf.float32)

        return image, label
    except FileNotFoundError:
        return None, None


def create_dataset(
    image_filenames: list[str], data_dir: str, labels_dir: str, batch_size: int
):
    dataset = map(lambda x: load_image_and_label(x, data_dir, labels_dir), image_filenames)
    images, labels = tuple(zip(*dataset))
    print(images)
    print("_--_--_")
    print(labels)



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
    train_dataset = create_dataset(
        train_image_filenames, data_dir, labels_dir, batch_size
    )
    val_dataset = create_dataset(val_image_filenames, data_dir, labels_dir, batch_size)

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
