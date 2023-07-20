#!/usr/bin/env python

import argparse
import tensorflow as tf
import cv2
import os
import numpy as np

from fcn_model import FeynmanModel

NUM_CLASSES = 2


def load_image_and_label(image_path: str, label_path: str):
    image = tf.io.decode_jpeg(tf.io.read_file(image_path), channels=3)
    image = FeynmanModel.preprocess_image(image)

    if label_path is not None:
        label_mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label_mask = label_mask.astype(np.float32) / 255.0  # Convert to a float mask (0.0 or 1.0)
        return image, label_mask
    else:
        return image, None


def load_dataset(image_filenames: list[str], images_dir: str, labels_dir: str):
    image_paths = [os.path.join(images_dir, filename) for filename in image_filenames]

    label_paths = [
        os.path.join(labels_dir, f"{os.path.splitext(filename)[0]}_mask.png")
        for filename in image_filenames
    ]
    # dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
    dataset = zip(image_paths, label_paths)
    dataset = map(lambda s: load_image_and_label(*s), dataset)

    out_dataset = []
    for image, label in dataset:
        if label is not None:
            out_dataset.append((image, label))
        else:
            print("One image failed to have a label!")

    return dataset


def augment_data(image, label):
    # TODO: add augmentations
    return image, label


def train_model(
    model,
    images_dir,
    labels_dir,
    num_epochs=10,
    batch_size=16,
    validation_split=0.2,
):
    image_filenames = sorted(os.listdir(images_dir))
    num_validation_samples = int(validation_split * len(image_filenames))
    train_image_filenames = image_filenames[:-num_validation_samples]
    val_image_filenames = image_filenames[-num_validation_samples:]

    train_dataset = load_dataset(train_image_filenames, images_dir, labels_dir)
    val_dataset = load_dataset(val_image_filenames, images_dir, labels_dir)

    # Augmentation for training dataset (you can customize this part)
    train_dataset = train_dataset.map(augment_data)

    train_dataset = (
        train_dataset.shuffle(buffer_size=1000)
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )
    val_dataset = val_dataset.batch(batch_size)

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",  # You should adjust the loss based on your task
        metrics=["accuracy"],
    )

    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix, save_weights_only=True, save_best_only=True
    )

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
    parser.add_argument("--epochs", help="Number of training epochs", default=10)
    args = parser.parse_args()

    print("\n" * 4)
    print("TRAINING BEGIN")
    print("\n" * 4)

    model = FeynmanModel(num_classes=NUM_CLASSES)
    train_model(model, args.data, args.labels)
