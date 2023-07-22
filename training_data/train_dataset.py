#!/usr/bin/env python

import argparse
import tensorflow as tf
import os

from fcn_model import FeynmanModel

NUM_CLASSES = 2

EXTENSION = "png"


def load_image(path: str):
    image = tf.io.decode_png(tf.io.read_file(path), channels=3)
    image = FeynmanModel.preprocess_image(image)

    return image


def load_image_and_label(image_path: str, label_path: str):
    print(f"proc \n{image_path=}\n{label_path=}\n")
    image = load_image(image_path)
    label = load_image(label_path)
    return image, label


def validate_filename(filename: str, directory_path: str):
    return filename.lower().endswith(f".{EXTENSION}") and os.path.isfile(
        os.path.join(directory_path, filename)
    )


def load_dataset(image_filenames: list[str], images_dir: str):
    # List all files in the dataset
    image_paths = [os.path.join(images_dir, filename) for filename in image_filenames]

    # Only fetch .png:s
    image_paths = list(
        filter(lambda path: validate_filename(path, images_dir), image_paths)
    )

    label_paths = [
        os.path.join(images_dir, f"labels/{os.path.splitext(filename)[0]}_mask.png")
        for filename in image_filenames
    ]

    # Remove pairs with missing label files
    valid_image_label_pairs = [
        (image_path, label_path)
        for image_path, label_path in zip(image_paths, label_paths)
        if os.path.isfile(label_path)
    ]

    dataset = tf.data.Dataset.from_tensor_slices(valid_image_label_pairs)
    print(dataset)
    dataset = dataset.map(
        lambda image_path, label_path: load_image_and_label(image_path, label_path)
    )

    out_dataset = []
    for image, label in dataset:
        if label is not None:
            out_dataset.append((image, label))
        else:
            print("One image failed to have a label!")

    print("Dataset loading done.")
    return tf.data.Dataset.from_tensor_slices(out_dataset)


def augment_data(image, label):
    # TODO: add augmentations
    return image, label


def train_model(
    model,
    images_dir,
    num_epochs=10,
    validation_split=0.2,
):
    image_filenames = sorted(os.listdir(images_dir))
    num_validation_samples = int(validation_split * len(image_filenames))
    train_image_filenames = image_filenames[:-num_validation_samples]
    val_image_filenames = image_filenames[-num_validation_samples:]

    train_dataset = load_dataset(train_image_filenames, images_dir)
    val_dataset = load_dataset(val_image_filenames, images_dir)

    # Augment the data
    train_dataset = train_dataset.map(augment_data)
    val_dataset = val_dataset.map(augment_data)

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
    parser.add_argument("--data", help="Dataset directory", default="dataset/")
    parser.add_argument(
        "--epochs", help="Number of training epochs", default=10, type=int
    )
    args = parser.parse_args()

    print("\n" * 4)
    print("TRAINING BEGIN")
    print("\n" * 4)

    model = FeynmanModel(num_classes=NUM_CLASSES)
    train_model(model, args.data, num_epochs=args.epochs)
