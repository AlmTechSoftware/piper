#!/usr/bin/env python

import argparse
from tensorflow.python.ops.gen_dataset_ops import ZipDataset
from tensorflow.python.keras.models import Model
# from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from .rcnn_model import compile_model


def train_data_setup(data_dir: str, labels_dir: str):
    # Set up data augmentation
    data_gen_args = dict(
        rescale=1.0 / 255,  # normalize pixel values
        rotation_range=20,  # randomly rotate images
        width_shift_range=0.1,  # randomly shift images horizontally
        height_shift_range=0.1,  # randomly shift images vertically
        zoom_range=0.1,  # randomly zoom images
        horizontal_flip=True,
    )  # randomly flip images

    image_data_generator = ImageDataGenerator(**data_gen_args)
    mask_data_generator = ImageDataGenerator(**data_gen_args)

    batch_size = 8

    image_generator = image_data_generator.flow_from_directory(
        data_dir,
        class_mode=None,
        batch_size=batch_size,
        seed=1,
    )

    mask_generator = mask_data_generator.flow_from_directory(
        labels_dir,
        class_mode=None,
        color_mode="grayscale",
        batch_size=batch_size,
        seed=1,
    )

    train_generator = ZipDataset(image_generator, mask_generator)

    return train_generator, image_generator, mask_generator


def train_model(
    model: Model, train_generator, image_generator, mask_generator, epochs: int = 10
):
    steps_per_epoch = len(image_generator)
    model.fit(train_generator, epochs=epochs, steps_per_epoch=steps_per_epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Data directory")
    parser.add_argument("--labels", help="Labels directory")
    args = parser.parse_args()

    model = compile_model()
    train_generator, image_generator, mask_generator = train_data_setup(
        args.data, args.labels
    )
    train_model(model, train_generator, image_generator, mask_generator)
    model.save("feynman.h5")
