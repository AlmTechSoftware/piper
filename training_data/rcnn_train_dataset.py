#!/usr/bin/env python

import os
import argparse
import tensorflow as tf
from tensorflow.python.ops.gen_dataset_ops import ZipDataset
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder
    conv1 = Conv2D(64, 3, activation="relu", padding="same")(inputs)
    conv1 = Conv2D(64, 3, activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation="relu", padding="same")(pool1)
    conv2 = Conv2D(128, 3, activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Decoder
    up8 = UpSampling2D(size=(2, 2))(pool2)
    up8 = Conv2D(128, 2, activation="relu", padding="same")(up8)
    merge9 = concatenate([conv2, up8], axis=3)
    conv9 = Conv2D(128, 3, activation="relu", padding="same")(merge9)
    conv9 = Conv2D(128, 3, activation="relu", padding="same")(conv9)

    # Output
    output = Conv2D(num_classes, 1, activation="softmax")(conv9)

    model = Model(inputs=inputs, outputs=output)
    return model


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
    target_size = (256, 256)

    image_generator = image_data_generator.flow_from_directory(
        data_dir,
        class_mode=None,
        target_size=target_size,
        batch_size=batch_size,
        seed=1,
    )

    mask_generator = mask_data_generator.flow_from_directory(
        labels_dir,
        class_mode=None,
        color_mode="grayscale",
        target_size=target_size,
        batch_size=batch_size,
        seed=1,
    )

    train_generator = ZipDataset(image_generator, mask_generator)

    return train_generator, image_generator, mask_generator


def compile_model(num_classes: int = 10, input_shape=(256, 256, 3)):
    model = create_model(input_shape, num_classes)

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def train_model(model: Model, train_generator, image_generator, mask_generator, epochs: int = 10):
    steps_per_epoch = len(image_generator)
    model.fit(train_generator, epochs=epochs, steps_per_epoch=steps_per_epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Data directory")
    parser.add_argument("--labels", help="Labels directory")
    args = parser.parse_args()

    model = compile_model()
    train_generator, image_generator, mask_generator = train_data_setup(args.data, args.labels)
    train_model(model, train_generator, image_generator, mask_generator)
    model.save("feynman.h5")
