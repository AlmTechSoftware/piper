import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Conv2DTranspose,
    concatenate,
)


class FeynmanModel:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        inputs = Input(
            shape=(None, None, 3)
        )  # Allow input images of any size with 3 color channels

        # Encoder
        conv1 = Conv2D(64, 3, activation="relu", padding="same")(inputs)
        conv1 = Conv2D(64, 3, activation="relu", padding="same")(conv1)

        conv2 = Conv2D(128, 3, activation="relu", padding="same")(conv1)
        conv2 = Conv2D(128, 3, activation="relu", padding="same")(conv2)

        conv3 = Conv2D(256, 3, activation="relu", padding="same")(conv2)
        conv3 = Conv2D(256, 3, activation="relu", padding="same")(conv3)

        # Bottleneck
        conv4 = Conv2D(512, 3, activation="relu", padding="same")(conv3)
        conv4 = Conv2D(512, 3, activation="relu", padding="same")(conv4)

        # Decoder
        up1 = Conv2DTranspose(256, 2, strides=(2, 2), padding="same")(conv4)
        conv5 = Conv2D(256, 3, activation="relu", padding="same")(up1)
        conv5 = Conv2D(256, 3, activation="relu", padding="same")(conv5)

        up2 = Conv2DTranspose(128, 2, strides=(2, 2), padding="same")(conv5)
        conv6 = Conv2D(128, 3, activation="relu", padding="same")(up2)
        conv6 = Conv2D(128, 3, activation="relu", padding="same")(conv6)

        up3 = Conv2DTranspose(64, 2, strides=(2, 2), padding="same")(conv6)
        conv7 = Conv2D(64, 3, activation="relu", padding="same")(up3)
        conv7 = Conv2D(64, 3, activation="relu", padding="same")(conv7)

        # Output
        output = Conv2D(1, 1, activation="sigmoid")(conv7)

        model = tf.keras.Model(inputs=inputs, outputs=output)
        return model

    def compile_model(self):
        self.model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

    def load_dataset(self, dataset_dir: str):
        # Load COCO format annotations
        with open(os.path.join(dataset_dir, "_annotations.coco.json"), "r") as file:
            coco_data = json.load(file)

            # TODO: make segments

        # Example code for loading JPG images and resizing them to the input shape
        image_paths = coco_data["images"]
        images = []
        for image_info in image_paths:
            image_path = os.path.join(dataset_dir, image_info["file_name"])
            image = Image.open(image_path)
            image_array = np.array(image)
            images.append(image_array)

        # Convert images list to numpy array
        images = np.array(images)

        # Dummy data for masks for demonstration purposes
        masks = np.random.randint(0, 2, size=(100, 256, 256, 1))

        return images, masks

    def train(self, dataset_dir: str, epochs: int = 10, batch_size: int = 32):
        self.compile_model()
        images, masks = self.load_dataset(dataset_dir)
        self.model.fit(images, masks, epochs=epochs, batch_size=batch_size)


model = FeynmanModel()
model.train(dataset_dir="./dataset/train/", epochs=10, batch_size=32)
