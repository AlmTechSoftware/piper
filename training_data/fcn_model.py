import tensorflow as tf
import os
import cv2
import numpy as np


class FeynmanModel(tf.keras.Model):
    def preprocess_image(image):
        image = tf.cast(image, np.float32)
        if image is not None:
            image /= 255
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return gray
        else:
            error(f"{image=}, image is None. Error or something idunno")

    def __init__(self, num_classes):
        super(FeynmanModel, self).__init__()
        self.num_classes = num_classes

        # Define the VGG16 model as the backbone
        self.backbone = tf.keras.applications.VGG16(
            include_top=False, weights="imagenet"
        )

        self.conv1 = tf.keras.layers.Conv2D(
            4096, (7, 7), activation="relu", padding="same"
        )
        self.conv2 = tf.keras.layers.Conv2D(
            4096, (1, 1), activation="relu", padding="same"
        )
        self.conv3 = tf.keras.layers.Conv2D(
            self.num_classes, (1, 1), activation="softmax", padding="valid"
        )

        self.upsample_32x = tf.keras.layers.Conv2DTranspose(
            self.num_classes, (64, 64), strides=(32, 32), padding="same"
        )
        self.upsample_2x = tf.keras.layers.Conv2DTranspose(
            self.num_classes, (4, 4), strides=(2, 2), padding="same"
        )

    def call(self, inputs):
        # Pass the input through the VGG16 backbone
        x = self.backbone(inputs)

        # Additional convolutions for segmentation
        x = self.conv1(x)
        x = self.conv2(x)

        # Upsample the output to the original image size
        x = self.upsample_2x(x)
        x = self.upsample_32x(x)

        # Apply the final convolutional layer for segmentation
        x = self.conv3(x)

        return x

    def segment_image(self, image):
        proc_image = FeynmanModel.preprocess_image(image)
        image_expanded = np.expand_dims(proc_image, axis=0)
        segmentation_map = self.predict(image_expanded)
        segmentation_mask = cv2.resize(
            segmentation_map[0], (image.shape[1], image.shape[0])
        )
        return segmentation_mask
