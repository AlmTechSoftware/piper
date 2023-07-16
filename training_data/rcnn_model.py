import tensorflow as tf
import os
import numpy as np
from tensorflow.python.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dropout,
    Conv2DTranspose,
    concatenate,
)


class FeynmanModel(tf.keras.Model):
    def __init__(self, num_classes):
        super(FeynmanModel, self).__init__()
        self.num_classes = num_classes
        self.encoder = self.build_encoder()
        self.bridge = self.build_bridge()
        self.decoder = self.build_decoder()
        self.output_layer = self.build_output_layer()

    def build_encoder(self):
        encoder = tf.keras.Sequential([
            Conv2D(64, 3, activation="relu", padding="same"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, 3, activation="relu", padding="same"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(256, 3, activation="relu", padding="same"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(512, 3, activation="relu", padding="same"),
            MaxPooling2D(pool_size=(2, 2))
        ])
        return encoder

    def build_bridge(self):
        return tf.keras.Sequential([
            Conv2D(1024, 3, activation="relu", padding="same"),
            Dropout(0.5)
        ])

    def build_decoder(self):
        return tf.keras.Sequential([
            Conv2DTranspose(512, 2, strides=(2, 2), padding="same"),
            Conv2D(512, 3, activation="relu", padding="same"),
            Conv2DTranspose(256, 2, strides=(2, 2), padding="same"),
            Conv2D(256, 3, activation="relu", padding="same"),
            Conv2DTranspose(128, 2, strides=(2, 2), padding="same"),
            Conv2D(128, 3, activation="relu", padding="same"),
            Conv2DTranspose(64, 2, strides=(2, 2), padding="same"),
            Conv2D(64, 3, activation="relu", padding="same")
        ])

    def build_output_layer(self):
        return Conv2D(self.num_classes, 1, activation="softmax", padding="same")

    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.bridge(x)
        x = self.decoder(x)
        output = self.output_layer(x)
        return tf.image.resize(output, tf.shape(inputs)[1:3])

    def preprocess_image(self, image):
        image = image.astype(np.float32) / 255.0
        target_height, target_width = self.encoder.layers[0].input_shape[1:3]
        image_resized = tf.image.resize(image, [target_height, target_width])
        image_expanded = np.expand_dims(image_resized, axis=0)
        return image_expanded

    def segment_image(self, image):
        image_expanded = self.preprocess_image(image)
        segmentation_map = self.predict(image_expanded)
        segmentation_map = np.squeeze(segmentation_map, axis=0)
        segmented_image = np.argmax(segmentation_map, axis=-1)
        return segmented_image

    def compile_model(self):
        self.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

    def train(
        self,
        images_dir: str,
        labels_dir: str,
        num_epochs: int,
        validation_split: float = 0.2,
    ):
        image_filenames = sorted(os.listdir(images_dir))
        num_validation_samples = int(validation_split * len(image_filenames))
        train_image_filenames = image_filenames[:-num_validation_samples]
        val_image_filenames = image_filenames[-num_validation_samples:]

        train_dataset = self.load_dataset(
            train_image_filenames, images_dir, labels_dir
        )
        val_dataset = self.load_dataset(val_image_filenames, images_dir, labels_dir)

        self.compile_model()

        checkpoint_dir = "./checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix, save_weights_only=True, save_best_only=True
        )

        self.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=num_epochs,
            callbacks=[checkpoint_callback],
            verbose=1,
        )

    def load_dataset(self, image_filenames, images_dir, labels_dir):
        image_paths = [os.path.join(images_dir, filename) for filename in image_filenames]
        label_paths = [
            os.path.join(labels_dir, f"{os.path.splitext(filename)[0]}.txt")
            for filename in image_filenames
        ]

        def load_image(image_path):
            image = tf.io.decode_jpeg(tf.io.read_file(image_path), channels=3)
            image_expanded = self.preprocess_image(image)
            return image_expanded

        def load_label(label_path):
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    label_values = [float(value) for value in f.read().split()]
                label = np.array(label_values, dtype=np.float32).reshape((-1, 2))
                return label
            else:
                return None

        image_dataset = tf.data.Dataset.from_generator(
            lambda: (load_image(path) for path in image_paths),
            output_signature=tf.TensorSpec(shape=(), dtype=tf.float32),
        )

        label_dataset = tf.data.Dataset.from_generator(
            lambda: (load_label(path) for path in label_paths),
            output_signature=tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
        ).filter(
            lambda x: x is not None
        )

        dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
        return dataset
