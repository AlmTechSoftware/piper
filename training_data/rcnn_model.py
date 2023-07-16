import tensorflow as tf
import os
import numpy as np
from tensorflow.python.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dropout,
    Conv2DTranspose,
    concatenate,
    # UpSampling2D,
)


class FeynmanModel(tf.keras.Model):
    def __init__(self, num_classes):
        super(FeynmanModel, self).__init__()
        self.num_classes = num_classes

        # Encoder
        self.conv1 = Conv2D(64, 3, activation="relu", padding="same")
        self.pool1 = MaxPooling2D(pool_size=(2, 2))
        self.conv2 = Conv2D(128, 3, activation="relu", padding="same")
        self.pool2 = MaxPooling2D(pool_size=(2, 2))
        self.conv3 = Conv2D(256, 3, activation="relu", padding="same")
        self.pool3 = MaxPooling2D(pool_size=(2, 2))
        self.conv4 = Conv2D(512, 3, activation="relu", padding="same")
        self.pool4 = MaxPooling2D(pool_size=(2, 2))

        # Bridge
        self.conv5 = Conv2D(1024, 3, activation="relu", padding="same")
        self.dropout = Dropout(0.5)

        # Decoder
        self.concat = lambda l: concatenate(l, axis=3)
        self.upconv1 = Conv2DTranspose(512, 2, strides=(2, 2), padding="same")
        # self.concat1 = concatenate([self.conv4, self.upconv1], axis=3)
        self.conv6 = Conv2D(512, 3, activation="relu", padding="same")
        self.upconv2 = Conv2DTranspose(256, 2, strides=(2, 2), padding="same")
        # self.concat2 = concatenate([self.conv3, self.upconv2], axis=3)
        self.conv7 = Conv2D(256, 3, activation="relu", padding="same")
        self.upconv3 = Conv2DTranspose(128, 2, strides=(2, 2), padding="same")
        # self.concat3 = concatenate([self.conv2, self.upconv3], axis=3)
        self.conv8 = Conv2D(128, 3, activation="relu", padding="same")
        self.upconv4 = Conv2DTranspose(64, 2, strides=(2, 2), padding="same")
        # self.concat4 = concatenate([self.conv1, self.upconv4], axis=3)
        self.conv9 = Conv2D(64, 3, activation="relu", padding="same")

        # Output
        self.conv10 = Conv2D(self.num_classes, 1, activation="softmax", padding="same")

    def call(self, inputs):
        # Encoder
        x1 = self.conv1(inputs)
        x1_pool = self.pool1(x1)
        x2 = self.conv2(x1_pool)
        x2_pool = self.pool2(x2)
        x3 = self.conv3(x2_pool)
        x3_pool = self.pool3(x3)
        x4 = self.conv4(x3_pool)
        x4_pool = self.pool4(x4)

        # Bridge
        x5 = self.conv5(x4_pool)
        x5_dropout = self.dropout(x5)

        # Decoder
        x6 = self.upconv1(x5_dropout)
        x6_concat = self.concat([x4, x6])
        x7 = self.conv6(x6_concat)
        x8 = self.upconv2(x7)
        x8_concat = self.concat([x3, x8])
        x9 = self.conv7(x8_concat)
        x10 = self.upconv3(x9)
        x10_concat = self.concat([x2, x10])
        x11 = self.conv8(x10_concat)
        x12 = self.upconv4(x11)
        x12_concat = self.concat([x1, x12])
        x13 = self.conv9(x12_concat)

        # Output
        output = self.conv10(x13)
        return tf.image.resize(output, tf.shape(inputs)[1:3])

    def preprocess_image(self, image):
        # Preprocess the image
        image = (
            image.astype(np.float32) / 255.0
        )  # Normalize pixel values between 0 and 1

        # Resize the image to match the model's input shape
        target_height, target_width = self.conv1.input_shape[1:3]
        image_resized = tf.image.resize(image, [target_height, target_width])

        # Expand dimensions to match the model's input shape
        image_expanded = np.expand_dims(image_resized, axis=0)

        return image_expanded

    def segment_image(self, image):
        # Preprocess the image
        image_expanded = self.preprocess_image(image)

        # Perform inference
        segmentation_map = self.predict(image_expanded)

        # Post-processing to obtain segmentation regions in the original image
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
        # Get the list of image filenames from the data directory
        image_filenames = sorted(os.listdir(images_dir))

        # Split the data into training and validation sets
        num_validation_samples = int(validation_split * len(image_filenames))
        train_image_filenames = image_filenames[:-num_validation_samples]
        val_image_filenames = image_filenames[-num_validation_samples:]

        def load_dataset(image_filenames: list[str], image_dir: str, labels_dir: str):
            image_paths = [
                os.path.join(image_dir, filename) for filename in image_filenames
            ]
            label_paths = [
                os.path.join(labels_dir, f"{os.path.splitext(filename)[0]}.txt")
                for filename in image_filenames
            ]

            def load_image(image_path: str):
                # Load the JPEG image
                image = tf.io.decode_jpeg(tf.io.read_file(image_path), channels=3)
                image_expanded = self.preprocess_image(image)

                return image_expanded

            def load_label(label_path: str):
                if os.path.exists(label_path):
                    with open(label_path, "r") as f:
                        label_values = [float(value) for value in f.read().split()]
                    label = np.array(label_values, dtype=np.float32).reshape((-1, 2))
                    return label
                else:
                    return None

            # image_dataset = map(load_image, image_paths)
            # label_dataset = map(load_label, label_paths)

            image_dataset = tf.data.Dataset.from_generator(
                lambda: (load_image(path) for path in image_paths),
                output_signature=tf.TensorSpec(shape=(), dtype=tf.float32),
            )

            label_dataset = tf.data.Dataset.from_generator(
                lambda: (load_label(path) for path in label_paths),
                output_signature=tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
            ).filter(
                lambda x: x is not None
            )  # Filter out None values

            dataset = tf.data.Dataset.zip((image_dataset, label_dataset))

            return dataset

        # Setup the datasets
        train_dataset = load_dataset(train_image_filenames, images_dir, labels_dir)
        val_dataset = load_dataset(val_image_filenames, images_dir, labels_dir)

        # Compile the model for training
        self.compile_model()

        # Set up callbacks for saving model checkpoints
        checkpoint_dir = "./checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix, save_weights_only=True, save_best_only=True
        )

        # Start training
        self.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=num_epochs,
            callbacks=[checkpoint_callback],
            verbose=1,
        )
