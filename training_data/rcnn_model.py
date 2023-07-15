import tensorflow as tf
from tensorflow.python.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dropout,
    concatenate,
    # UpSampling2D,
)


class FeynmanModel(tf.keras.Model):
    def __init__(self, num_classes):
        super(FeynmanModel, self).__init__()

        # Encoder
        self.encoder_layers = [
            Conv2D(
                64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
            ),
            Conv2D(
                64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
            ),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(
                128,
                3,
                activation="relu",
                padding="same",
                kernel_initializer="he_normal",
            ),
            Conv2D(
                128,
                3,
                activation="relu",
                padding="same",
                kernel_initializer="he_normal",
            ),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(
                256,
                3,
                activation="relu",
                padding="same",
                kernel_initializer="he_normal",
            ),
            Conv2D(
                256,
                3,
                activation="relu",
                padding="same",
                kernel_initializer="he_normal",
            ),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(
                512,
                3,
                activation="relu",
                padding="same",
                kernel_initializer="he_normal",
            ),
            Conv2D(
                512,
                3,
                activation="relu",
                padding="same",
                kernel_initializer="he_normal",
            ),
            Dropout(0.5),
            MaxPooling2D(pool_size=(2, 2)),
        ]

        # Decoder
        self.decoder_layers = [
            Conv2D(
                1024,
                3,
                activation="relu",
                padding="same",
                kernel_initializer="he_normal",
            ),
            Conv2D(
                1024,
                3,
                activation="relu",
                padding="same",
                kernel_initializer="he_normal",
            ),
            Dropout(0.5),
            Conv2D(
                512,
                2,
                activation="relu",
                padding="same",
                kernel_initializer="he_normal",
            ),
            concatenate,
            Conv2D(
                512,
                3,
                activation="relu",
                padding="same",
                kernel_initializer="he_normal",
            ),
            Conv2D(
                512,
                3,
                activation="relu",
                padding="same",
                kernel_initializer="he_normal",
            ),
            Conv2D(
                256,
                2,
                activation="relu",
                padding="same",
                kernel_initializer="he_normal",
            ),
            concatenate,
            Conv2D(
                256,
                3,
                activation="relu",
                padding="same",
                kernel_initializer="he_normal",
            ),
            Conv2D(
                256,
                3,
                activation="relu",
                padding="same",
                kernel_initializer="he_normal",
            ),
            Conv2D(
                128,
                2,
                activation="relu",
                padding="same",
                kernel_initializer="he_normal",
            ),
            concatenate,
            Conv2D(
                128,
                3,
                activation="relu",
                padding="same",
                kernel_initializer="he_normal",
            ),
            Conv2D(
                128,
                3,
                activation="relu",
                padding="same",
                kernel_initializer="he_normal",
            ),
            Conv2D(
                64, 2, activation="relu", padding="same", kernel_initializer="he_normal"
            ),
            concatenate,
            Conv2D(
                64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
            ),
            Conv2D(
                64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
            ),
            Conv2D(
                num_classes,
                3,
                activation="softmax",
                padding="same",
                kernel_initializer="he_normal",
            ),
        ]

    def call(self, inputs):
        # Encoder
        encoder_output = inputs
        encoder_skip_connections = []

        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output)
            encoder_skip_connections.append(encoder_output)

        encoder_skip_connections = encoder_skip_connections[:-1][
            ::-1
        ]  # Remove last layer and reverse

        # Decoder
        decoder_output = encoder_output

        for i, layer in enumerate(self.decoder_layers):
            if isinstance(layer, concatenate):
                decoder_output = layer(
                    [decoder_output, encoder_skip_connections[i // 2]]
                )
            else:
                decoder_output = layer(decoder_output)

        return decoder_output

    def compile_model(self):
        self.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )


if __name__ == "__main__":
    x = FeynmanModel(2)
    x.compile_model()
