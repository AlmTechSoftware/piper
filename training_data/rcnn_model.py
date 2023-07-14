import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dropout,
    UpSampling2D,
    concatenate,
)


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


def compile_model(num_classes: int = 10, input_shape=(None, None, 3)):
    model = create_model(input_shape, num_classes)

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model
