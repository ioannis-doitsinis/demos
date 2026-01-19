# src/model_def.py
import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.regularizers import l2

SEED = 42
tf.random.set_seed(SEED)


def build_ad_classifier(input_shape=(224, 224, 3)) -> tf.keras.Model:
    """
    AlexNet-inspired custom CNN for Ads vs Non-Ads classification.
    """
    model = Sequential()

    # Block 1
    model.add(
        Conv2D(
            32, (3, 3), strides=1, padding="same", activation="relu",
            input_shape=input_shape
        )
    )
    model.add(MaxPooling2D())
    model.add(BatchNormalization())

    # Block 2
    model.add(Conv2D(64, (3, 3), strides=1, padding="same", activation="relu"))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())

    # Block 3
    model.add(Conv2D(128, (3, 3), strides=1, padding="same", activation="relu"))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())

    # Classifier head
    model.add(Dropout(0.5, seed=SEED))
    model.add(Flatten())
    model.add(
        Dense(
            256,
            activation="relu",
            kernel_regularizer=l2(0.01)
        )
    )
    model.add(BatchNormalization())
    model.add(Dropout(0.5, seed=SEED))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=tf.keras.optimizers.Adagrad(
            learning_rate=0.001,
            initial_accumulator_value=0.1,
            epsilon=1e-7
        ),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"]
    )

    return model
