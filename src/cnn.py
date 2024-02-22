import tensorflow as tf
import keras.datasets.mnist as mnist
import keras.utils as k_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from keras.optimizers import SGD
from keras.losses import MeanSquaredError, CategoricalCrossentropy
import numpy as np
import pathlib

MODEL_DIR = "./models"

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Scale the images to the [0,1] range
x_train = tf.cast(x_train, dtype=tf.float32) / 255
x_test = tf.cast(x_test, dtype=tf.float32) / 255

# Make the images from shape (28,28) to (28,28,1)
x_train = tf.expand_dims(x_train, axis=-1)  # -1 adds a dimension in the last position
x_test = tf.expand_dims(x_test, axis=-1)
assert x_train[0].shape == (28, 28, 1)
# Now we have the labels in the form [5 0 4 ... 5 6 8]. We want that each one is represented as a vector (10,1)
y_train = k_utils.to_categorical(y_train, num_classes=10)
y_test = k_utils.to_categorical(y_test, num_classes=10)
assert y_train[0].shape == (10,)


def create_model() -> Sequential:
    model = Sequential(
        [
            Conv2D(filters=8, kernel_size=3, input_shape=(28, 28, 1), use_bias=False),
            MaxPool2D(pool_size=2),
            Flatten(),
            Dense(10, activation="softmax"),
        ]
    )
    print(model.summary())
    return model


def train(model: Sequential) -> Sequential:
    print("Training...")
    batch_size = 128
    epochs = 3
    model.compile(
        optimizer=SGD(learning_rate=0.005),
        loss=CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    model.fit(
        x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1
    )

    return model


def evaluate(model: Sequential) -> Sequential:
    print("Evaluating...")
    score = model.evaluate(x=x_test, y=y_test)
    print(score)


def predict(model: Sequential) -> Sequential:
    print("Showing the first 10 predictions")
    predictions = model.predict(x=x_test)
    for i in range(10):
        print(f"Real: {np.argmax(y_test[i])} - Predicted: {np.argmax(predictions[i])}")


def save_model(model: Sequential, format: str = "all"):
    models_dir = pathlib.Path(MODEL_DIR)
    models_dir.mkdir(exist_ok=True, parents=True)
    # Save in all formats
    match format:
        case "h5":
            model.save(f"{MODEL_DIR}/model.h5", save_format="h5")
        case "tf":
            model.save(f"{MODEL_DIR}/model.tf", save_format="tf")
        case "keras":
            model.save(f"{MODEL_DIR}/model.keras", save_format="keras")
        case _: # save all
            model.save(f"{MODEL_DIR}/model.h5", save_format="h5")
            model.save(f"{MODEL_DIR}/model.keras", save_format="keras")
            model.save(f"{MODEL_DIR}/model.tf", save_format="tf")


def main():
    model = create_model()
    model = train(model)
    evaluate(model)
    predict(model)
    save_model(model)


if __name__ == "__main__":
    main()
