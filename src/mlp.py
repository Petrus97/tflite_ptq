import tensorflow as tf

print(tf.__version__)
import keras
import numpy as np
import pathlib
import argparse

from keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from keras.optimizers import SGD
from keras.losses import MeanSquaredError, CategoricalCrossentropy

# My modules
from dataset import *
from tflite_utils import *

NET_TYPE="mlp"


def create_model() -> Sequential:
    model = Sequential(
        [
            Flatten(input_shape=(28, 28)),
            # Dense(30, activation=tf.nn.relu),
            Dense(10, activation=tf.nn.softmax)
        ]
    )
    print(model.summary())
    return model


def train(model: Sequential) -> Sequential:
    print("Training...")
    batch_size = 10
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


def save_model(model: Sequential, format: str = "keras"):
    models_dir = pathlib.Path(MODEL_DIR)
    models_dir.mkdir(exist_ok=True, parents=True)
    # Save in all formats
    match format:
        case "h5":
            model.save(f"{MODEL_DIR}/{NET_TYPE}_model.h5", save_format="h5")
        case "tf":
            model.save(f"{MODEL_DIR}/{NET_TYPE}_model.tf", save_format="tf")
        case "keras":
            model.save(f"{MODEL_DIR}/{NET_TYPE}_model.keras", save_format="keras")
        case "all":  # save all
            model.save(f"{MODEL_DIR}/{NET_TYPE}_model.h5", save_format="h5")
            model.save(f"{MODEL_DIR}/{NET_TYPE}_model.keras", save_format="keras")
            model.save(f"{MODEL_DIR}/{NET_TYPE}_model.tf", save_format="tf")


def load_model(path: str = f"{MODEL_DIR}/{NET_TYPE}_model.keras") -> Sequential:
    print("loading", path)
    model = keras.models.load_model(path)
    return model

def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI utility that enables training and evaluation. The model is always saved.")
    parser.add_argument("-l", "--load", help="Load a saved model")
    parser.add_argument("-t", "--train", action="store_true", help="Train the model (creates one or use the loaded one)")
    parser.add_argument("-e", "--eval", action="store_true", help="Evaluate the model and show the first 10 predictions.")
    parser.add_argument("--lite", default="int", help="Convert to TFLite and evaluate it. Available modes:[dyn,float16,int]") # FIXME missing argument doesn't get the default one
    args = parser.parse_args()
    return args


def main():
    args = argument_parser()
    print(args)
    if args.load:
        model = load_model(args.load)
    else:
        model = create_model()
    if args.train:
        model = train(model)
    if args.eval:
        evaluate(model)
        predict(model)
    if args.lite:
        convert_to_tflite(model, args.lite)
    save_model(model)



if __name__ == "__main__":
    main()
