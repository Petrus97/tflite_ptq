import tensorflow as tf

print(tf.__version__)
import numpy as np
import pathlib
import argparse


import keras
from keras.models import Sequential

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
            Dense(10, activation="softmax")
        ]
    )
    print(model.summary())
    return model


def train(model: Sequential) -> Sequential:
    print("Training...")
    batch_size = 100
    epochs = 10
    model.compile(
        optimizer=SGD(learning_rate=0.05),
        loss=CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    model.fit(
        x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1
    )

    return model


def evaluate(model:  Sequential) ->  Sequential:
    logging.info("Evaluating...")
    score = model.evaluate(x=x_test, y=y_test)
    logging.warn(score)


def predict(model:  Sequential) ->  Sequential:
    logging.info("Showing the first 10 predictions")
    from prettytable import PrettyTable
    table = PrettyTable()
    table.field_names = ["Predicted", "Original", "Confidence", "Match"]
    predictions = model.predict(x=x_test)
    for i in range(10):
        table.add_row(
            [
                np.argmax(predictions[i]),
                np.argmax(y_test[i]),
                np.max(predictions[i]),
                "✅" if np.argmax(predictions[i]) == np.argmax(y_test[i]) else "❌",
            ]
        )
    print(table)


def save_model(model: Sequential, format: str = "keras", model_name: str = ""):
    models_dir = pathlib.Path(MODEL_DIR)
    models_dir.mkdir(exist_ok=True, parents=True)
    # Save in all formats
    match format:
        case "h5":
            model.save(f"{MODEL_DIR}/{NET_TYPE}_model_{model_name}.h5", save_format="h5")
        case "tf":
            model.save(f"{MODEL_DIR}/{NET_TYPE}_model_{model_name}.tf", save_format="tf")
        case "keras":
            model.save(f"{MODEL_DIR}/{NET_TYPE}_model_{model_name}.keras", save_format="keras")
        case "all":  # save all
            model.save(f"{MODEL_DIR}/{NET_TYPE}_model_{model_name}.h5", save_format="h5")
            model.save(f"{MODEL_DIR}/{NET_TYPE}_model_{model_name}.keras", save_format="keras")
            model.save(f"{MODEL_DIR}/{NET_TYPE}_model_{model_name}.tf", save_format="tf")


def load_model(path: str = f"{MODEL_DIR}/{NET_TYPE}_model.keras") -> Sequential:
    print("loading", path)
    model = keras.models.load_model(path)
    return model

def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI utility that enables training and evaluation. The model is always saved.")
    parser.add_argument("-l", "--load", help="Load a saved model")
    parser.add_argument("-t", "--train", action="store_true", help="Train the model (creates one or use the loaded one)")
    parser.add_argument("-e", "--eval", action="store_true", help="Evaluate the model and show the first 10 predictions.")
    parser.add_argument("--lite", help="Convert to TFLite and evaluate it. Available modes:[dyn,float16,int]")
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
        print(type(model))
        tf_utils = TFLiteUtils(model, args.lite, model_name=NET_TYPE)
        tf_utils.set_dataset(x_train, y_train, x_test, y_test)
        tf_utils.convert_to_tflite()
        # convert_to_tflite(model, args.lite, model_name="")
    save_model(model, model_name="")



if __name__ == "__main__":
    main()
