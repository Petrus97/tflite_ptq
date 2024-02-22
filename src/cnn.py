import tensorflow as tf
print(tf.__version__)
import keras.datasets.mnist as mnist
import keras.utils as k_utils
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from keras.optimizers import SGD
from keras.losses import MeanSquaredError, CategoricalCrossentropy
import numpy as np
import pathlib
import argparse

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


def save_model(model: Sequential, format: str = "keras"):
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
        case "all":  # save all
            model.save(f"{MODEL_DIR}/model.h5", save_format="h5")
            model.save(f"{MODEL_DIR}/model.keras", save_format="keras")
            model.save(f"{MODEL_DIR}/model.tf", save_format="tf")


def load_model(path: str = f"{MODEL_DIR}/model.keras") -> Sequential:
    model = keras.models.load_model(path)
    return model

def predict_lite(interpreter, input, output, data_in):
    interpreter.set_tensor(input, data_in)
    interpreter.invoke()
    out = interpreter.get_tensor(output)
    return out

def evaluate_tflite(interpreter: tf.lite.Interpreter):
    interpreter.allocate_tensors() # Needed before execution!
    input = interpreter.get_input_details()[0]["index"]
    output = interpreter.get_output_details()[0]["index"]
    for i in range(10):
        input_image = x_test[i]
        input_image = np.expand_dims(input_image, axis=0) # To have (1,28,28,1) tensor
        out = predict_lite(interpreter, input, output, input_image)[0]
        digit = np.argmax(out)
        actual_digit = np.argmax(y_test[i])
        print(f"Predicted Digit: {digit} - Real Digit: {actual_digit}\nConfidence: {out[digit]}")
    # for i in range(20):
    #     input_image = x_test[i]
    #     input_image = np.expand_dims(input_image, axis=0) # To have (1,28,28,1) tensor
    #     interpreter.set_tensor(input, input_image)
    #     interpreter.invoke()
    #     out = interpreter.tensor(output)()[0]

    #     # Print the model's classification result
    #     digit = np.argmax(out)
    #     actual_digit = np.argmax(y_test[i])
    #     print(f"Predicted Digit: {digit} - Real Digit: {actual_digit}\nConfidence: {out[digit]}")


def convert_to_tflite(model: Sequential, mode: str):
    print("Converting to TFLite...")
    LITE_MODEL_DIR = MODEL_DIR + "/lite"
    lite_models_dir = pathlib.Path(LITE_MODEL_DIR)
    lite_models_dir.mkdir(exist_ok=True, parents=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    match mode:
        case "dyn":
            # Dynamic range quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_dyn_model = converter.convert()
            with open(f"{LITE_MODEL_DIR}/dyn_model.tflite", "wb") as f:
                f.write(tflite_dyn_model)
            interpreter = tf.lite.Interpreter(model_content=tflite_dyn_model)
            evaluate_tflite(interpreter)
        case "float16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            tflite_f16_model = converter.convert()
            with open(f"{LITE_MODEL_DIR}/f16_model.tflite", "wb") as f:
                f.write(tflite_f16_model)
            interpreter = tf.lite.Interpreter(model_content=tflite_f16_model)
            evaluate_tflite(interpreter)
        case "int":
            def representative_dataset():
                mnist_train, _ = tf.keras.datasets.mnist.load_data()
                images = tf.cast(mnist_train[0], tf.float32) / 255.0
                mnist_ds = tf.data.Dataset.from_tensor_slices((images)).batch(1)
                for input_value in mnist_ds.take(100):
                    input_value = tf.expand_dims(input_value, axis=-1)
#                    print(input_value.shape)
                    yield [input_value]
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.float32  # or tf.uint8
            converter.inference_output_type = tf.int8  # or tf.uint8
            tflite_int_model = converter.convert()
            with open(f"{LITE_MODEL_DIR}/int_model.tflite", "wb") as f:
                f.write(tflite_int_model)
            interpreter = tf.lite.Interpreter(model_content=tflite_int_model)
            evaluate_tflite(interpreter)



def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load", default="")
    parser.add_argument("-t", "--train", action='store_true')
    parser.add_argument("-e", "--eval", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = argument_parser()
    if args.load:
        model = load_model(args.load)
    else:
        model = create_model()
    if args.train:
        model = train(model)
    if args.eval:
        evaluate(model)
        predict(model)
    save_model(model)
    convert_to_tflite(model, mode="int")


if __name__ == "__main__":
    main()
