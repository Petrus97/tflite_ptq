import pathlib
from prettytable import PrettyTable
import numpy as np
from keras.models import Sequential
from dataset import *
from logger import logging

table = PrettyTable()
table.field_names = ["Predicted", "Original", "Confidence", "Match"]

def check(interpreter: tf.lite.Interpreter):
    # check the output tensor for the first image
    input_image = mnist_test[0][0]
    # print(input_image)
    input_image = np.expand_dims(input_image, axis=0)
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], input_image)
    # print(interpreter.get_tensor(interpreter.get_input_details()[0]["index"]))
    interpreter.invoke()
    for t_detail in interpreter.get_tensor_details():
        tensor = t_detail["index"]
        print("{}".format(t_detail["name"]))
        print(interpreter.get_tensor(tensor))
    out = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
    print(out)

def predict_lite(interpreter: tf.lite.Interpreter, data_in) -> np.ndarray:
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], data_in)
    interpreter.invoke()
    out = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
    return out

def evaluate_tflite(interpreter: tf.lite.Interpreter):
    interpreter.allocate_tensors()  # Needed before execution!
    for i in range(20):
        # input_image = mnist_test[0][i]
        input_image = x_test[i]
        logging.warn(input_image.shape)
        input_image = np.expand_dims(input_image, axis=0)  # To have (1,28,28,1) tensor
        out = predict_lite(interpreter, input_image)
        digit = np.argmax(out[0])
        actual_digit = np.argmax(y_test[i])
        confidence = out[0][digit]
        table.add_row([digit, actual_digit, confidence, "✅" if digit == actual_digit else "❌"])
        # print(
        #     f"Predicted Digit: {digit} - Real Digit: {actual_digit}\nConfidence: {out[0][digit]}"
        # )
        # if digit != actual_digit:
        #     print("Confidence vector:", out[0])
    print(table)
    correct = 0
    for i in range(10000):
        # input_image = mnist_test[0][i]
        input_image = x_test[i]
        input_image = np.expand_dims(input_image, axis=0)  # To have (1,28,28,1) tensor
        out = predict_lite(interpreter, input_image)
        digit = np.argmax(out[0])
        actual_digit = np.argmax(y_test[i])
        confidence = out[0][digit]
        if digit == actual_digit:
            correct += 1
        # table.add_row([digit, actual_digit, confidence, "✅" if digit == actual_digit else "❌"])
    print("Lite accuracy:", (correct/10000)*100, "%")
        # check the output tensor for the first image
    # input_image = mnist_test[0][10]
    input_image = x_test[10]
    # print(input_image)
    input_image = np.expand_dims(input_image, axis=0)
    # input_image.tofile("same_img.data")
    # np.save("image_test",input_image)
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], input_image)
    # print(interpreter.get_tensor(interpreter.get_input_details()[0]["index"]))
    interpreter.invoke()
    logging.debug(interpreter.get_tensor_details())
    for t_detail in interpreter.get_tensor_details():
        logging.info(t_detail)
        tensor = t_detail["index"]
        logging.debug("{} - {}".format(t_detail["name"], t_detail["index"]))
        try:
            array = interpreter.get_tensor(tensor)
            np.save(t_detail["name"].replace("/", "_"), array)
            print(array)
        except ValueError:
            print("Tensor data is null")
    out = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
    print(out)
    print(np.argmax(out))





def convert_to_tflite(model: Sequential, mode: str):
    print("Converting to TFLite...")
    print(model.summary())
    LITE_MODEL_DIR = MODEL_DIR + "/lite"
    lite_models_dir = pathlib.Path(LITE_MODEL_DIR)
    lite_models_dir.mkdir(exist_ok=True, parents=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    match mode:
        case "dyn":
            # Dynamic range quantization
            tflite_dyn_model = converter.convert()
            with open(f"{LITE_MODEL_DIR}/dyn_model.tflite", "wb") as f:
                f.write(tflite_dyn_model)
            interpreter = tf.lite.Interpreter(model_content=tflite_dyn_model)
            evaluate_tflite(interpreter)
        case "float16":
            # Float16 quantization
            converter.target_spec.supported_types = [tf.float16]
            tflite_f16_model = converter.convert()
            with open(f"{LITE_MODEL_DIR}/f16_model.tflite", "wb") as f:
                f.write(tflite_f16_model)
            interpreter = tf.lite.Interpreter(model_content=tflite_f16_model)
            evaluate_tflite(interpreter)
        case "int":
            # Full integer quantization
            def representative_dataset():
                images = x_train # tf.cast(mnist_train[0], tf.float32) / 255.0
                mnist_ds = tf.data.Dataset.from_tensor_slices((images)).batch(1)
                for input_value in mnist_ds.take(100):
                    yield [input_value]

            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.float32  # or tf.uint8
            converter.inference_output_type = tf.int8  # or tf.uint8
            tflite_int_model = converter.convert()
            with open(f"{LITE_MODEL_DIR}/int_model.tflite", "wb") as f:
                f.write(tflite_int_model)
            interpreter = tf.lite.Interpreter(model_content=tflite_int_model)
            evaluate_tflite(interpreter)

