import numpy as np
import tensorflow as tf
import keras.datasets.mnist as mnist
from prettytable import PrettyTable
import json

# Dataset
mnist_train, mnist_test = mnist.load_data()
(x_train, y_train) = mnist_train
(x_test, y_test) = mnist_test


def to_np_dtype(type_str: str):
    match type_str:
        case "int8":
            return np.int8
        case "int32":
            return np.int32
        case _:
            return np.float64


def quantize(array: np.ndarray, Z_input: np.int8, S_input: np.float64):
    """Quantization formulas
    r = S(q - Z)
    q = r/S + Z
    """
    quantized = np.zeros(array.shape, dtype=np.int8)
    match array.dtype:
        case np.uint8:
            for i in range(array.shape[1]):
                for j in range(array.shape[2]):
                    quantized[0][i][j] = array[0][i][j] + Z_input
        case np.float32:
            for i in range(array.shape[1]):
                for j in range(array.shape[2]):
                    quantized[0][i][j] = (array[0][i][j] / S_input) + Z_input
        case np.float64:
            for i in range(array.shape[1]):
                for j in range(array.shape[2]):
                    quantized[0][i][j] = (array[0][i][j] / S_input) + Z_input
        case _:
            print("another array dtype:", array.dtype)
    return quantized


def quantize_multiplier(scale: float):
    mantissa, exponent = np.frexp(scale)
    q_mantissa = np.round(mantissa * (1 << 31)).astype(np.int32)
    return (q_mantissa, exponent.astype(np.int32))


def multiply_by_quantize_mul(x, scale):
    q_mantissa, exponent = quantize_multiplier(scale)
    # print(q_mantissa, exponent)
    reduced_mantissa = (
        ((q_mantissa + (1 << 15)) >> 16) if q_mantissa < 0x7FFF0000 else 0x7FFF
    )
    total_shifts = 15 - exponent
    x = x * np.int64(reduced_mantissa)
    x = x + (1 << (total_shifts - 1))
    result = np.right_shift(x, total_shifts)
    return result


def do_dot(
    q_flat: np.ndarray,
    q_weights: np.ndarray,
    q_bias: np.ndarray,
    S_input,
    S_weights,
    S_output,
    Z_output,
) -> np.ndarray:
    dot_prod = np.dot(q_weights.astype(np.int32), q_flat.astype(np.int32))
    # print("####", dot_prod.shape, "-", q_bias.shape)
    acc = dot_prod + q_bias.flatten()
    scale = (S_input * S_weights) / S_output
    # print("Scale: ", scale)
    # acc = (scale * acc).astype(np.float32)
    acc = multiply_by_quantize_mul(acc.astype(np.int64), scale)
    acc += Z_output
    acc = np.maximum(acc, -128)
    acc = np.minimum(acc, 127)
    acc = np.rint(acc)  # round to the nearest integer
    acc = acc.astype(np.int8)
    return acc


# Load the data JSON
extr_json = open("extracted.json", "r").read()
extracted = json.loads(extr_json)

# Adjust the data
for layer in extracted["layers"]:
    layer["weights"]["data"] = np.asarray(
        layer["weights"]["data"], dtype=to_np_dtype(layer["weights"]["dtype"])
    )
    layer["bias"]["data"] = np.asarray(
        layer["bias"]["data"], dtype=to_np_dtype(layer["bias"]["dtype"])
    )


layers = extracted["layers"]


def invoke(image: np.ndarray):
    q_image = quantize(image, layers[0]["z_input"], layers[0]["s_input"])
    q_flat = q_image.flatten()
    acc = do_dot(
        q_flat,
        layers[0]["weights"]["data"],
        layers[0]["bias"]["data"],
        layers[0]["s_input"],
        layers[0]["s_weight"],
        layers[0]["s_output"],
        layers[0]["z_output"],
    )
    # print(acc.shape)
    acc = do_dot(
        acc,
        layers[1]["weights"]["data"],
        layers[1]["bias"]["data"],
        layers[1]["s_input"],
        layers[1]["s_weight"],
        layers[1]["s_output"],
        layers[1]["z_output"],
    )
    # print(acc.shape)
    return acc


def check_wrong():
    """
    Compare the output of the mistaken image from the extraction
    with the output of tensorflow lite
    """
    image = x_test[8]
    label = y_test[8]
    image = np.expand_dims(image, axis=0)  # from (28,28) to (1,28,28)
    acc = invoke(image)
    predicted = np.argmax(acc)
    print("CHECK WRONG:", acc, predicted, label)


check_wrong()

table = PrettyTable()
table.field_names = ["idx", "My", "TFLite", "MyPred", "TfPred", "SamePred", "Correct"]


def compare_results():
    interpreter = tf.lite.Interpreter("./models/lite/int_model.tflite")
    interpreter.allocate_tensors()
    idx = 0
    for image, label in zip(x_test, y_test):
        image = np.expand_dims(image, axis=0)
        interpreter.set_tensor(interpreter.get_input_details()[0]["index"], image)
        interpreter.invoke()
        ret = invoke(image)
        out = interpreter.get_tensor(9)[0]  # Check on TFLite which tensor!
        tf_lite_pred = np.argmax(out)
        my_pred = np.argmax(ret)
        if np.array_equal(ret, out) == False:
            table.add_row(
                [
                    idx,
                    ret,
                    out,
                    my_pred,
                    tf_lite_pred,
                    "✅" if my_pred == tf_lite_pred else "❌",
                    label,
                ]
            )
        idx += 1

    print(table)


compare_results()
