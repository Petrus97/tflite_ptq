import numpy as np
import tensorflow as tf
from scipy.special import softmax
import keras.datasets.mnist as mnist
from prettytable import PrettyTable
table = PrettyTable()
table.field_names = ["idx", "My", "TFLite", "MyPred", "TfPred", "Correct"]
mnist_train, mnist_test = mnist.load_data()
(x_train, y_train) = mnist_train
(x_test, y_test) = mnist_test

# Scaling factors
S_input = 1 / 255
S_weight = 0.00432676263153553
S_bias = 0.00001696769686532207
S_output = 0.08895974606275558

# Zero points
Z_input = -128 
Z_bias = 0
Z_weights = 0
Z_output = -13


input = np.load("serving_default_flatten_input:0.npy")
print(input.shape, input.dtype)

def quantize(array: np.ndarray):
    ''' Quantization formulas
        r = S(q - Z)
        q = r/S + Z
    '''
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


q_input = quantize(input)

weights_out = np.load("sequential_dense_MatMul.npy").astype(np.int8)
#print("WEIGHTS", weights_out)
bias_out = np.load("sequential_dense_BiasAdd_ReadVariableOp.npy").astype(np.int32)

# Precompute part
# qb − ZX qW
q_bias = bias_out - (weights_out @ (np.ones(shape=(784,), dtype=np.int32) * Z_input))
print(q_bias, q_bias.shape, q_bias.dtype)

# bias_out = np.reshape(bias_out, [10,1]).astype(np.int32)
# print("BIAS", bias_out)
# # print(bias_out.shape)
print(f"input {q_input.dtype} weights: {weights_out.dtype} bias: {bias_out.dtype}")
print(q_input.shape, weights_out.shape, bias_out.shape)

flatten_input = q_input.flatten()
print(flatten_input.shape)
dotdot = weights_out.astype(np.int32) @ flatten_input.astype(np.int32)
print(dotdot, dotdot.shape, dotdot.dtype)
acc = dotdot + q_bias
acc = ((S_input*S_weight)/S_output) * acc
acc += Z_output
acc = np.rint(acc)
print(acc)
print(acc.astype(np.int8))

def do_dot(q_flat: np.ndarray) -> np.ndarray:
    dot_prod = np.dot(weights_out.astype(np.int32), q_flat.astype(np.int32))
    acc = dot_prod + q_bias
    acc = (((S_input*S_weight)/S_output) * acc).astype(np.float32)
    acc += Z_output
    acc = np.maximum(acc, -128)
    acc = np.minimum(acc, 127)
    acc = np.rint(acc) # round to the nearest integer
    acc = acc.astype(np.int8)
    return acc

def evaluate_test_set(q_weights: np.ndarray, q_bias: np.ndarray):
    correct = 0
    for image, label in zip(x_test, y_test):
        # image = image / 255.0
        image = np.expand_dims(image, axis=0) # from (28,28) to (1,28,28)
        q_image = quantize(image)
        q_flat = q_image.flatten()
        acc = do_dot(q_flat=q_flat)
        predicted = np.argmax(acc)
        if label == predicted:
            correct += 1
    print("Lite accuracy:", (correct/10000)*100, "%")

evaluate_test_set(weights_out, q_bias)

def check_wrong(q_weights: np.ndarray, q_bias: np.ndarray):
    image = x_test[8837]
    label = y_test[8837]
    image = np.expand_dims(image, axis=0) # from (28,28) to (1,28,28)
    q_image = quantize(image)
    q_flat = q_image.flatten()
    acc = do_dot(q_flat=q_flat)
    predicted = np.argmax(acc)
    print(acc, predicted, label)

check_wrong(weights_out, q_bias)

def my_invoke(image: np.ndarray) -> np.ndarray:
    q_image = quantize(image)
    q_flat = q_image.flatten()
    acc = do_dot(q_flat=q_flat)
    return acc


def compare_results():
    interpreter = tf.lite.Interpreter("./models/lite/int_model.tflite")
    interpreter.allocate_tensors()
    idx = 0
    for image, label in zip(x_test, y_test):
        image = np.expand_dims(image, axis=0)
        interpreter.set_tensor(interpreter.get_input_details()[0]["index"], image)
        interpreter.invoke()
        ret = my_invoke(image)
        out = interpreter.get_tensor(6)[0]
        if(np.array_equal(ret, out) == False):
            table.add_row([idx, ret, out, np.argmax(ret), np.argmax(out), label])
        idx += 1

    print(table)

# compare_results()