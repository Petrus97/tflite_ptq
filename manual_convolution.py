import numpy as np
import tensorflow as tf
import keras.datasets.mnist as mnist
mnist_train, mnist_test = mnist.load_data()
(x_train, y_train) = mnist_train
(x_test, y_test) = mnist_test

from prettytable import PrettyTable
table = PrettyTable()
table.field_names = ["idx", "My", "TFLite", "MyPred", "TfPred", "Correct"]


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

def load_input() -> np.ndarray:
    from os.path import exists
    if exists("serving_default_flatten_input:0.npy"):
        input = np.load("serving_default_flatten_input:0.npy")
        return input
    elif exists("serving_default_conv2d_input:0.npy"):
        input = np.load("serving_default_conv2d_input:0.npy")
        return input
    else:
        raise FileNotFoundError("Input file not found")

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

filters = None
biases = None
weights = None
bias = None

def load_convolutional():
    global filters, biases
    if filters is not None and biases is not None:
        return filters, biases
    filters = np.load("sequential_conv2d_Conv2D1.npy").astype(np.int8)
    biases = np.load("sequential_conv2d_Conv2D.npy").astype(np.int32)
    # print("filter info", filters.shape, filters.dtype)
    # print("bias info", biases.shape, biases.dtype)
    # print(biases)
    return filters, biases

def load_dense():
    global weights, bias
    if weights is not None and bias is not None:
        return weights, bias
    weights = np.load("sequential_dense_MatMul.npy").astype(np.int8)
    bias = np.load("sequential_dense_BiasAdd_ReadVariableOp.npy").astype(np.int32)
    # print("weights info", weights.shape, weights.dtype)
    # print("bias info", bias.shape, bias.dtype)
    return weights, bias

# Do convolution
def convolution(input: np.ndarray, filters: np.ndarray, q_bias: np.int32) -> np.ndarray:
    # Step 1.2: calculate the scaline factor for the convolutional layer
    Sx = 1 / 255 # Input scaling factor
    Sw = 0.008923129178583622 # Weight scaling factor
    Sy = 0.02108168788254261 # Output scaling factor
    scale_factor = (Sx * Sw) / Sy
    Zy = -128 # Output zero point
    feature_maps = np.zeros(
        shape=(
            filters.shape[0], # same number as the number of filters
            input.shape[1] - filters.shape[1] + 1,  # img_rows - (filter_dim + 1)
            input.shape[2] - filters.shape[1] + 1,  # img_cols - (filter_dim + 1)
        ),
        dtype=np.int8
    )
    n_filters = filters.shape[0]
    for filter_idx in range(n_filters):
        filter = filters[filter_idx].reshape(filters.shape[1], filters.shape[2])
        reshaped_input = input.reshape(input.shape[1], input.shape[2])
        filter_size = filter.shape[0]
        feature_rows = feature_maps.shape[1]
        feature_cols = feature_maps.shape[2]
        out = np.zeros(shape=(
            feature_rows,
            feature_cols
        )).astype(np.int32)
        # print("out", out.shape, out.dtype)
        # Convolution operation
        for i in range(feature_rows):
            for j in range(feature_cols):
                for ii in range(filter_size):
                    for jj in range(filter_size):
                        out[i, j] += filter[ii, jj] * reshaped_input[i + ii, j + jj]
                # Relu layer
                out[i, j] = out[i, j] + q_bias
                out[i, j] = scale_factor * out[i, j]
                out[i, j] += Zy
                out[i, j] = max(out[i, j], -128)
                out[i, j] = min(out[i, j], 127)
        feature_maps[filter_idx] = out
    # print("feature maps", feature_maps.shape, feature_maps.dtype)
    return feature_maps

# Calculate q_b_conv = q_b - conv(q_w, Zx)
def conv(q_w: np.ndarray, Z_x: np.ndarray):
    # print(q_w.shape, Z_x.shape)
    # print(q_w)
    # print(Z_x)
    q_bias = np.int32(0)
    for i in range(q_w.shape[0]):
        for j in range(q_w.shape[1]):
            q_bias += np.int32(q_w[i][j]) * np.int32(Z_x[i][j])
    return q_bias

def dense_layer(q_flat: np.ndarray, q_weights: np.ndarray, q_bias: np.ndarray) -> np.ndarray:
    # XXX: This is a manual implementation of the dense layer, change if the network is trained again
    S_x = 0.02108168788254261 # Input scaling factor dense
    S_w = 0.004317202605307102 # Weight scaling factor dense
    S_y = 0.12329018861055374 # Output scaling factor dense
    Z_y = -19 # Output zero point dense
    scale_factor_dense = (S_x * S_w) / S_y

    dot_prod = np.dot(q_weights.astype(np.int32), q_flat.astype(np.int32))
    acc = dot_prod + q_bias
    acc = (scale_factor_dense * acc).astype(np.float32)
    acc += Z_y
    acc = np.maximum(acc, -128)
    acc = np.minimum(acc, 127)
    acc = np.rint(acc) # round to the nearest integer
    acc = acc.astype(np.int8)
    return acc

def evaluate_test_set():
    correct = 0
    for image, label in zip(x_test, y_test):
        # image = image / 255.0
        image = np.expand_dims(image, axis=(0, -1)) # from (28,28) to (1,28,28)
        ret = my_invoke(image)
        predicted = np.argmax(ret)
        if label == predicted:
            correct += 1
    print("Lite accuracy:", (correct/10000)*100, "%")

# evaluate_test_set(weights_out, q_bias)

def check_wrong(q_weights: np.ndarray, q_bias: np.ndarray):
    image = x_test[8837]
    label = y_test[8837]
    image = np.expand_dims(image, axis=0) # from (28,28) to (1,28,28)
    q_image = quantize(image)
    q_flat = q_image.flatten()
    acc = dense_layer(q_flat=q_flat, q_weights=q_weights, q_bias=q_bias)
    predicted = np.argmax(acc)
    print(acc, predicted, label)

# check_wrong(weights_out, q_bias)

def my_invoke(image: np.ndarray) -> np.ndarray:
    q_image = quantize(image)
    filters, biases = load_convolutional()
    Z_x = -128 # Input zero point conv
    q_bias_conv = precompute_qbias(layer="conv", q_b=biases, q_w=filters, Z_x=Z_x)
    q_conv = convolution(q_image.astype(np.int32), filters.astype(np.int32), q_bias_conv)
    maxpooled = maxpool2D(q_conv)
    q_weights, q_b_dense = load_dense()
    Z_x = -128 # Input zero point dense
    q_dense_bias = precompute_qbias(layer="dense", q_b=q_b_dense, q_w=q_weights, Z_x=Z_x)
    result = dense_layer(q_flat=maxpooled.flatten(), q_weights=q_weights, q_bias=q_dense_bias)
    return result


def compare_results():
    interpreter = tf.lite.Interpreter("./models/lite/int_cnn1d_conv_model.tflite")
    interpreter.allocate_tensors()
    idx = 0
    for image, label in zip(x_test, y_test):
        image = np.expand_dims(image, axis=(0, -1))
        # print(image.shape)
        interpreter.set_tensor(interpreter.get_input_details()[0]["index"], image)
        interpreter.invoke()
        ret = my_invoke(image)
        out = interpreter.get_tensor(10)[0]
        if(np.argmax(ret) != np.argmax(out)):
            print("idx:", idx, "My:", ret, "TFLite:", out, "MyPred:", np.argmax(ret), "TfPred:", np.argmax(out))
        if(np.array_equal(ret, out) == False):
            table.add_row([idx, ret, out, np.argmax(ret), np.argmax(out), label])
        idx += 1
    table_string = table.get_string()
    num_rows = len(table_string.split("\n"))
    if(num_rows > 40):
        with open("table.txt", "w") as f:
            f.write(table_string)
    else:
        print(table)

def maxpool2D(f_maps: np.ndarray, kernel_size = (2, 2), stride = 2):
    h_in = f_maps[0].shape[0] # rows
    w_in = f_maps[0].shape[1] # cols
    h_out = ((h_in - kernel_size[0]) // stride) + 1
    w_out = ((w_in - kernel_size[1]) // stride) + 1
    maxpooled_features = np.empty(shape=(
        f_maps.shape[0], # same number of the feature maps
        h_out,
        w_out
    ),
    dtype=np.int8
    )
    for f_map_idx in range(f_maps.shape[0]):
        f_map = f_maps[f_map_idx]
        # print(h_out, w_out)
        out = np.empty(shape=(
            h_out,
            w_out
        ),
        dtype=np.int8
        )
        for i in range(h_out):
            for j in range(w_out):
                max_value = -np.inf
                for ii in range(kernel_size[0]):
                    for jj in range(kernel_size[1]):
                        # Update max_value if the current element is greater
                        max_value = max(max_value, f_map[i * stride + ii, j * stride + jj])
                # Assign the maximum value to the corresponding output location
                out[i, j] = max_value
        maxpooled_features[f_map_idx] = out
    return maxpooled_features

def precompute_qbias(layer:str, q_b: np.int32, q_w: np.ndarray, Z_x: np.int32):
    if layer == "conv":
        # q_bias = q_b - conv(q_w, Zx)
        q_w = q_w.reshape(q_w.shape[1], q_w.shape[2])
        q_bias = q_b - conv(q_w.astype(np.int32), np.ones(shape=(q_w.shape[0], q_w.shape[1]), dtype=np.int32) * Z_x)
    elif layer == "dense":
        # q_bias = q_b - dot(q_w, Zx)
        q_bias = q_b - (q_w @ (np.ones(shape=(q_w.shape[1],), dtype=np.int32) * Z_x))
    else:
        raise ValueError("Invalid layer")
    return q_bias


def main():
    # Load the same input used in the tflite inference evaluation
    input = load_input()
    print("input meta:", input.shape, input.dtype)
    # Step 0: quantize the input
    q_input = quantize(input)
    print("q_input meta:",q_input.shape, q_input.dtype)
    # Step 1: Load the convolutional layer
    filters, biases = load_convolutional()
    print("filters meta:",filters.shape, filters.dtype)
    print("biases meta:",biases.shape, biases.dtype)
    # Step 1.1: Precompute the q_bias for the convolutional layer
    # Y = Conv (W, X) + b 
    # with Zw=0 Zb=0 Sb=SwSx and q_bias = q_b - conv(q_w, Zx)
    Z_x = -128 # Input zero point conv
    q_bias_conv = precompute_qbias(layer="conv", q_b=biases, q_w=filters, Z_x=Z_x)
    print("q_bias meta", q_bias_conv)
    # Step 2: Do the convolution
    # qY = (SwSx)/Sy (conv(qW, qX) + qBias) + Zy
    q_conv = convolution(q_input.astype(np.int32), filters.astype(np.int32), q_bias_conv)
    print("q_conv meta:", q_conv.shape, q_conv.dtype)
    # Step 3: Maxpool 2D
    maxpooled = maxpool2D(q_conv)
    print("maxpooled meta:", maxpooled.shape, maxpooled.dtype)
    # Step 4: Load the dense layer
    q_weights, q_b_dense = load_dense()
    Z_x = -128 # Input zero point dense
    q_dense_bias = precompute_qbias(layer="dense", q_b=q_b_dense, q_w=q_weights, Z_x=Z_x)
    result = dense_layer(q_flat=maxpooled.flatten(), q_weights=q_weights, q_bias=q_dense_bias)
    print("result meta:", result.shape, result.dtype)
    print(result, np.argmax(result))
    compare_results()
    # evaluate_test_set()


if __name__ == "__main__":
    main()