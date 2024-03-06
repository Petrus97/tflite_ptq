import numpy as np
from tensorflow._api.v2.nn import relu
from scipy.special import softmax
import keras.datasets.mnist as mnist
mnist_train, mnist_test = mnist.load_data()
(x_train, y_train) = mnist_train
(x_test, y_test) = mnist_test

TEST_NUM = 10

# print(y_test[TEST_NUM])
# print(x_test[TEST_NUM])

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
print(input.shape)
# print(input)

def quantize(array: np.ndarray):
    # r = S(q - Z)
    # q = r/S + Z
    quantized = np.zeros(array.shape, dtype=np.int8)
    for i in range(array.shape[1]):
        for j in range(array.shape[2]):
            quantized[0][i][j] = (array[0][i][j] / S_input) + Z_input
    return quantized

q_input = quantize(input)
# print(q_input)




# with open("data.txt", "w") as f:
#     for i in range(28):
#         for j in range(28):
#             f.write(str(input[0][i][i]))
#             f.write(", ")
#         f.write("\n")
# n,row,col= input.shape

# strstr = ""
# for i in range(row):
#     for j in range(col):
#         strstr += str(input[0][i][j]) + " "
#     strstr += '\n'
# print(strstr)
# print(input.shape)
# quantize = np.load("tfl.quantize.npy")
# print(quantize)

# print(quantize)
# reshape = np.reshape(quantize, [784, 1]).astype(np.int8)
# print(reshape)

# reshape = np.reshape(x_test[TEST_NUM], [784, 1])
# def to_int8(array: np.ndarray):
#     new = np.zeros(array.shape)
#     for i in range(len(array)):
#         new[i] = array[i][0] - 128
#     print("NEW", new)
# to_int8(reshape)
# print(reshape)

weights_out = np.load("sequential_dense_MatMul.npy").astype(np.int8)
#print("WEIGHTS", weights_out)
bias_out = np.load("sequential_dense_BiasAdd_ReadVariableOp.npy").astype(np.int32)

# Precumpute part
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
print(acc)
print(acc.astype(np.int8))


def my_dot(A: np.ndarray, B: np.ndarray, bias: np.ndarray):
    A_rows, A_cols = A.shape
    B_rows, B_cols = B.shape
    C_arr = np.zeros(shape=[A_rows, B_cols])
    C_rows, C_cols = C_arr.shape
    A_scale = 0.003921568859368563
    B_scale = 0.00432676263153553
    C_scale = 0.08895974606275558
    effective_scale = (A_scale * B_scale) / C_scale
    q_min = -128
    q_max = 127
    unscaled = np.zeros(shape=C_arr.shape)
    for i in range(C_rows):
        for j in range(C_cols):
            acc = 0
            for k in range(A_cols):
                acc += int(A[i, k]) * int(B[k, j])
            
            # acc = int((acc + bias[i,j])*effective_scale)
            acc = np.int32(acc)
            acc = np.int32(acc + bias[i,j])
            unscaled[i, j] = acc
            acc = int(acc * effective_scale)
            acc += 13
            acc = max(acc, q_min)
            acc = min(acc, q_max)
            C_arr[i, j] = np.int8(acc)
            # C_arr[i, j] = acc
    # print("Unscaled", unscaled)
    return C_arr

def apply_scaling(A: np.ndarray, bias: np.ndarray):
    A_scale = 0.003921568859368563
    B_scale = 0.00432676263153553
    C_scale = 0.08895974606275558
    effective_scale = (A_scale * B_scale) / C_scale
    q_min = -128
    q_max = 127
    A_rows, A_cols = A.shape
    for i in range(A_rows):
        for j in range(A_cols):
            A[i,j] = np.int32((A[i,j] + bias[i,j]) * effective_scale)
            A[i,j] -= -13
            A[i,j] = max(A[i, j], q_min)
            A[i,j] = min(A[i, j], q_max)
    return A

# qwqx = np.dot(weights_out, reshape)
# print("QWQX", qwqx)
# scaled = apply_scaling(qwqx)
# print("SCALED", scaled)
# my_result = my_dot(weights_out, reshape, bias_out)
# print("MY", my_result.astype(np.int8))
# result_int32 = qwqx + bias_out
# # print((result_int32 * (0.000016967/0.088959746)) + 13)
# print("DOT", result_int32)
# result_int8 = ((result_int32 * (0.000016967/0.088959746)) - 13 ).astype(np.int8)
# print(result_int8)
# smax = softmax(my_result)
# print(smax)
# print(np.argmax(smax))
# aaaa(out_hidden)