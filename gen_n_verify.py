import numpy as np
import tensorflow as tf
import keras.datasets.mnist as mnist
from prettytable import PrettyTable
import json

def to_np_dtype(type_str: str):
    match type_str:
        case "int8":
            return np.int8
        case "int32":
            return np.int32
        case _:
            return np.float64
        
def multiply_by_quantize_mul(x: np.int64, q_mantissa: np.int32, exponent: np.int32):
    # q_mantissa, exponent = quantize_multiplier(scale)
    # print(q_mantissa, exponent)
    reduced_mantissa = (
        ((q_mantissa + (1 << 15)) >> 16) if q_mantissa < 0x7FFF0000 else 0x7FFF
    )
    total_shifts = 15 - exponent
    x = np.int64(x) * np.int64(reduced_mantissa)
    x = x + (1 << (total_shifts - 1))
    result = np.right_shift(x, total_shifts)
    return result

class Layer:
    def __init__(self):
        pass

    def apply_layer(self, input: np.ndarray) -> np.ndarray:
        return np.ndarray([])  # Placeholder return value
    
    def generate_code(self):
        return ""

class Quantize(Layer):
    def __init__(self, Z_input: np.int8 = np.int8(0), S_input: np.float64 = np.float64(0)):
        self.array: np.ndarray = np.array([])
        self.Z_input = Z_input
        self.S_input = S_input

    def quantize(self, array: np.ndarray):
        """Quantization formulas
        r = S(q - Z)
        q = r/S + Z
        """
        self.array = array
        quantized = np.zeros(self.array.shape, dtype=np.int8)
        match self.array.dtype:
            case np.uint8:
                for i in range(self.array.shape[1]):
                    for j in range(self.array.shape[2]):
                        quantized[0][i][j] = self.array[0][i][j] + self.Z_input
            case np.float32:
                for i in range(self.array.shape[1]):
                    for j in range(self.array.shape[2]):
                        quantized[0][i][j] = (self.array[0][i][j] / self.S_input) + self.Z_input
            case np.float64:
                for i in range(self.array.shape[1]):
                    for j in range(self.array.shape[2]):
                        quantized[0][i][j] = (self.array[0][i][j] / self.S_input) + self.Z_input
            case _:
                print("another array dtype:", self.array.dtype)
        return quantized

    def apply_layer(self, array: np.ndarray):
        return self.quantize(array)

class Conv2D(Layer):
    def __init__(self, input_shape: tuple, output_shape: tuple):
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def set_fixed_point(self, mantissa: np.int32, exponent: np.int32):
        self.q_mantissa = mantissa
        self.exponent = exponent

    def set_zero_points(self, input_zero_point: np.int8, filter_zero_point: np.int8, bias_zero_point: np.int8, output_zero_point: np.int8):
        self.input_zero_point = input_zero_point
        self.filter_zero_point = filter_zero_point
        self.bias_zero_point = bias_zero_point
        self.output_zero_point = output_zero_point

    def set_filter(self, filter: list, dtype: str):
        self.filter = np.asarray(filter, dtype=to_np_dtype(dtype))
    
    def set_bias(self, bias: list, dtype: str):
        self.bias = np.asarray(bias, dtype=to_np_dtype(dtype))

    def __mul_by_filter__(self, input: np.ndarray, i,j,k,l) -> np.int32:
        acc = np.int32(0)
        match i:
            case 0:
                acc += np.int32(input[i][j][k][l]) * np.int32(self.filter[i][0][0][0])
                acc += np.int32(input[i][j][k+1][l]) * np.int32(self.filter[i][0][1][0])
                acc += np.int32(input[i][j][k+2][l]) * np.int32(self.filter[i][0][2][0])
                acc += np.int32(input[i][j+1][k][l]) * np.int32(self.filter[i][1][0][0])
                acc += np.int32(input[i][j+1][k+1][l]) * np.int32(self.filter[i][1][1][0])
                acc += np.int32(input[i][j+1][k+2][l]) * np.int32(self.filter[i][1][2][0])
                acc += np.int32(input[i][j+2][k][l]) * np.int32(self.filter[i][2][0][0])
                acc += np.int32(input[i][j+2][k+1][l]) * np.int32(self.filter[i][2][1][0])
                acc += np.int32(input[i][j+2][k+2][l]) * np.int32(self.filter[i][2][2][0])
            case _:
                # TODO check with more than one filters
                # FIXME The same of the moment
                acc += np.int32(input[i][j][k][l]) * np.int32(self.filter[i][0][0][0])
                acc += np.int32(input[i][j][k+1][l]) * np.int32(self.filter[i][0][1][0])
                acc += np.int32(input[i][j][k+2][l]) * np.int32(self.filter[i][0][2][0])
                acc += np.int32(input[i][j+1][k][l]) * np.int32(self.filter[i][1][0][0])
                acc += np.int32(input[i][j+1][k+1][l]) * np.int32(self.filter[i][1][1][0])
                acc += np.int32(input[i][j+1][k+2][l]) * np.int32(self.filter[i][1][2][0])
                acc += np.int32(input[i][j+2][k][l]) * np.int32(self.filter[i][2][0][0])
                acc += np.int32(input[i][j+2][k+1][l]) * np.int32(self.filter[i][2][1][0])
                acc += np.int32(input[i][j+2][k+2][l]) * np.int32(self.filter[i][2][2][0])
        return acc

    def conv2d(self, input: np.ndarray) -> np.ndarray:
        gen_code = True
        feature_map = np.zeros(self.output_shape, dtype=np.int8)
        for i in range(self.output_shape[0]): # output channels
            if gen_code:
                self.opt_conv_code = f"int32_t apply_filter_{i}(int8_t input[{self.input_shape[0]}][{self.input_shape[1]}][{self.input_shape[2]}][{self.input_shape[3]}], int i, int j, int k, int l) {{\n"
                self.opt_conv_code += f"    int32_t acc = 0;\n"
            for j in range(self.output_shape[1]): # output height
                for k in range(self.output_shape[2]): # output width
                    for l in range(self.output_shape[3]): # output depth
                        # feature_map[i][j][k][l] = self.output_zero_point
                        acc = np.int32(0)
                        # acc += self.__mul_by_filter__(input, i, j, k, l)
                        for m in range(self.filter.shape[1]):
                            for n in range(self.filter.shape[2]):
                                for o in range(self.filter.shape[3]):
                                    # feature_map[i][j][k][l] += input[i][m][n][o] * self.filter[i][m][n][o]
                                    acc += np.int32(input[i][j + m][k + n][l + o]) * np.int32(self.filter[i][m][n][o])
                                    if gen_code:
                                        self.opt_conv_code += f"    acc += input[i][j + {m}][k + {n}][l + {o}] * {self.filter[i][m][n][o]};\n"
                                    # print(f"acc += input[i][j + {m}][k + {n}][l + {o}] * filter[{i}][{m}][{n}][{o}]({self.filter[i][m][n][o]}) = {acc}")
                        ## acc = np.sum(np.int32(input[i, j:j+self.filter.shape[1], k:k+self.filter.shape[2], :]) * np.int32(self.filter[i]))
                        if gen_code:
                            self.opt_conv_code += f"    return acc;\n"
                            self.opt_conv_code += "}\n"
                            gen_code = False

                        # feature_map[i][j][k][l] += self.bias[i]
                        acc += self.bias[i]
                        # feature_map[i][j][k][l] = multiply_by_quantize_mul(feature_map[i][j][k][l], self.q_mantissa, self.exponent)
                        acc = multiply_by_quantize_mul(np.int64(acc), self.q_mantissa, self.exponent)
                        # feature_map[i][j][k][l] += self.output_zero_point
                        acc += self.output_zero_point
                        # feature_map[i][j][k][l] = np.clip(feature_map[i][j][k][l], -128, 127)
                        acc = np.clip(acc, -128, 127)
                        feature_map[i][j][k][l] = acc
            gen_code = True
        print(self.opt_conv_code)
        return feature_map
    
    def apply_layer(self, input: np.ndarray) -> np.ndarray:
        return self.conv2d(input)

    def generate_code(self):
        '''Generate C code for the convolution layer'''
        code = ""
        code += f"void conv2d(int8_t input[{self.input_shape[0]}][{self.input_shape[1]}][{self.input_shape[2]}][{self.input_shape[3]}], int8_t output[{self.output_shape[0]}][{self.output_shape[1]}][{self.output_shape[2]}][{self.output_shape[3]}])" + "{\n"
        code += f"    int8_t filter[{self.filter.shape[0]}][{self.filter.shape[1]}][{self.filter.shape[2]}][{self.filter.shape[3]}] = " + "{\n"
        for i in range(self.filter.shape[0]):
            code += "        {"
            for j in range(self.filter.shape[1]):
                code += " {"
                for k in range(self.filter.shape[2]):
                    code += " {"
                    for l in range(self.filter.shape[3]):
                        code += f" {self.filter[i][j][k][l]},"
                    code += " },"
                code += " },"
            code += " },\n"
        code += "    };\n"
        code += f"    const int32_t bias[{self.bias.shape[0]}] = " + "{"
        for i in range(self.bias.shape[0]):
            code += f" {self.bias[i]},"
        code += " };\n"
        code += f"    const int8_t output_zero_point = {self.output_zero_point};\n"
        code += f"    const int8_t input_zero_point = {self.input_zero_point};\n"
        code += f"    const int8_t filter_zero_point = {self.filter_zero_point};\n"
        code += f"    const int8_t bias_zero_point = {self.bias_zero_point};\n"
        code += f"    const int32_t q_mantissa = {self.q_mantissa};\n"
        code += f"    const int32_t exponent = {self.exponent};\n"
        code += f"    for(int i = 0; i < {self.output_shape[0]}; i++)" + "{\n"
        code += f"        for(int j = 0; j < {self.output_shape[1]}; j++)" + "{\n"
        code += f"            for(int k = 0; k < {self.output_shape[2]}; k++)" + "{\n"
        code += f"                for(int l = 0; l < {self.output_shape[3]}; l++)" + "{\n"
        code += f"                    int32_t acc = 0;\n"
        code += f"                    for(int m = 0; m < {self.filter.shape[1]}; m++)" + "{\n"
        code += f"                        for(int n = 0; n < {self.filter.shape[2]}; n++)" + "{\n"
        code += f"                            for(int o = 0; o < {self.filter.shape[3]}; o++)" + "{\n"
        code += f"                                acc += input[i][m + j][n + k][o] * filter[i][m][n][o];\n"
        code += "                            }\n"
        code += "                         }\n"
        code += "                     }\n"
        code += f"                    acc += bias[i];\n"
        code += f"                    acc = multiply_by_quantize_mul(acc, q_mantissa, exponent);\n"
        code += f"                    acc += output_zero_point;\n"
        code += f"                    acc = acc > 127 ? 127 : acc;\n"
        code += f"                    acc = acc < -128 ? -128 : acc;\n"
        code += f"                    output[i][j][k][l] = acc;\n"
        code += "                }\n"
        code += "            }\n"
        code += "        }\n"
        code += "    }\n"
        code += "}\n"
        return code

    def generate_opt_code(self):
        code = self.opt_conv_code
        code += f"void conv2d(int8_t input[{self.input_shape[0]}][{self.input_shape[1]}][{self.input_shape[2]}][{self.input_shape[3]}], int8_t output[{self.output_shape[0]}][{self.output_shape[1]}][{self.output_shape[2]}][{self.output_shape[3]}])" + "{\n"
        code += f"    int8_t filter[{self.filter.shape[0]}][{self.filter.shape[1]}][{self.filter.shape[2]}][{self.filter.shape[3]}] = " + "{\n"
        for i in range(self.filter.shape[0]):
            code += "        {"
            for j in range(self.filter.shape[1]):
                code += " {"
                for k in range(self.filter.shape[2]):
                    code += " {"
                    for l in range(self.filter.shape[3]):
                        code += f" {self.filter[i][j][k][l]},"
                    code += " },"
                code += " },"
            code += " },\n"
        code += "    };\n"
        code += f"    const int32_t bias[{self.bias.shape[0]}] = " + "{"
        for i in range(self.bias.shape[0]):
            code += f" {self.bias[i]},"
        code += " };\n"
        code += f"    const int8_t output_zero_point = {self.output_zero_point};\n"
        code += f"    const int8_t input_zero_point = {self.input_zero_point};\n"
        code += f"    const int8_t filter_zero_point = {self.filter_zero_point};\n"
        code += f"    const int8_t bias_zero_point = {self.bias_zero_point};\n"
        code += f"    const int32_t q_mantissa = {self.q_mantissa};\n"
        code += f"    const int32_t exponent = {self.exponent};\n"
        for i in range(self.output_shape[0]):
            code += f"    for(int j = 0; j < {self.output_shape[1]}; j++)" + "{\n"
            code += f"        for(int k = 0; k < {self.output_shape[2]}; k++)" + "{\n"
            code += f"            for(int l = 0; l < {self.output_shape[3]}; l++)" + "{\n"
            code += f"                int32_t acc = apply_filter_{i}(input, {i}, j, k, l);\n"
            code += f"                acc += bias[{i}];\n"
            code += f"                acc = multiply_by_quantize_mul(acc, q_mantissa, exponent);\n"
            code += f"                acc += output_zero_point;\n"
            code += f"                acc = acc > 127 ? 127 : acc;\n"
            code += f"                acc = acc < -128 ? -128 : acc;\n"
            code += f"                output[{i}][j][k][l] = acc;\n"
            code += "            }\n"
            code += "        }\n"
            code += "    }\n"
        code += "}\n"
        # code += f"    for(int i = 0; i < {self.output_shape[0]}; i++)" + "{\n"
        # code += f"        for(int j = 0; j < {self.output_shape[1]}; j++)" + "{\n"
        # code += f"            for(int k = 0; k < {self.output_shape[2]}; k++)" + "{\n"
        # code += f"                for(int l = 0; l < {self.output_shape[3]}; l++)" + "{\n"
        # code += f"                    int32_t acc = apply_filter_{i}(input, i, j, k, l);\n"
        # code += f"                    acc += bias[i];\n"
        # code += f"                    acc = multiply_by_quantize_mul(acc, q_mantissa, exponent);\n"
        # code += f"                    acc += output_zero_point;\n"
        # code += f"                    acc = acc > 127 ? 127 : acc;\n"
        # code += f"                    acc = acc < -128 ? -128 : acc;\n"
        # code += f"                    output[i][j][k][l] = acc;\n"
        # code += "                }\n"
        # code += "            }\n"
        # code += "        }\n"
        # code += "    }\n"
        # code += "}\n"
        return code


class MaxPool2D(Layer):
    def __init__(self, input_shape: tuple, output_shape: tuple) -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape

    def max_pool2d(self, input: np.ndarray) -> np.ndarray:
        maxpooled = np.zeros(self.output_shape, dtype=np.int8)
        for i in range(self.output_shape[0]):
            for j in range(self.output_shape[1]):
                for k in range(self.output_shape[2]):
                    for l in range(self.output_shape[3]):
                        maxpooled[i][j][k][l] = np.max(input[i][j*2:j*2+2, k*2:k*2+2, l])
        return maxpooled
    
    def apply_layer(self, input: np.ndarray) -> np.ndarray:
        return self.max_pool2d(input)
    
    def generate_code(self):
        '''Generate C code for the max pooling layer'''
        code = ""
        code += f"void max_pool2d(int8_t input[{self.input_shape[0]}][{self.input_shape[1]}][{self.input_shape[2]}][{self.input_shape[3]}], int8_t output[{self.output_shape[0]}][{self.output_shape[1]}][{self.output_shape[2]}][{self.output_shape[3]}])" + "{\n"
        code += f"    for(int i = 0; i < {self.output_shape[0]}; i++)" + "{\n"
        code += f"        for(int j = 0; j < {self.output_shape[1]}; j++)" + "{\n"
        code += f"            for(int k = 0; k < {self.output_shape[2]}; k++)" + "{\n"
        code += f"                for(int l = 0; l < {self.output_shape[3]}; l++)" + "{\n"
        code += f"                    int8_t max = input[i][j*2][k*2][l];\n"
        code += f"                    max = input[i][j*2][k*2+1][l] > max ? input[i][j*2][k*2+1][l] : max;\n"
        code += f"                    max = input[i][j*2+1][k*2][l] > max ? input[i][j*2+1][k*2][l] : max;\n"
        code += f"                    max = input[i][j*2+1][k*2+1][l] > max ? input[i][j*2+1][k*2+1][l] : max;\n"
        code += f"                    output[i][j][k][l] = max;\n"
        code += "                }\n"
        code += "            }\n"
        code += "        }\n"
        code += "    }\n"
        code += "}\n"
        return code

class Reshape(Layer):
    def __init__(self, input_shape: tuple, output_shape: tuple) -> None:
        self.input_shape = tuple(input_shape)
        self.output_shape = output_shape
    
    def reshape(self, input: np.ndarray) -> np.ndarray:
        # print(input.shape, self.input_shape, self.output_shape)
        if input.shape == self.input_shape:
            return input.reshape(self.output_shape)
        reshaped = np.zeros(self.output_shape, dtype=np.int8)
        for i in range(self.output_shape[0]):
            for j in range(self.output_shape[1]):
                for k in range(self.output_shape[2]):
                    reshaped[i][j][k] = input[i][j][k]
        return reshaped
    
    def apply_layer(self, input: np.ndarray) -> np.ndarray:
        return self.reshape(input)

class FullyConnected(Layer):
    def __init__(self, input_shape: tuple, output_shape: tuple) -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def set_fixed_point(self, mantissa: np.int32, exponent: np.int32):
        self.q_mantissa = mantissa
        self.exponent = exponent
    
    def set_zero_points(self, input_zero_point: np.int8, weight_zero_point: np.int8, bias_zero_point: np.int8, output_zero_point: np.int8):
        self.input_zero_point = input_zero_point
        self.weight_zero_point = weight_zero_point
        self.bias_zero_point = bias_zero_point
        self.output_zero_point = output_zero_point
    
    def set_weights(self, weights: list, dtype: str):
        self.weights = np.asarray(weights, dtype=to_np_dtype(dtype))

    def set_bias(self, bias: list, dtype: str):
        self.bias = np.asarray(bias, dtype=to_np_dtype(dtype))
    
    def __dot__(self, input: np.ndarray) -> np.ndarray:
        # print(input.shape, self.weights.shape)
        result = np.zeros(self.output_shape, dtype=np.int32)
        for i in range(self.output_shape[0]):
            for j in range(self.weights.shape[0]):
                for k in range(self.weights.shape[1]):
                    result[i][j] += np.int32(input[i][k]) * np.int32(self.weights[j][k])
        # print(result.shape)
        return result

    def fully_connected(self, input: np.ndarray) -> np.ndarray:
        dot_product = self.__dot__(input)
        dot_product += self.bias.transpose() # FIXME
        dot_product = multiply_by_quantize_mul(np.int64(dot_product), self.q_mantissa, self.exponent)
        dot_product += self.output_zero_point
        dot_product = np.clip(dot_product, -128, 127)
        dot_product = dot_product.astype(np.int8)
        return dot_product

    def apply_layer(self, input: np.ndarray) -> np.ndarray:
        return self.fully_connected(input)
    
    def generate_code(self):
        code = self.generate_dot()
        code += f"void fully_connected(int8_t input[{self.input_shape[0]}][{self.input_shape[1]}], int8_t output[{self.output_shape[1]}])" + "{\n"
        code += f"    const int8_t weights[{self.weights.shape[0]}][{self.weights.shape[1]}] = " + "{\n"
        for i in range(self.weights.shape[0]):
            code += "        {"
            for j in range(self.weights.shape[1]):
                code += f" {self.weights[i][j]},"
            code += " },\n"
        code += "    };\n"
        code += f"    const int32_t bias[{self.bias.shape[0]}] = " + "{"
        for i in range(self.bias.shape[0]):
            code += f" {self.bias[i][0]},"
        code += " };\n"
        code += f"    const int8_t output_zero_point = {self.output_zero_point};\n"
        code += f"    const int8_t input_zero_point = {self.input_zero_point};\n"
        code += f"    const int8_t weight_zero_point = {self.weight_zero_point};\n"
        code += f"    const int8_t bias_zero_point = {self.bias_zero_point};\n"
        code += f"    const int32_t q_mantissa = {self.q_mantissa};\n"
        code += f"    const int32_t exponent = {self.exponent};\n"
        code += f"    int32_t dot_result[{self.output_shape[1]}] = {{0}};\n"
        code += f"    dot_product(input, weights, dot_result);\n"
        code += f"    for(int i = 0; i < {self.output_shape[1]}; i++)" + "{\n"
        code += f"        dot_result[i] = dot_result[i] + bias[i];\n"
        code += f"        dot_result[i] = multiply_by_quantize_mul(dot_result[i], q_mantissa, exponent);\n"
        code += f"        dot_result[i] += output_zero_point;\n"
        code += f"        dot_result[i] = dot_result[i] > 127 ? 127 : dot_result[i];\n"
        code += f"        dot_result[i] = dot_result[i] < -128 ? -128 : dot_result[i];\n"
        code += f"        output[i] = dot_result[i];\n"
        code += "    }\n"
        code += "}\n"
        return code
    
    def generate_dot(self):
        code = ""
        code += f"void dot_product(int8_t input[{self.input_shape[0]}][{self.input_shape[1]}], const int8_t weights[{self.weights.shape[0]}][{self.weights.shape[1]}], int32_t dot_result[{self.weights.shape[0]}])" + "{\n"
        code += f"    for(int i = 0; i < {self.output_shape[0]}; i++)" + "{\n"
        code += f"        for(int j = 0; j < {self.weights.shape[0]}; j++)" + "{\n"
        code += f"            for(int k = 0; k < {self.weights.shape[1]}; k++)" + "{\n"
        code += f"                dot_result[j] += (int32_t)input[i][k] * (int32_t)weights[j][k];\n"
        code += "            }\n"
        code += "       }\n"
        code += "    }\n"
        code += "}\n"
        return code
    
    def generate_opt_dot(self):
        code = ""
        code += f"int32_t dot_product(int8_t input[{self.input_shape[0]}][{self.input_shape[1]}], int32_t out[{self.output_shape[1]}])" + "{\n"
        for i in range(self.weights.shape[0]):
            code += f"    out[{i}] = 0;\n"
            for j in range(self.weights.shape[1]):
                if(self.weights[i][j] == 0):
                    continue
                elif(self.weights[i][j] == 1):
                    code += f"    out[{i}] += input[{j}];\n"
                elif(self.weights[i][j] < 0):
                    code += f"    out[{i}] += multiply_n{abs(self.weights[i][j])}(input[{j}]);\n"
                else:
                    code += f"    out[{i}] += multiply_{self.weights[i][j]}(input[{j}]);\n"
        code += "}\n"
        return code
    
    


# Load the data JSON
extr_json = open("extracted.json", "r").read()
extracted = json.loads(extr_json)

mnist_train, mnist_test = mnist.load_data()
(x_train, y_train) = mnist_train
(x_test, y_test) = mnist_test

layers: list[Layer] = []

for layer in extracted["layers"]:
    if layer["type"] == "CONV_2D":
        conv2d = Conv2D(layer["input_shape"], layer["output_shape"])
        fixed_point = layer["fixed_point"]
        conv2d.set_fixed_point(fixed_point["mantissa"], fixed_point["exponent"])
        conv2d.set_zero_points(layer["z_input"], layer["z_weight"], layer["z_bias"], layer["z_output"])
        conv2d.set_filter(layer["weights"]["data"], layer["weights"]["dtype"])
        conv2d.set_bias(layer["bias"]["data"], layer["bias"]["dtype"])
        layers.append(conv2d)
    elif layer["type"] == "MAX_POOL_2D":
        maxpool2d = MaxPool2D(layer["input_shape"], layer["output_shape"])
        layers.append(maxpool2d)
    elif layer["type"] == "QUANTIZE":
        quantize = Quantize(np.int8(-128), layer["s_input"])
        layers.append(quantize)
    elif layer["type"] == "RESHAPE":
        reshape = Reshape(layer["input_shape"], layer["output_shape"])
        layers.append(reshape)
    elif layer["type"] == "FULLY_CONNECTED":
        fc = FullyConnected(layer["input_shape"], layer["output_shape"])
        fixed_point = layer["fixed_point"]
        fc.set_fixed_point(fixed_point["mantissa"], fixed_point["exponent"])
        fc.set_zero_points(layer["z_input"], layer["z_weight"], layer["z_bias"], layer["z_output"])
        fc.set_weights(layer["weights"]["data"], layer["weights"]["dtype"])
        fc.set_bias(layer["bias"]["data"], layer["bias"]["dtype"])
        layers.append(fc)
    else:
        print("Unknown layer type:", layer["type"])


def predict(model: list[Layer], x_test: np.ndarray, y_test: np.ndarray):
    table = PrettyTable()
    table.field_names = ["Predicted", "Original", "Confidence", "Match"]
    for i in range(10):
        predictions = x_test[i]
        predictions = predictions.reshape(1, 28, 28, 1)
        for layer in model:
            predictions = layer.apply_layer(predictions)
        table.add_row(
            [
                np.argmax(predictions),
                y_test[i],
                np.max(predictions),
                "✅" if np.argmax(predictions) == y_test[i] else "❌",
            ]
        )
    print(table)

def evaluate(model: list[Layer], x_test: np.ndarray, y_test: np.ndarray):
    correct = 0
    for i in range(len(x_test)):
        predictions = x_test[i]
        predictions = predictions.reshape(1, 28, 28, 1)
        for layer in model:
            predictions = layer.apply_layer(predictions)
        if np.argmax(predictions) == y_test[i]:
            correct += 1
    print("Accuracy:", correct / len(x_test))

def check_image(model: list[Layer], x_test: np.ndarray, y_test: np.ndarray, index: int):
    predictions = x_test[index]
    predictions = predictions.reshape(1, 28, 28, 1)
    for layer in model:
        predictions = layer.apply_layer(predictions)
    print(predictions)
    print("Predicted:", np.argmax(predictions))
    print("Original:", y_test[index])



check_image(layers, x_test, y_test, 8)
# predict(layers, x_test[:10], y_test[:10])
# evaluate(layers, x_test, y_test)

# print(layers[1].generate_code())
print(layers[1].generate_opt_code())
# print(layers[4].generate_opt_dot())
