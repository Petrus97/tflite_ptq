import numpy as np
import tensorflow as tf
import keras.datasets.mnist as mnist
from prettytable import PrettyTable
import json
from src.logger import logging

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

    def generate_opt_code(self):
        return ""

class Quantize(Layer):
    def __init__(self, input_shape, Z_input: np.int8 = np.int8(0), S_input: np.float64 = np.float64(0)):
        self.array: np.ndarray = np.ndarray([])
        self.input_shape = input_shape
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
    
    def generate_code(self):
        input_dim = 1
        for shape in self.input_shape:
            input_dim *= shape
        # TODO Generate Header files
        # code = "typedef union byte {\n"
        # code += f"    uint8_t u8[{input_dim}];\n"
        # code += f"    int8_t i8[{input_dim}];\n"
        # code += "} byte_t;\n\n"
        code = "void quantize(byte_t* image)" + "{\n"
        code += "    for(size_t i = 0; i < 784; i++)" + "{\n"
        code += f"        image->i8[i] = image->u8[i] + ({self.Z_input});\n"
        code += "    }\n"
        code += "}\n\n"
        code += "static int32_t multiply_by_quantize_mul(int64_t acc, int32_t q_mantissa, int32_t exp)" + "{\n"
        code += "    const int32_t reduced_mantissa = q_mantissa < 0x7FFF0000 ? ((q_mantissa + (1 << 15)) >> 16) : 0x7FFF;\n"
        code += "    const int64_t total_shifts = 15 - exp;\n"
        code += "    const int64_t round = (int64_t)(1) << (total_shifts - 1);\n"
        code += "    acc = acc * (int64_t)reduced_mantissa;\n"
        code += "    acc = acc + round;\n"
        code += "    int32_t result = acc >> total_shifts;\n"
        code += "    return result;\n"
        code += "}\n"
        return code
    
    def generate_opt_code(self):
        return self.generate_code()

class Conv2D(Layer):
    def __init__(self, input_shape: tuple, output_shape: tuple):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.code = ""
    
    def set_fixed_point(self, mantissa: np.int32, exponent: np.int32):
        self.q_mantissa = mantissa
        self.exponent = exponent

    def set_fixed_points(self, fixed_points: list):
        self.fixed_points = fixed_points

    def set_zero_points(self, input_zero_point: np.int8, filter_zero_point: np.int8, bias_zero_point: np.int8, output_zero_point: np.int8):
        self.input_zero_point = input_zero_point
        self.filter_zero_point = filter_zero_point
        self.bias_zero_point = bias_zero_point
        self.output_zero_point = output_zero_point

    def set_filter(self, filter: list, dtype: str):
        self.filter = np.asarray(filter, dtype=to_np_dtype(dtype))
    
    def set_bias(self, bias: list, dtype: str):
        self.bias = np.asarray(bias, dtype=to_np_dtype(dtype))


    def conv_by_filter(self, input: np.ndarray, filter: np.ndarray, ch_idx: int, fixed_point: dict, output_zero_point: np.int8, feature_map: np.ndarray) -> np.ndarray:
        out_height = input.shape[1] - filter.shape[1] + 1
        out_width = input.shape[2] - filter.shape[2] + 1
        filter_height = filter.shape[1]
        filter_width = filter.shape[2]
        filter_depth = filter.shape[3]
        # Code generation
        self.opt_conv_code = ""
        self.opt_conv_code += f"int32_t apply_filter_{ch_idx}(int8_t input[{self.input_shape[0]}][{self.input_shape[1]}][{self.input_shape[2]}][{self.input_shape[3]}], int out_h, int out_w, int ch_idx) " + "{\n"
        self.opt_conv_code += f"    int32_t acc = 0;\n"
        # self.opt_conv_code = f"int32_t apply_filter_{ch_idx}(int8_t input[{self.input_shape[0]}][{self.input_shape[1]}][{self.input_shape[2]}][{self.input_shape[3]}], int ch_idx, int8_t feature_map[{self.output_shape[0]}][{self.output_shape[1]}][{self.output_shape[2]}][{self.output_shape[3]}]) " + "{\n"
        # self.opt_conv_code += f"    for(int out_h = 0; out_h < {out_height}; out_h++)" + "{\n"
        # self.opt_conv_code += f"        for(int out_w = 0; out_w < {out_width}; out_w++)" + "{\n"
        # self.opt_conv_code += f"            int32_t acc = 0;\n"
        for out_h in range(out_height):
            for out_w in range(out_width):
                acc = np.int32(0)
                for cin in range(filter_depth):
                    for f_h in range(filter_height):
                        for f_w in range(filter_width):
                            acc += np.int32(input[0][out_h+f_h][out_w+f_w][cin]) * np.int32(filter[ch_idx][f_h][f_w][cin])
                acc += self.bias[ch_idx]
                acc = multiply_by_quantize_mul(np.int64(acc), fixed_point["mantissa"], fixed_point["exponent"])
                acc += output_zero_point
                acc = np.clip(acc, -128, 127)
                feature_map[0][out_h][out_w][ch_idx] = acc
        for cin in range(filter_depth):
            for f_h in range(filter_height):
                for f_w in range(filter_width):
                    # self.opt_conv_code += f"    acc += input[0][out_h + {f_h}][out_w + {f_w}][{cin}] * {filter[ch_idx][f_h][f_w][cin]}; // filter[{ch_idx}][{f_h}][{f_w}][{cin}];\n"
                    if(filter[ch_idx][f_h][f_w][cin] == 0):
                        continue
                    elif(filter[ch_idx][f_h][f_w][cin] == 1):
                        self.opt_conv_code += f"    acc += input[0][out_h + {f_h}][out_w + {f_w}][{cin}];\n"
                    elif(filter[ch_idx][f_h][f_w][cin] < 0):
                        self.opt_conv_code += f"    acc += multiply_n{abs(filter[ch_idx][f_h][f_w][cin])}(input[0][out_h + {f_h}][out_w + {f_w}][{cin}]);\n"
                    else:
                        self.opt_conv_code += f"    acc += multiply_{filter[ch_idx][f_h][f_w][cin]}(input[0][out_h + {f_h}][out_w + {f_w}][{cin}]);\n"

        self.opt_conv_code += f"    return acc;\n"
        self.opt_conv_code += "}\n"
        self.code += self.opt_conv_code
        # self.opt_conv_code += f"            acc += bias[{ch_idx}];\n"
        # self.opt_conv_code += f"            acc = multiply_by_quantize_mul(acc, {fixed_point['mantissa']}, {fixed_point['exponent']});\n"
        # self.opt_conv_code += f"            acc += output_zero_point;\n"
        # self.opt_conv_code += f"            acc = acc > 127 ? 127 : acc;\n"
        # self.opt_conv_code += f"            acc = acc < -128 ? -128 : acc;\n"
        # self.opt_conv_code += f"            feature_map[0][out_h][out_w][{ch_idx}] = acc;\n"
        # self.opt_conv_code += "        }\n"
        # self.opt_conv_code += "    }\n"
        # self.opt_conv_code += "}\n"
        return feature_map

    def conv2d(self, input: np.ndarray) -> np.ndarray:
        input.flags.writeable = False # make input read-only
        # logging.info(f"input shape: {input.shape}, {len(input.shape)}, {self.input_shape}")
        # logging.info(f"filter shape: {self.filter.shape}")
        # logging.info(f"bias shape: {self.bias.shape}")
        # logging.info(f"output shape: {self.output_shape}")
        # logging.info(f"bias {self.bias}")
        gen_code = True
        feature_map = np.zeros(self.output_shape, dtype=np.int8)
        # assert dimensions
        assert(len(input.shape) == 4)
        assert(len(self.filter.shape) == 4)
        assert(len(self.bias.shape) == 1)
        assert(len(self.output_shape) == 4)
        # Extract input dimensions
        input_batch_size = input.shape[0]
        input_height = input.shape[1]
        input_width = input.shape[2]
        input_depth = input.shape[3]
        # Extract filter dimensions
        filter_number = self.filter.shape[0]
        filter_height = self.filter.shape[1]
        filter_width = self.filter.shape[2]
        filter_depth = self.filter.shape[3]
        # Extract output dimensions
        output_batch_size = self.output_shape[0]
        output_height = self.output_shape[1]
        output_width = self.output_shape[2]
        output_channels = self.output_shape[3]
        # assert dimensions
        assert(output_channels == filter_number)
        assert(filter_depth == input_depth)
        assert(input_batch_size == output_batch_size and input_batch_size == 1) # only single core batch execution
        
        # Since we asserted that we have a single batch size, we can iterate over the output channels
        for cout in range(output_channels):
            feature_map = self.conv_by_filter(input, self.filter, cout, self.fixed_points[cout], self.output_zero_point, feature_map)

        line_width = np.get_printoptions()['linewidth']
        threshold = np.get_printoptions()['threshold']
        np.set_printoptions(linewidth=np.inf, threshold=np.inf)
        f1 = feature_map[0,:,:,0]
        f2 = feature_map[0,:,:,1]
        print(f1)
        print(f2)
        np.set_printoptions(linewidth=line_width, threshold=threshold)
        # print(self.code)
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
        code += f"    const int32_t fixed_points[{self.filter.shape[3]}][2] = " + "{\n"
        for i in range(self.output_shape[0]):
            code += f"        {{ {self.fixed_points[i]['mantissa']}, {self.fixed_points[i]['exponent']} }},\n"
        code += "    };\n"
        code += f"    for(int cout = 0; cout < {self.output_shape[3]}; cout++)" + "{\n"
        code += f"        for(int out_h = 0; out_h < {self.output_shape[1]}; out_h++)" + "{\n"
        code += f"            for(int out_w = 0; out_w < {self.output_shape[2]}; out_w++)" + "{\n"
        code += f"                int32_t acc = 0;\n"
        code += f"                for(int f_h = 0; f_h < {self.filter.shape[1]}; f_h++)" + "{\n"
        code += f"                    for(int f_w = 0; f_w < {self.filter.shape[2]}; f_w++)" + "{\n"
        code += f"                        for(int f_d = 0; f_d < {self.filter.shape[3]}; f_d++)" + "{\n"
        code += f"                            acc += input[0][out_h + f_h][out_w + f_w][f_d] * filter[cout][f_h][f_w][f_d];\n"
        code += "                        }\n"
        code += "                    }\n"
        code += "                 }\n"
        code += f"                acc += bias[cout];\n"
        code += f"                acc = multiply_by_quantize_mul(acc, fixed_points[cout][0], fixed_points[cout][1]);\n"
        code += f"                acc += output_zero_point;\n"
        code += f"                acc = acc > 127 ? 127 : acc;\n"
        code += f"                acc = acc < -128 ? -128 : acc;\n"
        code += f"                output[0][out_h][out_w][cout] = acc;\n"
        code += "            }\n"
        code += "        }\n"
        code += "    }\n"
        code += "}\n"
        return code

    def generate_opt_code(self):
        code = self.code
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
        code += f"    const int32_t fixed_points[{self.filter.shape[3]}][2] = " + "{\n"
        for i in range(self.output_shape[0]):
            code += f"        {{ {self.fixed_points[i]['mantissa']}, {self.fixed_points[i]['exponent']} }},\n"
        code += "    };\n"
        # code += f"    const int32_t q_mantissa = {self.q_mantissa};\n"
        # code += f"    const int32_t exponent = {self.exponent};\n"
        for i in range(self.output_shape[3]):
            code += f"    for(int out_h = 0; out_h < {self.output_shape[1]}; out_h++)" + "{\n"
            code += f"        for(int out_w = 0; out_w < {self.output_shape[2]}; out_w++)" + "{\n"
            code += f"            int32_t acc = apply_filter_{i}(input, out_h, out_w, {i});\n"
            code += f"            acc += bias[{i}];\n"
            code += f"            acc = multiply_by_quantize_mul(acc, fixed_points[{i}][0], fixed_points[{i}][1]);\n"
            code += f"            acc += output_zero_point;\n"
            code += f"            acc = acc > 127 ? 127 : acc;\n"
            code += f"            acc = acc < -128 ? -128 : acc;\n"
            code += f"            output[0][out_h][out_w][{i}] = acc;\n"
            code += "        }\n"
            code += "    }\n"
        code += "}\n"
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
        line_width = np.get_printoptions()['linewidth']
        threshold = np.get_printoptions()['threshold']
        np.set_printoptions(linewidth=np.inf, threshold=np.inf)
        m1 = maxpooled[0,:,:,0]
        m2 = maxpooled[0,:,:,1]
        # print(m1)
        # print(m2)
        np.set_printoptions(linewidth=line_width, threshold=threshold)
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

    def generate_opt_code(self):
        return self.generate_code()

class Reshape(Layer):
    def __init__(self, input_shape: tuple, output_shape: tuple) -> None:
        self.input_shape = tuple(input_shape)
        self.output_shape = output_shape
    
    def reshape(self, input: np.ndarray) -> np.ndarray:
        # print(input.shape, self.input_shape, self.output_shape)
        if input.shape == self.input_shape:
            return input.reshape(self.output_shape)
        reshaped = np.zeros(self.output_shape, dtype=np.int8)
        logging.info(f"reshaped shape: {reshaped.shape}")
        logging.info(f"input shape: {input.shape}")
        for i in range(self.input_shape[0]):
            for j in range(self.input_shape[1]):
                for k in range(self.input_shape[2]):
                    reshaped[0][i][j][k] = input[0][i][j][k]
        return reshaped
    
    def apply_layer(self, input: np.ndarray) -> np.ndarray:
        return self.reshape(input)
    
    def generate_code(self):
        return super().generate_code()

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
        # print(input)
        dot_product = self.__dot__(input)
        dot_product += self.bias.transpose() # FIXME
        dot_product = multiply_by_quantize_mul(np.int64(dot_product), self.q_mantissa, self.exponent)
        dot_product += self.output_zero_point
        dot_product = np.clip(dot_product, -128, 127)
        dot_product = dot_product.astype(np.int8)
        return dot_product

    def apply_layer(self, input: np.ndarray) -> np.ndarray:
        return self.fully_connected(input)
    
    def generate_code(self, opt: bool = False):
        code = self.generate_dot() if opt == False else self.generate_opt_dot()
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
        if opt == False:
            code += f"    dot_product(input, weights, dot_result);\n"
        else:
            code += f"    dot_product(input, dot_result);\n"
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
        code += f"static void dot_product(int8_t input[{self.input_shape[0]}][{self.input_shape[1]}], const int8_t weights[{self.weights.shape[0]}][{self.weights.shape[1]}], int32_t dot_result[{self.weights.shape[0]}])" + "{\n"
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
        code += f"static void dot_product(int8_t input[{self.input_shape[0]}][{self.input_shape[1]}], int32_t out[{self.output_shape[1]}])" + "{\n"
        for i in range(self.weights.shape[0]):
            code += f"    out[{i}] = 0;\n"
            for j in range(self.weights.shape[1]):
                if(self.weights[i][j] == 0):
                    continue
                elif(self.weights[i][j] == 1):
                    code += f"    out[{i}] += input[0][{j}];\n"
                elif(self.weights[i][j] < 0):
                    code += f"    out[{i}] += multiply_n{abs(self.weights[i][j])}(input[0][{j}]);\n"
                else:
                    code += f"    out[{i}] += multiply_{self.weights[i][j]}(input[0][{j}]);\n"
        code += "}\n"
        return code
    
    def generate_opt_code(self):
        return self.generate_code(opt=True)


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
        # conv2d.set_fixed_point(fixed_point["mantissa"], fixed_point["exponent"])
        conv2d.set_fixed_points(fixed_point)
        conv2d.set_zero_points(layer["z_input"], layer["z_weight"], layer["z_bias"], layer["z_output"])
        conv2d.set_filter(layer["weights"]["data"], layer["weights"]["dtype"])
        conv2d.set_bias(layer["bias"]["data"], layer["bias"]["dtype"])
        layers.append(conv2d)
    elif layer["type"] == "MAX_POOL_2D":
        maxpool2d = MaxPool2D(layer["input_shape"], layer["output_shape"])
        layers.append(maxpool2d)
    elif layer["type"] == "QUANTIZE":
        quantize = Quantize(layer["input_shape"], Z_input=np.int8(-128), S_input=layer["s_input"])
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

# code = ""
# code += "#include <stdint.h>\n"
# code += "#include <stdio.h>\n"
# code += "#include <stdlib.h>\n"
code = ""
for layer in layers:
    code += layer.generate_opt_code()
    code += "\n"
with open("test_generated/src/model_opt.c", "w") as f:
    f.write(code)