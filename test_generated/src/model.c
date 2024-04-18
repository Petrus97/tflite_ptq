#include "model.h"

void quantize(byte_t* image){
    for(size_t i = 0; i < 784; i++){
        image->i8[i] = image->u8[i] + (-128);
    }
}

static int32_t multiply_by_quantize_mul(int64_t acc, int32_t q_mantissa, int32_t exp){
    const int32_t reduced_mantissa = q_mantissa < 0x7FFF0000 ? ((q_mantissa + (1 << 15)) >> 16) : 0x7FFF;
    const int64_t total_shifts = 15 - exp;
    const int64_t round = (int64_t)(1) << (total_shifts - 1);
    acc = acc * (int64_t)reduced_mantissa;
    acc = acc + round;
    int32_t result = acc >> total_shifts;
    return result;
}

void conv2d_0(int8_t input[1][28][28][1], int8_t output[1][26][26][4]){
    int8_t filter[4][3][3][1] = {
        { { { 127, }, { 47, }, { 69, }, }, { { 96, }, { -36, }, { -75, }, }, { { 76, }, { 32, }, { -43, }, }, },
        { { { -32, }, { -10, }, { 41, }, }, { { -35, }, { -72, }, { 127, }, }, { { -92, }, { -20, }, { 80, }, }, },
        { { { 57, }, { 101, }, { 41, }, }, { { 81, }, { 84, }, { 127, }, }, { { 27, }, { -28, }, { 12, }, }, },
        { { { 127, }, { 4, }, { -50, }, }, { { 80, }, { -47, }, { -88, }, }, { { 48, }, { 63, }, { -84, }, }, },
    };
    const int32_t bias[4] = { 37504, -1664, 64256, 6784, };
    const int8_t output_zero_point = -128;
    const int8_t input_zero_point = -128;
    const int8_t filter_zero_point = 0;
    const int8_t bias_zero_point = 0;
    const int32_t fixed_points[4][2] = {
        { 1847679730, -9 },
        { 1895443745, -10 },
        { 2147030750, -9 },
        { 1734748036, -9 },
    };
    for(int cout = 0; cout < 4; cout++){
        for(int out_h = 0; out_h < 26; out_h++){
            for(int out_w = 0; out_w < 26; out_w++){
                int32_t acc = 0;
                for(int f_h = 0; f_h < 3; f_h++){
                    for(int f_w = 0; f_w < 3; f_w++){
                        for(int f_d = 0; f_d < 1; f_d++){
                            acc += input[0][out_h + f_h][out_w + f_w][f_d] * filter[cout][f_h][f_w][f_d];
                        }
                    }
                 }
                acc += bias[cout];
                acc = multiply_by_quantize_mul(acc, fixed_points[cout][0], fixed_points[cout][1]);
                acc += output_zero_point;
                acc = acc > 127 ? 127 : acc;
                acc = acc < -128 ? -128 : acc;
                output[0][out_h][out_w][cout] = acc;
            }
        }
    }
}

void max_pool2d_0(int8_t input[1][26][26][4], int8_t output[1][13][13][4]){
    for(int i = 0; i < 1; i++){
        for(int j = 0; j < 13; j++){
            for(int k = 0; k < 13; k++){
                for(int l = 0; l < 4; l++){
                    int8_t max = input[i][j*2][k*2][l];
                    max = input[i][j*2][k*2+1][l] > max ? input[i][j*2][k*2+1][l] : max;
                    max = input[i][j*2+1][k*2][l] > max ? input[i][j*2+1][k*2][l] : max;
                    max = input[i][j*2+1][k*2+1][l] > max ? input[i][j*2+1][k*2+1][l] : max;
                    output[i][j][k][l] = max;
                }
            }
        }
    }
}

void conv2d_1(int8_t input[1][13][13][4], int8_t output[1][12][12][2]){
    int8_t filter[2][2][2][4] = {
        { { { 127, 55, 73, 106, }, { -33, -56, -74, 63, }, }, { { 33, -60, 97, 18, }, { 39, -25, 85, -28, }, }, },
        { { { -81, 4, -79, -75, }, { -83, -16, -35, -103, }, }, { { 71, -29, 127, 31, }, { 56, -44, 90, -12, }, }, },
    };
    const int32_t bias[2] = { 53760, -22784, };
    const int8_t output_zero_point = -128;
    const int8_t input_zero_point = -128;
    const int8_t filter_zero_point = 0;
    const int8_t bias_zero_point = 0;
    const int32_t fixed_points[2][2] = {
        { 1475699733, -8 },
        { 1878952878, -8 },
    };
    for(int cout = 0; cout < 2; cout++){
        for(int out_h = 0; out_h < 12; out_h++){
            for(int out_w = 0; out_w < 12; out_w++){
                int32_t acc = 0;
                for(int f_h = 0; f_h < 2; f_h++){
                    for(int f_w = 0; f_w < 2; f_w++){
                        for(int f_d = 0; f_d < 4; f_d++){
                            acc += input[0][out_h + f_h][out_w + f_w][f_d] * filter[cout][f_h][f_w][f_d];
                        }
                    }
                 }
                acc += bias[cout];
                acc = multiply_by_quantize_mul(acc, fixed_points[cout][0], fixed_points[cout][1]);
                acc += output_zero_point;
                acc = acc > 127 ? 127 : acc;
                acc = acc < -128 ? -128 : acc;
                output[0][out_h][out_w][cout] = acc;
            }
        }
    }
}

void max_pool2d_1(int8_t input[1][12][12][2], int8_t output[1][6][6][2]){
    for(int i = 0; i < 1; i++){
        for(int j = 0; j < 6; j++){
            for(int k = 0; k < 6; k++){
                for(int l = 0; l < 2; l++){
                    int8_t max = input[i][j*2][k*2][l];
                    max = input[i][j*2][k*2+1][l] > max ? input[i][j*2][k*2+1][l] : max;
                    max = input[i][j*2+1][k*2][l] > max ? input[i][j*2+1][k*2][l] : max;
                    max = input[i][j*2+1][k*2+1][l] > max ? input[i][j*2+1][k*2+1][l] : max;
                    output[i][j][k][l] = max;
                }
            }
        }
    }
}


static void dot_product(int8_t input[1][72], const int8_t weights[10][72], int32_t dot_result[10]){
    for(int i = 0; i < 1; i++){
        for(int j = 0; j < 10; j++){
            for(int k = 0; k < 72; k++){
                dot_result[j] += (int32_t)input[i][k] * (int32_t)weights[j][k];
            }
       }
    }
}
void fully_connected(int8_t input[1][72], int8_t output[10]){
    const int8_t weights[10][72] = {
        { 29, -14, -9, 18, 3, -40, 22, 15, 4, -15, 5, -21, -35, 15, -32, 2, -9, 2, 34, -22, 50, 3, 14, 16, -17, 23, 10, 20, 33, -26, -4, -82, -11, -51, 37, 0, 33, -16, 11, -2, 34, -53, -127, -19, -28, 23, 39, -2, 27, -27, 16, -48, 51, -7, -2, 37, 12, 10, -16, -36, -10, 18, 20, 5, 64, 4, -25, -2, -23, -27, 15, 32, },
        { -3, 10, 22, -5, 24, -24, 19, -40, 12, -18, -16, 41, -10, 38, 16, -20, -30, -17, 14, -15, -37, -53, 19, -32, 26, 25, -10, -31, -43, 20, 90, 19, 18, -69, -59, -34, 26, -1, -38, -27, -14, -20, 72, -17, -41, -66, -16, 23, 12, -9, -3, 0, -13, 17, 91, -37, -42, -5, -5, -25, 6, -10, 27, -1, 12, -18, -42, -13, 39, 7, 43, -18, },
        { -5, -4, 8, 2, -2, 11, 12, 5, -17, -19, -12, -22, 19, 30, 29, 22, 38, -13, 13, -24, 5, -44, -4, -32, 2, -32, -36, 53, -80, 34, -45, 10, 5, -27, 33, -32, 8, 48, -72, 54, -18, 40, 17, 59, -5, -5, -23, 80, 31, 16, 36, -15, 44, 17, 26, 14, -9, 55, 9, 91, -15, -22, -3, 16, 11, 8, -41, 6, 4, -4, 3, -12, },
        { -25, 45, -1, 25, 12, 22, 9, -15, -34, -29, -37, -37, 33, 38, 39, 21, -8, 31, 12, 14, 41, -45, 31, -50, -9, -35, -74, 37, -95, 81, 47, 59, -6, 3, -21, -28, 30, 17, -53, 22, -27, -17, -8, 19, 5, 5, -13, 32, 49, 3, 13, 19, -49, 52, -52, 59, 35, 38, -2, -55, 61, -15, 11, 24, 5, 65, 8, 8, 15, 9, 8, -8, },
        { -7, 11, -32, 41, -19, -22, -25, -22, 15, 0, 30, 21, 6, 26, -14, -41, 37, -55, -7, -105, -19, 0, 21, 10, 21, -41, 14, 6, 101, -13, -7, -11, 72, -30, -11, 5, 21, 10, 51, -12, 52, -12, 10, 33, 65, -12, 19, 4, -26, -40, -45, -2, -68, -8, -10, -41, -52, -64, -23, 13, 34, -14, -41, 30, -10, 27, 8, -63, 17, 16, 21, 34, },
        { 25, -27, -1, 4, 3, -5, -45, 16, 35, -15, -14, 35, -18, 23, -13, -21, 37, 19, 18, 2, -7, 14, 20, 118, -18, -21, 10, -48, 88, 5, 29, 36, -92, 43, -115, 66, 23, 21, 18, -21, 10, -40, -52, -1, -39, -31, -39, -3, 8, 26, 16, 70, -31, 63, -15, 53, -2, -1, 52, 5, -26, 17, -51, 24, 5, 63, -1, 53, 13, -2, 3, 7, },
        { -37, -7, 13, -1, 12, -23, 9, 1, 52, -18, 36, 29, 18, -15, 1, -57, -21, 7, -41, -1, -57, -21, -31, 4, 14, -21, -3, 5, 36, -27, -12, 42, -86, 36, 22, 0, -44, -8, 18, -38, 37, -7, 18, 11, -19, -12, 25, -23, -15, -50, 14, -73, 104, -23, 56, 49, 26, 18, 24, -40, -6, 20, -49, -31, 9, -46, -46, -40, -8, -9, -19, -17, },
        { -5, 7, 1, 15, 16, -27, -21, -24, -69, -3, -35, -11, 43, -11, 49, 22, 71, 25, 49, 37, 28, 32, 26, -9, 52, -17, -1, 1, -55, 2, -10, -27, 91, -17, -13, -10, 2, -10, -6, 1, -86, -29, -58, 40, 76, -13, 19, -18, 20, 14, -42, -25, -33, -2, -18, -42, -40, -32, -19, -51, -28, -24, 24, 62, -6, 10, 43, -30, -21, -20, -42, 34, },
        { -38, -13, 27, -10, -7, 8, 9, 8, -11, 23, -4, -6, -19, 40, 18, -6, -5, 19, -16, -25, -11, -11, 5, 51, 5, -3, 22, -36, 62, -38, -10, 33, -22, 7, 25, 16, 20, -20, -51, 24, 1, 45, 56, -48, -20, -23, -36, 5, 6, -21, -5, -31, 20, -28, -36, 4, -23, 9, 18, -17, -37, -16, 33, -41, 37, 4, 36, 37, 17, -26, 29, 29, },
        { -34, -4, -21, -21, 32, -1, -38, 49, -43, -4, -35, -32, -45, 14, -16, 26, -49, 71, 30, 66, 6, -1, -4, -30, 26, -24, 2, -21, 42, -18, -26, 41, 48, 13, 13, -41, -7, -7, 29, -64, 6, 34, -22, -17, 53, -86, 15, -55, 21, -39, -8, -14, -66, 9, -28, -3, -23, -43, 0, -43, -25, -35, -28, 54, -14, 32, -14, -29, -6, -9, 39, 33, },
    };
    const int32_t bias[10] = { -7025, -32464, 42253, 46566, -15872, 49541, -43402, -18417, 3873, -49757, };
    const int8_t output_zero_point = -18;
    const int8_t input_zero_point = -128;
    const int8_t weight_zero_point = 0;
    const int8_t bias_zero_point = 0;
    const int32_t q_mantissa = 2072683058;
    const int32_t exponent = -9;
    int32_t dot_result[10] = {0};
    dot_product(input, weights, dot_result);
    for(int i = 0; i < 10; i++){
        dot_result[i] = dot_result[i] + bias[i];
        dot_result[i] = multiply_by_quantize_mul(dot_result[i], q_mantissa, exponent);
        dot_result[i] += output_zero_point;
        dot_result[i] = dot_result[i] > 127 ? 127 : dot_result[i];
        dot_result[i] = dot_result[i] < -128 ? -128 : dot_result[i];
        output[i] = dot_result[i];
    }
}

