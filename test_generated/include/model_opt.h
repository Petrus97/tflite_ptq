#ifndef MODEL_OPT_H
#define MODEL_OPT_H
#include <stdint.h>
#include <stdlib.h>

typedef union byte {
    uint8_t u8[784];
    int8_t i8[784];
} byte_t;

void quantize(byte_t* image);
void conv2d_0(int8_t input[1][28][28][1], int8_t output[1][26][26][4]);
void max_pool2d_0(int8_t input[1][26][26][4], int8_t output[1][13][13][4]);
void conv2d_1(int8_t input[1][13][13][4], int8_t output[1][12][12][2]);
void max_pool2d_1(int8_t input[1][12][12][2], int8_t output[1][6][6][2]);
void fully_connected(int8_t input[1][72], int8_t output[10]);

#endif
