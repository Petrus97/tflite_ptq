#ifndef MODEL_OPT_H
#define MODEL_OPT_H
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef union byte {
    uint8_t u8[784];
    int8_t i8[784];
} byte_t;

void quantize(byte_t* image);
void conv2d(int8_t input[1][28][28][1], int8_t output[1][26][26][2]);
void max_pool2d(int8_t input[1][26][26][2], int8_t output[1][13][13][2]);
void fully_connected(int8_t input[1][338], int8_t output[10]);

#endif
