#ifndef FILTERS_H
#define FILTERS_H

#include <stdint.h>

int32_t apply_filter_0(int8_t input[1][28][28][1], int out_h, int out_w, int ch_idx);
int32_t apply_filter_1(int8_t input[1][28][28][1], int out_h, int out_w, int ch_idx);

#endif