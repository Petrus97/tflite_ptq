#include "filters.h"
#include "multiply_inlined.h"

int32_t apply_filter_0(int8_t input[1][28][28][1], int out_h, int out_w, int ch_idx) {
    int32_t acc = 0;
    acc += multiply_n1(input[0][out_h + 0][out_w + 0][0]);
    acc += multiply_51(input[0][out_h + 0][out_w + 1][0]);
    acc += multiply_n12(input[0][out_h + 0][out_w + 2][0]);
    acc += multiply_89(input[0][out_h + 1][out_w + 0][0]);
    acc += multiply_69(input[0][out_h + 1][out_w + 1][0]);
    acc += multiply_41(input[0][out_h + 1][out_w + 2][0]);
    acc += multiply_71(input[0][out_h + 2][out_w + 0][0]);
    acc += multiply_98(input[0][out_h + 2][out_w + 1][0]);
    acc += multiply_127(input[0][out_h + 2][out_w + 2][0]);
    return acc;
}

int32_t apply_filter_1(int8_t input[1][28][28][1], int out_h, int out_w, int ch_idx) {
    int32_t acc = 0;
    acc += multiply_36(input[0][out_h + 0][out_w + 0][0]);
    acc += multiply_n6(input[0][out_h + 0][out_w + 1][0]);
    acc += multiply_85(input[0][out_h + 0][out_w + 2][0]);
    acc += multiply_n27(input[0][out_h + 1][out_w + 0][0]);
    acc += multiply_34(input[0][out_h + 1][out_w + 1][0]);
    acc += multiply_127(input[0][out_h + 1][out_w + 2][0]);
    acc += multiply_n13(input[0][out_h + 2][out_w + 0][0]);
    acc += multiply_72(input[0][out_h + 2][out_w + 1][0]);
    acc += multiply_114(input[0][out_h + 2][out_w + 2][0]);
    return acc;
}