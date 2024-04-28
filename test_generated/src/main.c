
#ifdef DEBUG
#include "model.h"
#else
#include "model_opt.h"
#endif

#if defined(__AVR)
#include <avr/io.h>
#elif defined(__MSP430__)
#else
#include <stdio.h>
#endif

#include "macros.h"

static int argmax(const int8_t* arr, const size_t size)
{
    int max = 0;
    int8_t max_val = 0;
    for (int i = 1; i < size; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
            max = i;
        }
    }
    return max;
}

int8_t feature_maps_0[1][26][26][4] = { 0 };
int8_t maxpooled_0[1][13][13][4] = { 0 };
int8_t feature_maps_1[1][12][12][2] = { 0 };
int8_t maxpooled[72] = { 0 };
int8_t result[10] = { 0 };

int main()
{
    byte_t image = {
        {
#include "../image_8.dat"
        }
    };
    quantize(&image);
    // PRINT_ARRAY(image.i8, BUF_SIZE(image.i8), "%3d");
    conv2d_0((int8_t(*)[28][28][1])image.i8, feature_maps_0);
    max_pool2d_0(feature_maps_0, maxpooled_0);
    conv2d_1(maxpooled_0, feature_maps_1);
    max_pool2d_1(feature_maps_1, (int8_t(*)[6][6][2])maxpooled);
    fully_connected((int8_t(*)[72])maxpooled, result);
    PRINT_ARRAY(result, 10, "%3d");
    int inference = argmax(result, 10);
    // for (size_t cout = 0; cout < 2; cout++) {
    //     printf("{");
    //     for (size_t i = 0; i < 26; i++) {
    //         for (size_t j = 0; j < 26; j++) {
    //             if (j + 1 == 26) {
    //                 printf("%4d\n", feature_maps[0][i][j][cout]);
    //                 continue;
    //             }
    //             printf("%4d ", feature_maps[0][i][j][cout]);
    //         }
    //     }
    //     printf("}\n");
    // }

    // max_pool2d(feature_maps, (int8_t(*)[13][13][2])maxpooled);
    // fully_connected((int8_t(*)[338])maxpooled, result);
    // PRINT_ARRAY(result, BUF_SIZE(result), "%3d");

    return inference;
}