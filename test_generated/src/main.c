
#ifdef DEBUG
#include "model.h"
#else
#include "model_opt.h"
#endif


#define BUF_SIZE(buf) sizeof(buf) / sizeof(buf[0])
// Macro to define a generic print function for arrays
#define PRINT_ARRAY(arr, size, formatSpecifier) \
    do {                                        \
        printf("{ ");                           \
        for (int i = 0; i < size; i++) {        \
            printf(formatSpecifier, arr[i]);    \
            if (i != size - 1) {                \
                printf(", ");                   \
            }                                   \
        }                                       \
        printf(" }\n");                         \
    } while (0)
// Macro to define a generic print function for matrixes
#define PRINT_MATRIX(arr, row, col, formatSpecifier) \
    do {                                             \
        printf("{ ");                                \
        for (int i = 0; i < row; i++) {              \
            for (int j = 0; j < col; j++) {          \
                printf(formatSpecifier, arr[i][j]);  \
                if (i != col - 1) {                  \
                    printf(", ");                    \
                }                                    \
            }                                        \
        }                                            \
        printf(" }\n");                              \
    } while (0)

    int8_t feature_maps[1][26][26][2]
    = { 0 };
int8_t maxpooled[338] = { 0 };
int8_t result[10];

int main()
{
    byte_t image = {
        {
#include "../image_8.dat"
        }
    };
    quantize(&image);
    // PRINT_ARRAY(image.i8, BUF_SIZE(image.i8), "%3d");
    conv2d((int8_t(*)[28][28][1])image.i8, feature_maps);

    for (size_t cout = 0; cout < 2; cout++) {
        printf("{");
        for (size_t i = 0; i < 26; i++) {
            for (size_t j = 0; j < 26; j++) {
                if (j + 1 == 26) {
                    printf("%4d\n", feature_maps[0][i][j][cout]);
                    continue;
                }
                printf("%4d ", feature_maps[0][i][j][cout]);
            }
        }
        printf("}\n");
    }

    max_pool2d(feature_maps, (int8_t(*)[13][13][2])maxpooled);
    fully_connected((int8_t(*)[338])maxpooled, result);
    PRINT_ARRAY(result, BUF_SIZE(result), "%3d");
    return 0;
}