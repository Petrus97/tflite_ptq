#ifndef MACROS_H
#define MACROS_H

#define BUF_SIZE(buf) sizeof(buf) / sizeof(buf[0])

#if defined(__x86_64) || defined(__x86_64__)
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

#else
#define PRINT_ARRAY(arr, size, formatSpecifier) ((void)0)
#define PRINT_MATRIX(arr, row, col, formatSpecifier) ((void)0)
#endif

#endif