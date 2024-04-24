SHELL = /bin/sh

CC=/home/ale19/ti/msp430-gcc/bin/msp430-elf-gcc-9.3.1
CFLAGS=-Wall -Werror -Wpedantic -O3
READELF=/home/ale19/ti/msp430-gcc/bin/msp430-elf-readelf
LDLIBS=
LDFLAGS=

#
# Directories
#
SRC_DIR=./src
INC_DIR=./include
OBJ_DIR=./objs

#
# Automatic variables for .c and .o files
#
SRCS=$(wildcard $(SRC_DIR)/*.c)
OBJS=$(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))

ifdef OPT
SRCS := $(filter-out $(SRC_DIR)/model.c, $(SRCS))
else
SRCS := $(filter-out $(SRC_DIR)/model_opt.c, $(SRCS))
endif


ifdef OPT
BIN = msp430.opt.elf
else
BIN = msp430.elf
endif

MCU=msp430f5529
MCU_SERIES=f5series
MSPGCC_INC=/home/ale19/ti/msp430-gcc/msp430-elf/include
MSP430_INC=/home/ale19/ti/ccs1260/ccs/ccs_base/msp430/include_gcc

LDFLAGS += -Wl,-Map,$(BIN).map
LDFLAGS += -Wl,--gc-sections -L$(MSP430_INC)
LDFLAGS += -Wl,--start-group -lgcc -lc
LDFLAGS += -Wl,--end-group


CFLAGS += -DDEBUG=1 -DMSP430=1 -Wno-unused-variable -save-temps=obj -fverbose-asm
CFLAGS += -mmcu=$(MCU) -mhwmult=$(MCU_SERIES) -mlarge -mcode-region=none -mdata-region=lower
CFLAGS += -I$(MSPGCC_INC) -I$(INC_DIR) -I$(MSP430_INC)


#
# Compile the binary
#
all: prebuild $(BIN)

$(BIN): $(OBJS)
# -MMD -MP -MF"multiply.d_raw" -MT"multiply.o"
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJS): $(OBJ_DIR)/%.o:$(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@


#
# Add the operations to do before start the compilation
#
prebuild:
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(BIN)* $(OBJ_DIR)/* *.i *.s *.d *.map

# all:
# 	$(TICC) -O3  operations_0.c multiply.c -Wl,--gc-sections -L"/home/ale19/ti/ccs1260/ccs/ccs_base/msp430/include_gcc" -mmcu=msp430f5529

# do-ccs:
# # build multiply.o
# #	$(CC) -c -mmcu=msp430f5529 -mhwmult=f5series -I"/home/ale19/ti/ccs1260/ccs/ccs_base/msp430/include_gcc" -I"./" -I"/home/ale19/ti/msp430-gcc/msp430-elf/include" -O0 -g -gdwarf-3 -gstrict-dwarf -Wall -mlarge -mcode-region=none -mdata-region=lower -MMD -MP -MF"multiply.d_raw" -MT"multiply.o"   -o"multiply.o" "multiply.c" -save-temps
# 	$(TICC) -c -mmcu=msp430f5529 -mhwmult=f5series -I"/home/ale19/ti/ccs1260/ccs/ccs_base/msp430/include_gcc" -I"./" -I"/home/ale19/ti/msp430-gcc/msp430-elf/include" -O3 -Wall -mlarge -mcode-region=none -mdata-region=lower -MMD -MP -MF"multiply.d_raw" -MT"multiply.o"   -o"multiply.o" "multiply.c" -save-temps

# #build operations_0.o
# #	$(CC) -c -mmcu=msp430f5529 -mhwmult=f5series -I"/home/ale19/ti/ccs1260/ccs/ccs_base/msp430/include_gcc" -I"./" -I"/home/ale19/ti/msp430-gcc/msp430-elf/include" -O0 -g -gdwarf-3 -gstrict-dwarf -Wall -mlarge -mcode-region=none -mdata-region=lower -MMD -MP -MF"operations_0.d_raw" -MT"operations_0.o"   -o"operations_0.o" "operations_0.c" -save-temps -fverbose-asm
# 	$(TICC) -c -mmcu=msp430f5529 -mhwmult=f5series -I"/home/ale19/ti/ccs1260/ccs/ccs_base/msp430/include_gcc" -I"./" -I"/home/ale19/ti/msp430-gcc/msp430-elf/include" -O3 -Wall -mlarge -mcode-region=none -mdata-region=lower -MMD -MP -MF"operations_0.d_raw" -MT"operations_0.o"   -o"operations_0.o" "operations_0.c" -save-temps 
# # link
# 	$(TICC) -mhwmult=f5series -O3 -Wall -mcode-region=none -mlarge -Wl,-Map,"second_try.map" -Wl,--gc-sections -L"/home/ale19/ti/ccs1260/ccs/ccs_base/msp430/include_gcc" -mmcu=msp430f5529 -o"second_try.out" "./operations_0.o" "./multiply.o" -T"/home/ale19/c/play-compiler/msp430-gcc-support-files/include/msp430f5529.ld"  -Wl,--start-group -lgcc -lc -Wl,--end-group  -save-temps 
# # show elf sections
# 	$(TIELF) -S second_try.out