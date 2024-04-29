SHELL = /bin/sh

CC=riscv32-unknown-elf-gcc
CFLAGS=-Wall -Werror -Wpedantic -O2 -g
READELF=riscv32-unknown-elf-readelf
OBJDUMP=riscv32-unknown-elf-objdump
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
BIN = riscv.opt.elf
else
BIN = riscv.elf
endif

LDFLAGS += -Wl,-Map,$(BIN).map
LDFLAGS += -Wl,--gc-sections
LDFLAGS += -Wl,--start-group -lgcc -lc
LDFLAGS += -Wl,--end-group


CFLAGS += -DDEBUG=1 -Wno-unused-variable -save-temps=obj -fverbose-asm
CFLAGS += -I$(INC_DIR)


#
# Compile the binary
#
all: prebuild $(BIN) dump

$(BIN): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJS): $(OBJ_DIR)/%.o:$(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

dump:
	$(OBJDUMP) -D $(BIN) > $(OBJ_DIR)/$(BIN).s
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