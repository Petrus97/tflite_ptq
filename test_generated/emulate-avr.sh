qemu-system-avr \
    -machine mega2560 \
    -bios $1 \
    -gdb tcp::3333 \
    -icount 0 \
    -nographic \
    -plugin /home/ale19/Programs/qemu-8.2.2/build/tests/plugin/libinsn.so \
    -d plugin \
    -S
#    -S
    # -serial tcp::5678,server=on,wait=off \
    # -monitor stdio \
    # -S
#     -serial tcp::5678,server=on,wait=off \
    # -monitor stdio \
    # -D $1.txt \