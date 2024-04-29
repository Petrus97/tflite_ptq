	qemu-riscv32 \
	-cpu rv32 \
    -g 3333 \
	-one-insn-per-tb \
	-plugin /home/ale19/Programs/qemu-8.2.2/build/tests/plugin/libinsn.so \
	-d plugin \
    -D dump-$1.txt \
	$1

# -plugin /home/ale19/Programs/qemu-8.2.2/build/contrib/plugins/libexeclog.so \