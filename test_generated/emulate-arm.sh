	qemu-arm \
	-cpu cortex-m0 \
    -g 3333 \
	-one-insn-per-tb \
	-plugin /home/ale19/Programs/qemu-8.2.2/build/tests/plugin/libinsn.so \
	-plugin /home/ale19/Programs/qemu-8.2.2/build/contrib/plugins/libexeclog.so \
	-d plugin \
    -D dump-$1.txt \
	$1