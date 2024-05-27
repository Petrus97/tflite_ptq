ENTRY(_start)
MEMORY
{
    ram : ORIGIN = 0x00010000, LENGTH = 0x2000 /* 8KB RAM */
    rom : ORIGIN = 0x00012000, LENGTH = 0xBB80 /* 48KB ROM */
}
SECTIONS
{
    .text : { *(.text*) } > rom
    .rodata : { *(.rodata*) } > rom
    .bss : { *(.bss*) } > ram
    .data : { *(.data*) } > ram
}