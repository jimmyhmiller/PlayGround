#include <stdio.h>

void debugger_info() {
    // This is where the debugger would collect info
    printf("Debug info collection point\n");
}

int main(int argc, char **argv) {
    printf("Hello from test program!\n");
    debugger_info();
    printf("Goodbye from test program!\n");
    return 0;
}