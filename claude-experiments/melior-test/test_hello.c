#include <stdio.h>

// Declare our compiled function
extern void hello(void);

int main() {
    printf("Testing our MLIR-compiled hello function:\n");
    hello();
    printf("Hello function completed successfully!\n");
    return 0;
}
