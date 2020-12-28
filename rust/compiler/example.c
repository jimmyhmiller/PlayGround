#include <stdio.h>

int fib(int64_t n) {
    if (n == 0 ){
        return 0;
    } else if (n == 1) {
        return 1;
    } else {
        return fib(n-1) + fib(n-2);
    }
}

int main() {
    int x = fib(40);
    printf("%d\n", x);
    return 0;
}


