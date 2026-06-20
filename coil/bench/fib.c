#include <stdio.h>
static long fib(long n){ return n < 2 ? n : fib(n-1) + fib(n-2); }
int main(void){ printf("%ld\n", fib(40)); return 0; }
