#include <stdio.h>
int main(void) {
    int c = getchar();
    unsigned long b = (c < 0) ? 0UL : (unsigned long)c + 1UL;
    unsigned long count = b * 1000000UL;
    unsigned long acc = 0;
    for (unsigned long k = 0; k < count; k++)
        acc ^= 7UL*k*k*k + 5UL*k*k + 3UL*k + 11UL;
    printf("%lu\n", acc);
    return 0;
}
