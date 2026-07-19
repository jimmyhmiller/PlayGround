// Hand-written twin of examples/nbody_aos.tal — struct array, raw doubles.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>

typedef struct { double x, y, vx, vy; } Body;

int main(void) {
    unsigned char d = 0; ssize_t r = read(0, &d, 1); (void)r;
    unsigned long steps = (unsigned long)d + 8000;
    unsigned long n = 50000;
    Body *a = malloc(n * sizeof(Body));
    for (unsigned long i = 0; i < n; i++) { a[i].x = 1.5; a[i].y = 0.0; a[i].vx = 0.0; a[i].vy = 0.8; }
    for (unsigned long s = 0; s < steps; s++) {
        for (unsigned long i = 0; i < n; i++) {
            double x = a[i].x, y = a[i].y, vx = a[i].vx, vy = a[i].vy;
            double r2 = x*x + y*y + 0.01;
            double inv = 1.0 / (r2 * sqrt(r2));
            double nvx = vx + (-(x*inv)) * 0.01;
            double nvy = vy + (-(y*inv)) * 0.01;
            double nx = x + nvx*0.01;
            double ny = y + nvy*0.01;
            a[i].x = nx; a[i].y = ny; a[i].vx = nvx; a[i].vy = nvy;
        }
    }
    unsigned long acc = 0;
    for (unsigned long i = 0; i < n; i++) acc += (unsigned long)(a[i].x + 1000.0);
    free(a);
    printf("%lu\n", acc);
    return 0;
}
