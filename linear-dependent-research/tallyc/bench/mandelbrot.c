// Hand-written twin of examples/mandelbrot.tal — same algorithm, raw doubles.
#include <stdio.h>
#include <unistd.h>

static unsigned long escape(double cx, double cy, unsigned long fuel) {
    double zx = 0.0, zy = 0.0;
    unsigned long count = 0;
    while (fuel > 0) {
        double zx2 = zx*zx, zy2 = zy*zy;
        if (!(zx2 + zy2 < 4.0)) return count;
        double nzx = zx2 - zy2 + cx;
        double nzy = 2.0*zx*zy + cy;
        zx = nzx; zy = nzy;
        count++; fuel--;
    }
    return count;
}

int main(void) {
    unsigned char d = 0; ssize_t n = read(0, &d, 1); (void)n;
    unsigned long mi = (unsigned long)d + 3000;
    unsigned long acc = 0;
    for (unsigned long r = 0; r < 1200; r++) {
        double cy = -1.5 + (double)r * 0.0025;
        for (unsigned long col = 0; col < 1000; col++) {
            double cx = -2.5 + (double)col * 0.003;
            acc += escape(cx, cy, mi);
        }
    }
    printf("%lu\n", acc);
    return 0;
}
