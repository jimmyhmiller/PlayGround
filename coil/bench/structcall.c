#include <stdio.h>
typedef struct { long x, y, z; } V3;
static long dot(V3 a, V3 b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
int main(void){
  V3 v = {0, 2, 3}; long acc = 0;
  for (long i = 0; i < 100000000L; i++){ v.x = i; acc += dot(v, v); }
  printf("%ld\n", acc);
  return 0;
}
