#include <stdio.h>
static long tak(long x,long y,long z){
  if (!(y < x)) return z;
  return tak(tak(x-1,y,z), tak(y-1,z,x), tak(z-1,x,y));
}
int main(void){ printf("%ld\n", tak(24,16,8)); return 0; }
