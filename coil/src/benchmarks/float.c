#include <stdio.h>
int main(void){
  double s = 0.0; long n = 200000000;
  for (long i = 1; i <= n; i++) s += 1.0 / (double)i;
  printf("%.12f\n", s);
  return 0;
}
