#include <stdio.h>
#include <stdlib.h>
int main(void){
  unsigned long n = 20000000UL;
  unsigned long *a = malloc(n * sizeof(unsigned long));
  for (unsigned long i = 0; i < n; i++) a[i] = i * 2654435761UL;
  unsigned long s = 0;
  for (unsigned long i = 0; i < n; i++) s += a[i];
  printf("%lu\n", s);
  return 0;
}
