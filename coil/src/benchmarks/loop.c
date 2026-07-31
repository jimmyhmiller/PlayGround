#include <stdio.h>
int main(void){
  unsigned long acc = 0, n = 300000000UL;
  for (unsigned long i = 0; i < n; i++) acc = acc * 1000003UL + i;
  printf("%lu\n", acc);
  return 0;
}
