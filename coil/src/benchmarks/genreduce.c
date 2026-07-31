#include <stdio.h>
int main(void){
  long a[1024];
  for (long i = 0; i < 1024; i++) a[i] = i * 2654435761L;
  long acc = 0;
  for (long i = 0; i < 200000000L; i++) acc += a[i & 1023];
  printf("%ld\n", acc);
  return 0;
}
