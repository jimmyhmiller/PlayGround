#include <stdio.h>

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

double putchard(double X) {
  fputc((char)X, stderr);
  return 0;
}

double printd(double X) {
  fprintf(stderr, "%f\n", X);
  return 0;
}
