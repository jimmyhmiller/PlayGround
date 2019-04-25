#include <stdio.h>
#include <sys/mman.h>

#define bool_f 0x2F
#define bool_t 0x6F
#define null_rep 0x3F
#define fx_mask 0x03
#define fx_tag 0x00
#define fx_shift 2
#define char_shift 8
#define char_tag 0x0F
#define char_mask 0x3F

typedef unsigned int ptr;


static void print_char (char x) {
  switch (x) {
  case '\t': { printf("#\\tab"); break; }
  case '\n': { printf("#\\newline"); break; }
  case '\r': { printf("#\\return"); break; }
  case ' ': { printf("#\\space"); break; }
  default: { printf("#\\%c", x); break; }
  }
}

static void print_ptr(ptr x) {
  if ((x & fx_mask) == fx_tag) {
    printf("%d", ((int)x) >> fx_shift);
  } else if (x == bool_f) {
    printf("#f");
  } else if (x == bool_t) {
    printf("#t");
  } else if (x == null_rep) {
    printf("()");
  } else if ((x & char_mask) == char_tag) {
    char c = (char) (x >> char_shift);
    print_char(c);
  } else {
    printf("#<unknown 0x%08x>", x);
  }
  printf("\n");
}

static char* allocate_protected_space(int size) {
  int page = getpagesize();
  int status;
  int aligned_size = ((size + page - 1) / page) * page;
  char* p = mmap(0, aligned_size + 2 * page,
                 PROT_READ | PROT_WRITE,
                 MAP_ANONYMOUS | MAP_PRIVATE,
                 0, 0);
  if (p == MAP_FAILED) { }
  status = mprotect(p, page, PROT_NONE);
  if (status != 0) { perror("mprotect"); exit(status); }
  status = mprotect(p + page + aligned_size, page, PROT_NONE);
  if (status != 0) { perror("mprotect"); exit(status); }
  return (p + page);
 }

static void deallocate_protected_space(char* p, int size) {
  int page = getpagesize();
  int status;
  int aligned_size = ((size + page - 1) / page);
  status = munmap(p - page, aligned_size + 2 * page);
  if (status != 0) { perror("mprotect"); exit(status); }
}


int main(int argc, char** argv) {
  int stack_size = (16 * 4096); // 16k cells
  char* stack_top = allocate_protected_space(stack_size);
  char* stack_base = stack_top + stack_size;
  print_ptr(scheme_entry(stack_base));
  deallocate_protected_space(stack_top, stack_size);
  return 0;
}
