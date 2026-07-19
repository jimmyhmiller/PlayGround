#include <stdint.h>
#include <stdio.h>

typedef struct { uint8_t a, b, c, d; } S4;
typedef struct { int32_t a, b; } S8;
typedef struct { int64_t a, b; } S16;
typedef struct { double a, b, c, d; } S32;

typedef struct {
  int64_t (*f4)(S4);
  int64_t (*f8)(S8);
  int64_t (*f16)(S16);
  int64_t (*f32)(S32);
} Table;

static int failures = 0;

static void report(const char *label, int ok, const char *got,
                   const char *expected) {
  printf("  %-28s %-28s expected %-20s %s\n", label, got, expected,
         ok ? "ok" : "CORRUPT");
  if (!ok) {
    failures++;
  }
}

static int64_t got4(S4 s) {
  char buf[64];
  snprintf(buf, sizeof buf, "%d,%d,%d,%d", s.a, s.b, s.c, s.d);
  report("call-ptr  4 bytes (4x u8)", s.a == 1 && s.b == 2 && s.c == 3 && s.d == 4,
         buf, "1,2,3,4");
  return 0;
}

static int64_t got8(S8 s) {
  char buf[64];
  snprintf(buf, sizeof buf, "%d,%d", s.a, s.b);
  report("call-ptr  8 bytes (2x i32)", s.a == 11 && s.b == 22, buf, "11,22");
  return 0;
}

static int64_t got16(S16 s) {
  char buf[64];
  snprintf(buf, sizeof buf, "%lld,%lld", (long long)s.a, (long long)s.b);
  report("call-ptr 16 bytes (2x i64)", s.a == 111 && s.b == 222, buf, "111,222");
  return 0;
}

static int64_t got32(S32 s) {
  char buf[64];
  snprintf(buf, sizeof buf, "%.0f,%.0f,%.0f,%.0f", s.a, s.b, s.c, s.d);
  report("call-ptr 32 bytes (4x f64)",
         s.a == 1 && s.b == 2 && s.c == 3 && s.d == 4, buf, "1,2,3,4");
  return 0;
}

/* The control: same 4-byte struct, but reached by a direct extern call. */
int64_t sink_direct(S4 s) {
  char buf[64];
  snprintf(buf, sizeof buf, "%d,%d,%d,%d", s.a, s.b, s.c, s.d);
  report("direct    4 bytes (4x u8)", s.a == 1 && s.b == 2 && s.c == 3 && s.d == 4,
         buf, "1,2,3,4");
  return 0;
}

extern int64_t via_callptr(Table *table, S4 s4, S8 s8, S16 s16, S32 s32);
extern int64_t via_direct(S4 s4);

int main(void) {
  Table table = {got4, got8, got16, got32};
  S4 s4 = {1, 2, 3, 4};
  S8 s8 = {11, 22};
  S16 s16 = {111, 222};
  S32 s32 = {1, 2, 3, 4};

  printf("Coil -> C, struct passed by value:\n");
  via_callptr(&table, s4, s8, s16, s32);
  via_direct(s4);

  printf("\n%d of 5 corrupt\n", failures);
  return failures == 0 ? 0 : 1;
}
