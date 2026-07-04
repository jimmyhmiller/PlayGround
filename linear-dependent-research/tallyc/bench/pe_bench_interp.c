// A tree-walking interpreter in C, over the SAME AST as the tally interpreter:
// dispatch on the node tag + recurse, per node, per iteration. The direct twin
// of `eval` in examples/pe_bench.tal — this is what the tally interpreter (no
// PE) is doing, written in C.
#include <stdio.h>
#include <stdlib.h>
typedef enum { VAR, LIT, ADD, MUL } Tag;
typedef struct Expr { Tag tag; unsigned long lit; struct Expr *a, *b; } Expr;
static Expr* mk(Tag t, unsigned long lit, Expr* a, Expr* b) {
    Expr* e = malloc(sizeof(Expr));
    e->tag = t; e->lit = lit; e->a = a; e->b = b;
    return e;
}
static unsigned long eval(const Expr* e, unsigned long x) {
    switch (e->tag) {
        case VAR: return x;
        case LIT: return e->lit;
        case ADD: return eval(e->a, x) + eval(e->b, x);
        case MUL: return eval(e->a, x) * eval(e->b, x);
    }
    return 0;
}
int main(void) {
    int c = getchar();
    unsigned long b = (c < 0) ? 0UL : (unsigned long)c + 1UL;
    unsigned long count = b * 1000000UL;
    // 7*x^3 + 5*x^2 + 3*x + 11   (same tree as poly() in the tally program)
    Expr* v = mk(VAR,0,0,0);
    Expr* poly =
      mk(ADD,0, mk(MUL,0, mk(LIT,7,0,0), mk(MUL,0,v,mk(MUL,0,v,v))),
      mk(ADD,0, mk(MUL,0, mk(LIT,5,0,0), mk(MUL,0,v,v)),
      mk(ADD,0, mk(MUL,0, mk(LIT,3,0,0), v), mk(LIT,11,0,0))));
    unsigned long acc = 0;
    for (unsigned long k = 0; k < count; k++)
        acc ^= eval(poly, k);
    printf("%lu\n", acc);
    return 0;
}
