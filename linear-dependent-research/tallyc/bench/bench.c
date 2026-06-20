// The C twin of examples/bench.tal — the SAME workload written by hand: fold N
// transactions on an intrusive circular doubly-linked list with O(1)
// remove-by-cursor (new sentinel, insert a node, unlink+free it, free the list),
// summing the value each transaction round-trips through the heap.
//
//   cc -O2 bench/bench.c -o bench_c && ./bench_c
//
// Compile at -O2 and compare to tally's `tally_dep_main`: because the workload is
// pure (no escaping allocation), BOTH the C compiler and tally+LLVM evaluate the
// whole thing and fold it to a single constant. The safe, linearity-checked,
// dependently-typed tally version and this raw-pointer C compile to the SAME
// machine code — the erased proofs cost exactly nothing.

#include <stdio.h>
#include <stdlib.h>

typedef struct node { struct node *next, *prev; long elem; } node;

static long once(long x) {
    node *s = (node *)malloc(sizeof(node));   // new: empty circular sentinel
    s->next = s; s->prev = s;
    node *n = (node *)malloc(sizeof(node));   // insert at tail
    node *prev = s->prev;
    n->elem = x; n->prev = prev; n->next = s;
    prev->next = n; s->prev = n;
    n->prev->next = n->next;                  // remove by cursor (O(1))
    n->next->prev = n->prev;
    long v = n->elem;
    free(n);
    free(s);                                  // free the empty list
    return v;
}

static long run(long n) {
    long acc = 0;
    for (long k = 0; k < n; k++) acc += once(k);
    return acc;
}

int main(void) {
    printf("%lld\n", (long long)run(1000000));
    return 0;
}
