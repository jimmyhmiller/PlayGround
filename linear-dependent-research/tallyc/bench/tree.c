// The C twin of examples/tree.tal — same workload: build a real binary tree of
// depth d (each node a DISTINCT malloc'd cell, left/right labels 2*label and
// 2*label+1, so no sharing), then traverse it summing every label. This is genuine
// allocation + traversal: 2^d - 1 mallocs, kept live across the build, then walked.
//
//   cc -O2 bench/tree.c -o tree_c && ./tree_c

#include <stdio.h>
#include <stdlib.h>

typedef struct node { struct node *l; long x; struct node *r; } node;

static node *build(long d, long label) {
    if (d == 0) return NULL;                       // Leaf
    node *n = (node *)malloc(sizeof(node));        // a distinct node per call
    n->l = build(d - 1, label + label);            // left:  2*label
    n->x = label;
    n->r = build(d - 1, label + label + 1);        // right: 2*label + 1
    return n;
}

static long sum(node *t) {
    if (!t) return 0;                              // Leaf
    return sum(t->l) + t->x + sum(t->r);
}

int main(int argc, char **argv) {
    long d = (argc > 1) ? atol(argv[1]) : 22;
    printf("%lld\n", (long long)sum(build(d, 1)));
    return 0;
}
