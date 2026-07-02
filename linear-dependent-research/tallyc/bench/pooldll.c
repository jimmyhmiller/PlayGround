// The hand-written C twin of examples/pooldll.tal: a pool-backed circular
// sentinel DLL (freelist + 256-cell blocks, same allocator design), running
// 1,000,000 insert + O(1)-remove-by-cursor transactions. Build with -O2.
#include <stdio.h>
#include <stdlib.h>

typedef struct node { struct node *prev, *next; long long elem; } node;
typedef struct blk { struct blk *next; node cells[256]; } blk;
typedef struct pool { node *freelist; blk *blocks; node *cur, *end; } pool;

static node *palloc_(pool *P) {
    if (P->freelist) { node *n = P->freelist; P->freelist = *(node **)n; return n; }
    if (P->cur < P->end) return P->cur++;
    blk *b = malloc(sizeof(blk));
    b->next = P->blocks; P->blocks = b;
    P->cur = b->cells + 1; P->end = b->cells + 256;
    return b->cells;
}
static void pfree_(pool *P, node *n) { *(node **)n = P->freelist; P->freelist = n; }

int main(void) {
    pool P = {0, 0, 0, 0};
    node *s = palloc_(&P);
    s->prev = s; s->next = s; s->elem = 0;
    long long acc = 0;
    for (long long k = 1000000 - 1; k >= 0; k--) {
        // insert after sentinel
        node *n = palloc_(&P);
        n->prev = s; n->next = s->next; n->elem = k;
        s->next->prev = n; s->next = n;
        // remove by cursor
        n->prev->next = n->next; n->next->prev = n->prev;
        acc += n->elem;
        pfree_(&P, n);
    }
    for (blk *b = P.blocks; b;) { blk *nb = b->next; free(b); b = nb; }
    printf("%lld\n", acc);
    return 0;
}
