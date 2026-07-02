// The hand-written C twin of examples/intrusive.tal: an intrusive singly-linked
// list (payload embedded next to the link), 4,000,000 nodes — build by
// prepending, sum + free in one consuming walk. Build with -O2.
#include <stdio.h>
#include <stdlib.h>

typedef struct node {
    long long x, y;
    struct node *next;
} node;

#define N 4000000ll

int main(void) {
    node *head = NULL;
    for (long long k = N; k > 0; k--) {
        node *n = malloc(sizeof(node));
        n->x = k - 1;
        n->y = 2;
        n->next = head;
        head = n;
    }
    long long acc = 0;
    while (head) {
        node *t = head->next;
        acc += head->x + head->y;
        free(head);
        head = t;
    }
    printf("%lld\n", acc);
    return 0;
}
