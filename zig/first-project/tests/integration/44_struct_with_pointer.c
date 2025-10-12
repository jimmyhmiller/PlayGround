#include <stdio.h>
#include <stdlib.h>

typedef struct Node Node;
struct Node {
    long long value;
    Node* next;
};


typedef struct {
    Node n1;
    Node n2;
    Node* ptr1;
    Node* ptr2;
    Node* next_ptr;
    long long val;
} Namespace_user;

Namespace_user g_user;


void init_namespace_user(Namespace_user* ns) {
}

int main() {
    init_namespace_user(&g_user);
    g_user.n1 = (Node){10, NULL};
    g_user.n2 = (Node){20, NULL};
    g_user.ptr1 = ({ Node* __tmp_ptr = malloc(sizeof(Node)); *__tmp_ptr = g_user.n1; __tmp_ptr; });
    g_user.ptr2 = ({ Node* __tmp_ptr = malloc(sizeof(Node)); *__tmp_ptr = g_user.n2; __tmp_ptr; });
    g_user.ptr2->next = g_user.ptr1;
    g_user.next_ptr = g_user.ptr2->next;
    g_user.val = g_user.next_ptr->value;
    free(g_user.ptr1);
    free(g_user.ptr2);
    printf(((const char*)"%lld\n"), g_user.val);
    return 0;
}
