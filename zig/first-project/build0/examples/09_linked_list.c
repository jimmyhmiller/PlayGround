#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "stdio.h"
#include "stdlib.h"
typedef struct Node Node;
struct Node {
    int32_t value;
    Node* next;
};


typedef struct {
    Node* (*make_node)(int32_t, Node*);
    Node* (*prepend)(int32_t, Node*);
    int32_t (*print_list)(Node*);
    int32_t (*list_length)(Node*);
    int32_t (*list_sum)(Node*);
    int32_t (*main_fn)();
} Namespace_user;

Namespace_user g_user;

static Node* make_node(int32_t, Node*);
static Node* prepend(int32_t, Node*);
static int32_t print_list(Node*);
static int32_t list_length(Node*);
static int32_t list_sum(Node*);
static int32_t main_fn();

void init_namespace_user(Namespace_user* ns) {
    ns->make_node = &make_node;
    ns->prepend = &prepend;
    ns->print_list = &print_list;
    ns->list_length = &list_length;
    ns->list_sum = &list_sum;
    ns->main_fn = &main_fn;
}

static Node* make_node(int32_t value, Node* next) {
    return ({ Node* node = (Node*)({ Node* __tmp_ptr = malloc(sizeof(Node)); *__tmp_ptr = (Node){value, NULL}; __tmp_ptr; }); node->value = value; node->next = next; node; });
}
static Node* prepend(int32_t value, Node* list) {
    return g_user.make_node(value, list);
}
static int32_t print_list(Node* list) {
    return ((list == NULL) ? printf("nil\n") : ({ int32_t _1 = printf("%d -> ", list->value); g_user.print_list(list->next); }));
}
static int32_t list_length(Node* list) {
    return ((list == NULL) ? 0 : (1 + g_user.list_length(list->next)));
}
static int32_t list_sum(Node* list) {
    return ((list == NULL) ? 0 : (list->value + g_user.list_sum(list->next)));
}
static int32_t main_fn() {
    return ({ Node* list = (Node*)NULL; list = g_user.prepend(5, list); list = g_user.prepend(4, list); list = g_user.prepend(3, list); list = g_user.prepend(2, list); list = g_user.prepend(1, list); printf("List: "); g_user.print_list(list); printf("Length: %d\n", g_user.list_length(list)); printf("Sum: %d\n", g_user.list_sum(list)); 0; });
}
int main() {
    init_namespace_user(&g_user);
    g_user.main_fn();
    return 0;
}
