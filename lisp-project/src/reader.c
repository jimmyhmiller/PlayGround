#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "stdio.h"
#include "stdlib.h"
typedef struct {
    uint8_t* data;
    int32_t count;
    int32_t capacity;
} Vector;

typedef enum {
    ValueTag_Nil,
    ValueTag_Number,
    ValueTag_Symbol,
    ValueTag_String,
    ValueTag_List,
    ValueTag_Vector,
} ValueTag;

typedef struct {
    ValueTag tag;
    int64_t num_val;
    uint8_t* str_val;
    uint8_t* cons_val;
    uint8_t* vec_val;
} Value;

typedef struct {
    uint8_t* car;
    uint8_t* cdr;
} Cons;


typedef struct {
    Value (*make_nil)();
    Value (*make_number)(int64_t);
    Value (*make_symbol)(uint8_t*);
    Value (*make_string)(uint8_t*);
    Value* (*make_vector_with_capacity)(int32_t, int32_t);
    int32_t (*vector_set)(Value*, int32_t, Value*);
    Value* (*make_cons)(Value*, Value*);
    Value* (*car)(Value*);
    Value* (*cdr)(Value*);
    int32_t (*print_list_contents)(Value*);
    int32_t (*print_vector_contents)(Value*, int32_t);
    int32_t (*print_value_ptr)(Value*);
    int32_t (*print_value)(Value);
    int32_t (*main_fn)();
} Namespace_user;

Namespace_user g_user;

static Value make_nil();
static Value make_number(int64_t);
static Value make_symbol(uint8_t*);
static Value make_string(uint8_t*);
static Value* make_vector_with_capacity(int32_t, int32_t);
static int32_t vector_set(Value*, int32_t, Value*);
static Value* make_cons(Value*, Value*);
static Value* car(Value*);
static Value* cdr(Value*);
static int32_t print_list_contents(Value*);
static int32_t print_vector_contents(Value*, int32_t);
static int32_t print_value_ptr(Value*);
static int32_t print_value(Value);
static int32_t main_fn();

void init_namespace_user(Namespace_user* ns) {
    ns->make_nil = &make_nil;
    ns->make_number = &make_number;
    ns->make_symbol = &make_symbol;
    ns->make_string = &make_string;
    ns->make_vector_with_capacity = &make_vector_with_capacity;
    ns->vector_set = &vector_set;
    ns->make_cons = &make_cons;
    ns->car = &car;
    ns->cdr = &cdr;
    ns->print_list_contents = &print_list_contents;
    ns->print_vector_contents = &print_vector_contents;
    ns->print_value_ptr = &print_value_ptr;
    ns->print_value = &print_value;
    ns->main_fn = &main_fn;
}

static Value make_nil() {
    return (Value){ValueTag_Nil, 0, NULL, NULL, NULL};
}
static Value make_number(int64_t n) {
    return (Value){ValueTag_Number, n, NULL, NULL, NULL};
}
static Value make_symbol(uint8_t* s) {
    return (Value){ValueTag_Symbol, 0, s, NULL, NULL};
}
static Value make_string(uint8_t* s) {
    return (Value){ValueTag_String, 0, s, NULL, NULL};
}
static Value* make_vector_with_capacity(int32_t count, int32_t capacity) {
    return ({ Vector* vec = (Vector*)((Vector*)malloc(16)); ({ uint8_t* data = (uint8_t*)malloc((capacity * 8)); vec->data = data; vec->count = count; vec->capacity = capacity; ({ Value* val = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; }); val->tag = ValueTag_Vector; val->vec_val = ((uint8_t*)vec); val; }); }); });
}
static int32_t vector_set(Value* vec_val, int32_t index, Value* elem) {
    return ({ uint8_t* vec_ptr = (uint8_t*)vec_val->vec_val; ({ Vector* vec = (Vector*)((Vector*)vec_ptr); ({ uint8_t* data = (uint8_t*)vec->data; ({ uint8_t* elem_loc = (uint8_t*)((uint8_t*)(((long long)data) + (index * 8))); ({ Value** elem_ptr = (Value**)((Value**)elem_loc); (*elem_ptr = elem); 0; }); }); }); }); });
}
static Value* make_cons(Value* car_val, Value* cdr_val) {
    return ({ Cons* cons_cell = (Cons*)((Cons*)malloc(16)); cons_cell->car = ((uint8_t*)car_val); cons_cell->cdr = ((uint8_t*)cdr_val); ({ Value* val = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; }); val->tag = ValueTag_List; val->cons_val = ((uint8_t*)cons_cell); val; }); });
}
static Value* car(Value* v) {
    return ({ uint8_t* cons_ptr = (uint8_t*)v->cons_val; ({ Cons* cons_cell = (Cons*)((Cons*)cons_ptr); ((Value*)cons_cell->car); }); });
}
static Value* cdr(Value* v) {
    return ({ uint8_t* cons_ptr = (uint8_t*)v->cons_val; ({ Cons* cons_cell = (Cons*)((Cons*)cons_ptr); ((Value*)cons_cell->cdr); }); });
}
static int32_t print_list_contents(Value* v) {
    return ({ ValueTag tag = v->tag; ((tag == ValueTag_Nil) ? printf(")") : ((tag == ValueTag_List) ? ({ int32_t _1 = g_user.print_value_ptr(g_user.car(v)); ({ Value* next = (Value*)g_user.cdr(v); ({ ValueTag next_tag = next->tag; ((next_tag == ValueTag_Nil) ? printf(")") : ({ int32_t _2 = printf(" "); g_user.print_list_contents(next); })); }); }); }) : ({ int32_t _1 = printf(". "); ({ int32_t _2 = g_user.print_value_ptr(v); printf(")"); }); }))); });
}
static int32_t print_vector_contents(Value* vec_val, int32_t index) {
    return ({ uint8_t* vec_ptr = (uint8_t*)vec_val->vec_val; ({ Vector* vec = (Vector*)((Vector*)vec_ptr); ({ int32_t count = vec->count; ((index >= count) ? printf("]") : ({ uint8_t* data = (uint8_t*)vec->data; ({ uint8_t* elem_loc = (uint8_t*)((uint8_t*)(((long long)data) + (index * 8))); ({ Value** elem_ptr_ptr = (Value**)((Value**)elem_loc); ({ Value* elem = (Value*)(*elem_ptr_ptr); g_user.print_value_ptr(elem); (((index + 1) < count) ? ({ int32_t _1 = printf(" "); g_user.print_vector_contents(vec_val, (index + 1)); }) : printf("]")); }); }); }); })); }); }); });
}
static int32_t print_value_ptr(Value* v) {
    return ({ ValueTag tag = v->tag; ((tag == ValueTag_Nil) ? printf("nil") : ((tag == ValueTag_Number) ? printf("%lld", v->num_val) : ((tag == ValueTag_Symbol) ? printf("%s", v->str_val) : ((tag == ValueTag_String) ? printf("\"%s\"", v->str_val) : ((tag == ValueTag_List) ? ({ int32_t _1 = printf("("); g_user.print_list_contents(v); }) : ((tag == ValueTag_Vector) ? ({ int32_t _1 = printf("["); g_user.print_vector_contents(v, 0); }) : printf("<unknown>"))))))); });
}
static int32_t print_value(Value v) {
    return ((v.tag == ValueTag_Nil) ? printf("nil\n") : ((v.tag == ValueTag_Number) ? printf("%lld\n", v.num_val) : ((v.tag == ValueTag_Symbol) ? printf("%s\n", v.str_val) : ((v.tag == ValueTag_String) ? printf("\"%s\"\n", v.str_val) : printf("<unknown>\n")))));
}
static int32_t main_fn() {
    printf("Testing basic value types:\n");
    ({ Value v1 = g_user.make_nil(); printf("nil value: "); g_user.print_value(v1); });
    ({ Value v2 = g_user.make_number(42); printf("number value: "); g_user.print_value(v2); });
    ({ Value v3 = g_user.make_number(-100); printf("negative number: "); g_user.print_value(v3); });
    ({ Value v4 = g_user.make_symbol("foo"); printf("symbol: "); g_user.print_value(v4); });
    ({ Value v5 = g_user.make_string("hello world"); printf("string: "); g_user.print_value(v5); });
    printf("\nTesting lists:\n");
    ({ Value* nil_val = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; }); ({ Value* num3 = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_number(3); __tmp_ptr; }); ({ Value* num2 = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_number(2); __tmp_ptr; }); ({ Value* num1 = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_number(1); __tmp_ptr; }); ({ Value* list3 = (Value*)g_user.make_cons(num3, nil_val); ({ Value* list2 = (Value*)g_user.make_cons(num2, list3); ({ Value* list1 = (Value*)g_user.make_cons(num1, list2); printf("list (1 2 3): "); g_user.print_value_ptr(list1); printf("\n"); }); }); }); }); }); }); });
    ({ Value* nil_val = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; }); ({ Value* bar = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_symbol("bar"); __tmp_ptr; }); ({ Value* foo = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_symbol("foo"); __tmp_ptr; }); ({ Value* list2 = (Value*)g_user.make_cons(bar, nil_val); ({ Value* list1 = (Value*)g_user.make_cons(foo, list2); printf("list (foo bar): "); g_user.print_value_ptr(list1); printf("\n"); }); }); }); }); });
    printf("\nTesting vectors:\n");
    ({ Value* vec = (Value*)g_user.make_vector_with_capacity(3, 3); ({ Value* num1 = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_number(1); __tmp_ptr; }); ({ Value* num2 = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_number(2); __tmp_ptr; }); ({ Value* num3 = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_number(3); __tmp_ptr; }); g_user.vector_set(vec, 0, num1); g_user.vector_set(vec, 1, num2); g_user.vector_set(vec, 2, num3); printf("vector [1 2 3]: "); g_user.print_value_ptr(vec); printf("\n"); }); }); }); });
    ({ Value* vec = (Value*)g_user.make_vector_with_capacity(2, 2); ({ Value* foo = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_symbol(":foo"); __tmp_ptr; }); ({ Value* bar = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_symbol(":bar"); __tmp_ptr; }); g_user.vector_set(vec, 0, foo); g_user.vector_set(vec, 1, bar); printf("vector [:foo :bar]: "); g_user.print_value_ptr(vec); printf("\n"); }); }); });
    return 0;
}
int main() {
    init_namespace_user(&g_user);
    g_user.main_fn();
    return 0;
}
