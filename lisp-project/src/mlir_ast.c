#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "stdio.h"
#include "stdlib.h"
#include "string.h"
typedef enum {
    ValueTag_Nil,
    ValueTag_Number,
    ValueTag_Symbol,
    ValueTag_String,
    ValueTag_List,
    ValueTag_Vector,
    ValueTag_Keyword,
    ValueTag_Map,
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
    uint8_t* data;
    int32_t count;
    int32_t capacity;
} Vector;


typedef struct {
    Value (*make_nil)();
    Value (*make_symbol)(uint8_t*);
    Value (*make_string)(uint8_t*);
    Value* (*make_empty_vector)();
    Value* (*make_empty_map)();
    Value* (*make_cons)(Value*, Value*);
    Value* (*car)(Value*);
    Value* (*cdr)(Value*);
    int32_t (*is_symbol_op)(Value*);
    int32_t (*is_symbol_block)(Value*);
    int32_t (*is_op)(Value*);
    int32_t (*is_block)(Value*);
    Value* (*get_op_name)(Value*);
    Value* (*get_op_result_types)(Value*);
    Value* (*get_op_operands)(Value*);
    Value* (*get_op_attributes)(Value*);
    Value* (*get_op_regions)(Value*);
    Value* (*get_block_args)(Value*);
    Value* (*get_block_operations)(Value*);
    int32_t (*main_fn)();
} Namespace_user;

Namespace_user g_user;

static Value make_nil();
static Value make_symbol(uint8_t*);
static Value make_string(uint8_t*);
static Value* make_empty_vector();
static Value* make_empty_map();
static Value* make_cons(Value*, Value*);
static Value* car(Value*);
static Value* cdr(Value*);
static int32_t is_symbol_op(Value*);
static int32_t is_symbol_block(Value*);
static int32_t is_op(Value*);
static int32_t is_block(Value*);
static Value* get_op_name(Value*);
static Value* get_op_result_types(Value*);
static Value* get_op_operands(Value*);
static Value* get_op_attributes(Value*);
static Value* get_op_regions(Value*);
static Value* get_block_args(Value*);
static Value* get_block_operations(Value*);
static int32_t main_fn();

void init_namespace_user(Namespace_user* ns) {
    ns->make_nil = &make_nil;
    ns->make_symbol = &make_symbol;
    ns->make_string = &make_string;
    ns->make_empty_vector = &make_empty_vector;
    ns->make_empty_map = &make_empty_map;
    ns->make_cons = &make_cons;
    ns->car = &car;
    ns->cdr = &cdr;
    ns->is_symbol_op = &is_symbol_op;
    ns->is_symbol_block = &is_symbol_block;
    ns->is_op = &is_op;
    ns->is_block = &is_block;
    ns->get_op_name = &get_op_name;
    ns->get_op_result_types = &get_op_result_types;
    ns->get_op_operands = &get_op_operands;
    ns->get_op_attributes = &get_op_attributes;
    ns->get_op_regions = &get_op_regions;
    ns->get_block_args = &get_block_args;
    ns->get_block_operations = &get_block_operations;
    ns->main_fn = &main_fn;
}

static Value make_nil() {
    return (Value){ValueTag_Nil, 0, NULL, NULL, NULL};
}
static Value make_symbol(uint8_t* s) {
    return (Value){ValueTag_Symbol, 0, s, NULL, NULL};
}
static Value make_string(uint8_t* s) {
    return (Value){ValueTag_String, 0, s, NULL, NULL};
}
static Value* make_empty_vector() {
    return ({ Vector* vec = (Vector*)((Vector*)malloc(16)); uint8_t* data = (uint8_t*)malloc(8); vec->data = data; vec->count = 0; vec->capacity = 1; ({ Value* val = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; }); val->tag = ValueTag_Vector; val->vec_val = ((uint8_t*)vec); val; }); });
}
static Value* make_empty_map() {
    return ({ Vector* vec = (Vector*)((Vector*)malloc(16)); uint8_t* data = (uint8_t*)malloc(8); vec->data = data; vec->count = 0; vec->capacity = 1; ({ Value* val = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; }); val->tag = ValueTag_Map; val->vec_val = ((uint8_t*)vec); val; }); });
}
static Value* make_cons(Value* car_val, Value* cdr_val) {
    return ({ Cons* cons_cell = (Cons*)((Cons*)malloc(16)); cons_cell->car = ((uint8_t*)car_val); cons_cell->cdr = ((uint8_t*)cdr_val); ({ Value* val = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; }); val->tag = ValueTag_List; val->cons_val = ((uint8_t*)cons_cell); val; }); });
}
static Value* car(Value* v) {
    return ({ uint8_t* cons_ptr = (uint8_t*)v->cons_val; Cons* cons_cell = (Cons*)((Cons*)cons_ptr); ((Value*)cons_cell->car); });
}
static Value* cdr(Value* v) {
    return ({ uint8_t* cons_ptr = (uint8_t*)v->cons_val; Cons* cons_cell = (Cons*)((Cons*)cons_ptr); ((Value*)cons_cell->cdr); });
}
static int32_t is_symbol_op(Value* v) {
    return ({ ValueTag tag = v->tag; ((tag == ValueTag_Symbol) ? ({ uint8_t* str_val = (uint8_t*)v->str_val; int32_t cmp_result = strcmp(str_val, "op"); ((cmp_result == 0) ? 1 : 0); }) : 0); });
}
static int32_t is_symbol_block(Value* v) {
    return ({ ValueTag tag = v->tag; ((tag == ValueTag_Symbol) ? ({ uint8_t* str_val = (uint8_t*)v->str_val; int32_t cmp_result = strcmp(str_val, "block"); ((cmp_result == 0) ? 1 : 0); }) : 0); });
}
static int32_t is_op(Value* v) {
    return ({ ValueTag tag = v->tag; ((tag == ValueTag_List) ? g_user.is_symbol_op(g_user.car(v)) : 0); });
}
static int32_t is_block(Value* v) {
    return ({ ValueTag tag = v->tag; ((tag == ValueTag_List) ? g_user.is_symbol_block(g_user.car(v)) : 0); });
}
static Value* get_op_name(Value* op_form) {
    return ((g_user.is_op(op_form) == 1) ? ({ Value* rest = (Value*)g_user.cdr(op_form); ValueTag rest_tag = rest->tag; ((rest_tag == ValueTag_List) ? g_user.car(rest) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; })); }) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; }));
}
static Value* get_op_result_types(Value* op_form) {
    return ((g_user.is_op(op_form) == 1) ? ({ Value* rest = (Value*)g_user.cdr(op_form); ValueTag rest_tag = rest->tag; ((rest_tag == ValueTag_List) ? ({ Value* rest2 = (Value*)g_user.cdr(rest); ValueTag rest2_tag = rest2->tag; ((rest2_tag == ValueTag_List) ? g_user.car(rest2) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; })); }) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; })); }) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; }));
}
static Value* get_op_operands(Value* op_form) {
    return ((g_user.is_op(op_form) == 1) ? ({ Value* rest = (Value*)g_user.cdr(op_form); ValueTag rest_tag = rest->tag; ((rest_tag == ValueTag_List) ? ({ Value* rest2 = (Value*)g_user.cdr(rest); ValueTag rest2_tag = rest2->tag; ((rest2_tag == ValueTag_List) ? ({ Value* rest3 = (Value*)g_user.cdr(rest2); ValueTag rest3_tag = rest3->tag; ((rest3_tag == ValueTag_List) ? g_user.car(rest3) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; })); }) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; })); }) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; })); }) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; }));
}
static Value* get_op_attributes(Value* op_form) {
    return ((g_user.is_op(op_form) == 1) ? ({ Value* rest = (Value*)g_user.cdr(op_form); ValueTag rest_tag = rest->tag; ((rest_tag == ValueTag_List) ? ({ Value* rest2 = (Value*)g_user.cdr(rest); ValueTag rest2_tag = rest2->tag; ((rest2_tag == ValueTag_List) ? ({ Value* rest3 = (Value*)g_user.cdr(rest2); ValueTag rest3_tag = rest3->tag; ((rest3_tag == ValueTag_List) ? ({ Value* rest4 = (Value*)g_user.cdr(rest3); ValueTag rest4_tag = rest4->tag; ((rest4_tag == ValueTag_List) ? g_user.car(rest4) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; })); }) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; })); }) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; })); }) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; })); }) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; }));
}
static Value* get_op_regions(Value* op_form) {
    return ((g_user.is_op(op_form) == 1) ? ({ Value* rest = (Value*)g_user.cdr(op_form); ValueTag rest_tag = rest->tag; ((rest_tag == ValueTag_List) ? ({ Value* rest2 = (Value*)g_user.cdr(rest); ValueTag rest2_tag = rest2->tag; ((rest2_tag == ValueTag_List) ? ({ Value* rest3 = (Value*)g_user.cdr(rest2); ValueTag rest3_tag = rest3->tag; ((rest3_tag == ValueTag_List) ? ({ Value* rest4 = (Value*)g_user.cdr(rest3); ValueTag rest4_tag = rest4->tag; ((rest4_tag == ValueTag_List) ? ({ Value* rest5 = (Value*)g_user.cdr(rest4); ValueTag rest5_tag = rest5->tag; ((rest5_tag == ValueTag_List) ? g_user.car(rest5) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; })); }) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; })); }) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; })); }) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; })); }) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; })); }) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; }));
}
static Value* get_block_args(Value* block_form) {
    return ((g_user.is_block(block_form) == 1) ? ({ Value* rest = (Value*)g_user.cdr(block_form); ValueTag rest_tag = rest->tag; ((rest_tag == ValueTag_List) ? g_user.car(rest) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; })); }) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; }));
}
static Value* get_block_operations(Value* block_form) {
    return ((g_user.is_block(block_form) == 1) ? ({ Value* rest = (Value*)g_user.cdr(block_form); ValueTag rest_tag = rest->tag; ((rest_tag == ValueTag_List) ? ({ Value* rest2 = (Value*)g_user.cdr(rest); ValueTag rest2_tag = rest2->tag; ((rest2_tag == ValueTag_List) ? g_user.car(rest2) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; })); }) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; })); }) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; }));
}
static int32_t main_fn() {
    printf("Testing is-symbol-op:\n");
    ({ Value* op_sym = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_symbol("op"); __tmp_ptr; }); int32_t result1 = g_user.is_symbol_op(op_sym); printf("  symbol 'op': %d (expected 1)\n", result1); });
    ({ Value* block_sym = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_symbol("block"); __tmp_ptr; }); int32_t result2 = g_user.is_symbol_op(block_sym); printf("  symbol 'block': %d (expected 0)\n", result2); });
    ({ Value* nil_val = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; }); int32_t result3 = g_user.is_symbol_op(nil_val); printf("  nil: %d (expected 0)\n", result3); });
    printf("\nTesting is-symbol-block:\n");
    ({ Value* block_sym = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_symbol("block"); __tmp_ptr; }); int32_t result4 = g_user.is_symbol_block(block_sym); printf("  symbol 'block': %d (expected 1)\n", result4); });
    ({ Value* op_sym = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_symbol("op"); __tmp_ptr; }); int32_t result5 = g_user.is_symbol_block(op_sym); printf("  symbol 'op': %d (expected 0)\n", result5); });
    printf("\nTesting is-op predicate:\n");
    ({ Value* nil_val = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; }); Value* op_sym = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_symbol("op"); __tmp_ptr; }); Value* op_list = (Value*)g_user.make_cons(op_sym, nil_val); int32_t result6 = g_user.is_op(op_list); printf("  (op): %d (expected 1)\n", result6); });
    ({ Value* nil_val = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; }); Value* block_sym = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_symbol("block"); __tmp_ptr; }); Value* block_list = (Value*)g_user.make_cons(block_sym, nil_val); int32_t result7 = g_user.is_op(block_list); printf("  (block): %d (expected 0)\n", result7); });
    printf("\nTesting is-block predicate:\n");
    ({ Value* nil_val = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; }); Value* block_sym = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_symbol("block"); __tmp_ptr; }); Value* block_list = (Value*)g_user.make_cons(block_sym, nil_val); int32_t result8 = g_user.is_block(block_list); printf("  (block): %d (expected 1)\n", result8); });
    ({ Value* nil_val = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; }); Value* op_sym = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_symbol("op"); __tmp_ptr; }); Value* op_list = (Value*)g_user.make_cons(op_sym, nil_val); int32_t result9 = g_user.is_block(op_list); printf("  (op): %d (expected 0)\n", result9); });
    printf("\nTesting get-op-name:\n");
    ({ Value* nil_val = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; }); Value* name_str = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_string("func.func"); __tmp_ptr; }); Value* rest = (Value*)g_user.make_cons(name_str, nil_val); Value* op_sym = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_symbol("op"); __tmp_ptr; }); Value* op_list = (Value*)g_user.make_cons(op_sym, rest); Value* extracted_name = (Value*)g_user.get_op_name(op_list); ValueTag name_tag = extracted_name->tag; ((name_tag == ValueTag_String) ? ({ uint8_t* name_val = (uint8_t*)extracted_name->str_val; printf("  extracted name: \"%s\" (expected \"func.func\")\n", name_val); }) : printf("  ERROR: extracted name is not a string!\n")); });
    ({ Value* nil_val = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; }); Value* op_sym = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_symbol("op"); __tmp_ptr; }); Value* op_list = (Value*)g_user.make_cons(op_sym, nil_val); Value* extracted_name = (Value*)g_user.get_op_name(op_list); ValueTag name_tag = extracted_name->tag; ((name_tag == ValueTag_Nil) ? printf("  (op) with no name: nil (expected nil)\n") : printf("  ERROR: should have returned nil!\n")); });
    printf("\nTesting get-op-result-types:\n");
    ({ Value* nil_val = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; }); Value* empty_vec = (Value*)g_user.make_empty_vector(); Value* rest2 = (Value*)g_user.make_cons(empty_vec, nil_val); Value* name_str = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_string("name"); __tmp_ptr; }); Value* rest1 = (Value*)g_user.make_cons(name_str, rest2); Value* op_sym = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_symbol("op"); __tmp_ptr; }); Value* op_list = (Value*)g_user.make_cons(op_sym, rest1); Value* extracted_types = (Value*)g_user.get_op_result_types(op_list); ValueTag types_tag = extracted_types->tag; ((types_tag == ValueTag_Vector) ? printf("  extracted result-types: vector (expected vector)\n") : printf("  ERROR: extracted result-types is not a vector!\n")); });
    printf("\nTesting get-op-operands:\n");
    ({ Value* nil_val = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; }); Value* operands_vec = (Value*)g_user.make_empty_vector(); Value* rest3 = (Value*)g_user.make_cons(operands_vec, nil_val); Value* types_vec = (Value*)g_user.make_empty_vector(); Value* rest2 = (Value*)g_user.make_cons(types_vec, rest3); Value* name_str = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_string("name"); __tmp_ptr; }); Value* rest1 = (Value*)g_user.make_cons(name_str, rest2); Value* op_sym = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_symbol("op"); __tmp_ptr; }); Value* op_list = (Value*)g_user.make_cons(op_sym, rest1); Value* extracted_operands = (Value*)g_user.get_op_operands(op_list); ValueTag operands_tag = extracted_operands->tag; ((operands_tag == ValueTag_Vector) ? printf("  extracted operands: vector (expected vector)\n") : printf("  ERROR: extracted operands is not a vector!\n")); });
    printf("\nTesting complete op extraction:\n");
    ({ Value* nil_val = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; }); Value* regions = (Value*)g_user.make_empty_vector(); Value* rest5 = (Value*)g_user.make_cons(regions, nil_val); Value* attrs = (Value*)g_user.make_empty_map(); Value* rest4 = (Value*)g_user.make_cons(attrs, rest5); Value* operands = (Value*)g_user.make_empty_vector(); Value* rest3 = (Value*)g_user.make_cons(operands, rest4); Value* types = (Value*)g_user.make_empty_vector(); Value* rest2 = (Value*)g_user.make_cons(types, rest3); Value* name = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_string("arith.constant"); __tmp_ptr; }); Value* rest1 = (Value*)g_user.make_cons(name, rest2); Value* op_sym = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_symbol("op"); __tmp_ptr; }); Value* complete_op = (Value*)g_user.make_cons(op_sym, rest1); printf("  Testing extraction from complete op form:\n"); ({ Value* ext_name = (Value*)g_user.get_op_name(complete_op); ValueTag name_tag = ext_name->tag; ((name_tag == ValueTag_String) ? printf("    name: OK\n") : printf("    name: ERROR\n")); }); ({ Value* ext_types = (Value*)g_user.get_op_result_types(complete_op); ValueTag types_tag = ext_types->tag; ((types_tag == ValueTag_Vector) ? printf("    result-types: OK\n") : printf("    result-types: ERROR\n")); }); ({ Value* ext_operands = (Value*)g_user.get_op_operands(complete_op); ValueTag operands_tag = ext_operands->tag; ((operands_tag == ValueTag_Vector) ? printf("    operands: OK\n") : printf("    operands: ERROR\n")); }); ({ Value* ext_attrs = (Value*)g_user.get_op_attributes(complete_op); ValueTag attrs_tag = ext_attrs->tag; ((attrs_tag == ValueTag_Map) ? printf("    attributes: OK\n") : printf("    attributes: ERROR\n")); }); ({ Value* ext_regions = (Value*)g_user.get_op_regions(complete_op); ValueTag regions_tag = ext_regions->tag; ((regions_tag == ValueTag_Vector) ? printf("    regions: OK\n") : printf("    regions: ERROR\n")); }); });
    printf("\nTesting block extraction:\n");
    ({ Value* nil_val = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; }); Value* operations = (Value*)g_user.make_empty_vector(); Value* rest2 = (Value*)g_user.make_cons(operations, nil_val); Value* block_args = (Value*)g_user.make_empty_vector(); Value* rest1 = (Value*)g_user.make_cons(block_args, rest2); Value* block_sym = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_symbol("block"); __tmp_ptr; }); Value* complete_block = (Value*)g_user.make_cons(block_sym, rest1); printf("  Testing extraction from complete block form:\n"); ({ Value* ext_args = (Value*)g_user.get_block_args(complete_block); ValueTag args_tag = ext_args->tag; ((args_tag == ValueTag_Vector) ? printf("    block-args: OK\n") : printf("    block-args: ERROR\n")); }); ({ Value* ext_ops = (Value*)g_user.get_block_operations(complete_block); ValueTag ops_tag = ext_ops->tag; ((ops_tag == ValueTag_Vector) ? printf("    operations: OK\n") : printf("    operations: ERROR\n")); }); });
    printf("\nAll tests passed!\n");
    return 0;
}
int main() {
    init_namespace_user(&g_user);
    g_user.main_fn();
    return 0;
}
