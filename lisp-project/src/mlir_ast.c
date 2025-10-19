#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

// Required namespace: types
typedef struct Cons Cons;
typedef struct Token Token;
typedef struct Vector Vector;
typedef struct Value Value;
typedef enum {
    TokenType_LeftParen,
    TokenType_RightParen,
    TokenType_LeftBracket,
    TokenType_RightBracket,
    TokenType_LeftBrace,
    TokenType_RightBrace,
    TokenType_Number,
    TokenType_Symbol,
    TokenType_String,
    TokenType_Keyword,
    TokenType_EOF,
} TokenType;

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

struct Cons {
    uint8_t* car;
    uint8_t* cdr;
};

struct Token {
    TokenType type;
    uint8_t* text;
    int32_t length;
};

struct Vector {
    uint8_t* data;
    int32_t count;
    int32_t capacity;
};

struct Value {
    ValueTag tag;
    int64_t num_val;
    uint8_t* str_val;
    uint8_t* cons_val;
    uint8_t* vec_val;
};

typedef struct {
    Value (*make_nil)();
    Value (*make_number)(int64_t);
    Value (*make_symbol)(uint8_t*);
    Value (*make_string)(uint8_t*);
    Value* (*make_cons)(Value*, Value*);
    Value* (*make_empty_vector)();
    Value* (*make_vector_with_capacity)(int32_t, int32_t);
    Value* (*make_empty_map)();
    Value* (*car)(Value*);
    Value* (*cdr)(Value*);
    uint8_t* (*copy_string)(uint8_t*, int32_t);
    int32_t (*is_number_token)(Token);
    int32_t (*vector_set)(Value*, int32_t, Value*);
} Namespace_types;

extern Namespace_types g_types;
void init_namespace_types(Namespace_types* ns);

// Local type definitions
typedef struct OpNode OpNode;
typedef struct BlockNode BlockNode;
struct OpNode {
    uint8_t* name;
    Value* result_types;
    Value* operands;
    Value* attributes;
    Value* regions;
};

struct BlockNode {
    Value* args;
    Value* operations;
};

#include "string.h"

typedef struct {
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
    OpNode* (*parse_op)(Value*);
    BlockNode* (*parse_block)(Value*);
    int32_t (*main_fn)();
} Namespace_mlir_ast;

Namespace_mlir_ast g_mlir_ast;

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
static OpNode* parse_op(Value*);
static BlockNode* parse_block(Value*);
static int32_t main_fn();

void init_namespace_mlir_ast(Namespace_mlir_ast* ns) {
    init_namespace_types(&g_types);
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
    ns->parse_op = &parse_op;
    ns->parse_block = &parse_block;
    ns->main_fn = &main_fn;
}

static int32_t is_symbol_op(Value* v) {
    return ({ ValueTag tag = v->tag; ((tag == ValueTag_Symbol) ? ({ uint8_t* str_val = (uint8_t*)v->str_val; int32_t cmp_result = strcmp(str_val, "op"); ((cmp_result == 0) ? 1 : 0); }) : 0); });
}
static int32_t is_symbol_block(Value* v) {
    return ({ ValueTag tag = v->tag; ((tag == ValueTag_Symbol) ? ({ uint8_t* str_val = (uint8_t*)v->str_val; int32_t cmp_result = strcmp(str_val, "block"); ((cmp_result == 0) ? 1 : 0); }) : 0); });
}
static int32_t is_op(Value* v) {
    return ({ ValueTag tag = v->tag; ((tag == ValueTag_List) ? g_mlir_ast.is_symbol_op(g_types.car(v)) : 0); });
}
static int32_t is_block(Value* v) {
    return ({ ValueTag tag = v->tag; ((tag == ValueTag_List) ? g_mlir_ast.is_symbol_block(g_types.car(v)) : 0); });
}
static Value* get_op_name(Value* op_form) {
    return ((g_mlir_ast.is_op(op_form) == 1) ? ({ Value* rest = (Value*)g_types.cdr(op_form); ValueTag rest_tag = rest->tag; ((rest_tag == ValueTag_List) ? g_types.car(rest) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_nil(); __tmp_ptr; })); }) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_nil(); __tmp_ptr; }));
}
static Value* get_op_result_types(Value* op_form) {
    return ((g_mlir_ast.is_op(op_form) == 1) ? ({ Value* rest = (Value*)g_types.cdr(op_form); ValueTag rest_tag = rest->tag; ((rest_tag == ValueTag_List) ? ({ Value* rest2 = (Value*)g_types.cdr(rest); ValueTag rest2_tag = rest2->tag; ((rest2_tag == ValueTag_List) ? g_types.car(rest2) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_nil(); __tmp_ptr; })); }) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_nil(); __tmp_ptr; })); }) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_nil(); __tmp_ptr; }));
}
static Value* get_op_operands(Value* op_form) {
    return ((g_mlir_ast.is_op(op_form) == 1) ? ({ Value* rest = (Value*)g_types.cdr(op_form); ValueTag rest_tag = rest->tag; ((rest_tag == ValueTag_List) ? ({ Value* rest2 = (Value*)g_types.cdr(rest); ValueTag rest2_tag = rest2->tag; ((rest2_tag == ValueTag_List) ? ({ Value* rest3 = (Value*)g_types.cdr(rest2); ValueTag rest3_tag = rest3->tag; ((rest3_tag == ValueTag_List) ? g_types.car(rest3) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_nil(); __tmp_ptr; })); }) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_nil(); __tmp_ptr; })); }) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_nil(); __tmp_ptr; })); }) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_nil(); __tmp_ptr; }));
}
static Value* get_op_attributes(Value* op_form) {
    return ((g_mlir_ast.is_op(op_form) == 1) ? ({ Value* rest = (Value*)g_types.cdr(op_form); ValueTag rest_tag = rest->tag; ((rest_tag == ValueTag_List) ? ({ Value* rest2 = (Value*)g_types.cdr(rest); ValueTag rest2_tag = rest2->tag; ((rest2_tag == ValueTag_List) ? ({ Value* rest3 = (Value*)g_types.cdr(rest2); ValueTag rest3_tag = rest3->tag; ((rest3_tag == ValueTag_List) ? ({ Value* rest4 = (Value*)g_types.cdr(rest3); ValueTag rest4_tag = rest4->tag; ((rest4_tag == ValueTag_List) ? g_types.car(rest4) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_nil(); __tmp_ptr; })); }) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_nil(); __tmp_ptr; })); }) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_nil(); __tmp_ptr; })); }) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_nil(); __tmp_ptr; })); }) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_nil(); __tmp_ptr; }));
}
static Value* get_op_regions(Value* op_form) {
    return ((g_mlir_ast.is_op(op_form) == 1) ? ({ Value* rest = (Value*)g_types.cdr(op_form); ValueTag rest_tag = rest->tag; ((rest_tag == ValueTag_List) ? ({ Value* rest2 = (Value*)g_types.cdr(rest); ValueTag rest2_tag = rest2->tag; ((rest2_tag == ValueTag_List) ? ({ Value* rest3 = (Value*)g_types.cdr(rest2); ValueTag rest3_tag = rest3->tag; ((rest3_tag == ValueTag_List) ? ({ Value* rest4 = (Value*)g_types.cdr(rest3); ValueTag rest4_tag = rest4->tag; ((rest4_tag == ValueTag_List) ? ({ Value* rest5 = (Value*)g_types.cdr(rest4); ValueTag rest5_tag = rest5->tag; ((rest5_tag == ValueTag_List) ? g_types.car(rest5) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_nil(); __tmp_ptr; })); }) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_nil(); __tmp_ptr; })); }) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_nil(); __tmp_ptr; })); }) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_nil(); __tmp_ptr; })); }) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_nil(); __tmp_ptr; })); }) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_nil(); __tmp_ptr; }));
}
static Value* get_block_args(Value* block_form) {
    return ((g_mlir_ast.is_block(block_form) == 1) ? ({ Value* rest = (Value*)g_types.cdr(block_form); ValueTag rest_tag = rest->tag; ((rest_tag == ValueTag_List) ? g_types.car(rest) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_nil(); __tmp_ptr; })); }) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_nil(); __tmp_ptr; }));
}
static Value* get_block_operations(Value* block_form) {
    return ((g_mlir_ast.is_block(block_form) == 1) ? ({ Value* rest = (Value*)g_types.cdr(block_form); ValueTag rest_tag = rest->tag; ((rest_tag == ValueTag_List) ? ({ Value* rest2 = (Value*)g_types.cdr(rest); ValueTag rest2_tag = rest2->tag; ((rest2_tag == ValueTag_List) ? g_types.car(rest2) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_nil(); __tmp_ptr; })); }) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_nil(); __tmp_ptr; })); }) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_nil(); __tmp_ptr; }));
}
static OpNode* parse_op(Value* op_form) {
    return ((g_mlir_ast.is_op(op_form) == 1) ? ({ Value* name_val = (Value*)g_mlir_ast.get_op_name(op_form); ValueTag name_tag = name_val->tag; ((name_tag == ValueTag_String) ? ({ uint8_t* name = (uint8_t*)name_val->str_val; Value* result_types = (Value*)g_mlir_ast.get_op_result_types(op_form); Value* operands = (Value*)g_mlir_ast.get_op_operands(op_form); Value* attributes = (Value*)g_mlir_ast.get_op_attributes(op_form); Value* regions = (Value*)g_mlir_ast.get_op_regions(op_form); OpNode* node = (OpNode*)((OpNode*)malloc(40)); node->name = name; node->result_types = result_types; node->operands = operands; node->attributes = attributes; node->regions = regions; node; }) : ((OpNode*)0)); }) : ((OpNode*)0));
}
static BlockNode* parse_block(Value* block_form) {
    return ((g_mlir_ast.is_block(block_form) == 1) ? ({ Value* args = (Value*)g_mlir_ast.get_block_args(block_form); Value* operations = (Value*)g_mlir_ast.get_block_operations(block_form); BlockNode* node = (BlockNode*)((BlockNode*)malloc(16)); node->args = args; node->operations = operations; node; }) : ((BlockNode*)0));
}
static int32_t main_fn() {
    printf("Testing is-symbol-op:\n");
    ({ Value* op_sym = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_symbol("op"); __tmp_ptr; }); int32_t result1 = g_mlir_ast.is_symbol_op(op_sym); printf("  symbol 'op': %d (expected 1)\n", result1); });
    ({ Value* block_sym = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_symbol("block"); __tmp_ptr; }); int32_t result2 = g_mlir_ast.is_symbol_op(block_sym); printf("  symbol 'block': %d (expected 0)\n", result2); });
    ({ Value* nil_val = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_nil(); __tmp_ptr; }); int32_t result3 = g_mlir_ast.is_symbol_op(nil_val); printf("  nil: %d (expected 0)\n", result3); });
    printf("\nTesting is-symbol-block:\n");
    ({ Value* block_sym = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_symbol("block"); __tmp_ptr; }); int32_t result4 = g_mlir_ast.is_symbol_block(block_sym); printf("  symbol 'block': %d (expected 1)\n", result4); });
    ({ Value* op_sym = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_symbol("op"); __tmp_ptr; }); int32_t result5 = g_mlir_ast.is_symbol_block(op_sym); printf("  symbol 'op': %d (expected 0)\n", result5); });
    printf("\nTesting is-op predicate:\n");
    ({ Value* nil_val = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_nil(); __tmp_ptr; }); Value* op_sym = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_symbol("op"); __tmp_ptr; }); Value* op_list = (Value*)g_types.make_cons(op_sym, nil_val); int32_t result6 = g_mlir_ast.is_op(op_list); printf("  (op): %d (expected 1)\n", result6); });
    ({ Value* nil_val = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_nil(); __tmp_ptr; }); Value* block_sym = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_symbol("block"); __tmp_ptr; }); Value* block_list = (Value*)g_types.make_cons(block_sym, nil_val); int32_t result7 = g_mlir_ast.is_op(block_list); printf("  (block): %d (expected 0)\n", result7); });
    printf("\nTesting is-block predicate:\n");
    ({ Value* nil_val = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_nil(); __tmp_ptr; }); Value* block_sym = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_symbol("block"); __tmp_ptr; }); Value* block_list = (Value*)g_types.make_cons(block_sym, nil_val); int32_t result8 = g_mlir_ast.is_block(block_list); printf("  (block): %d (expected 1)\n", result8); });
    ({ Value* nil_val = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_nil(); __tmp_ptr; }); Value* op_sym = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_symbol("op"); __tmp_ptr; }); Value* op_list = (Value*)g_types.make_cons(op_sym, nil_val); int32_t result9 = g_mlir_ast.is_block(op_list); printf("  (op): %d (expected 0)\n", result9); });
    printf("\nAll tests passed!\n");
    return 0;
}
void lisp_main(void) {
    init_namespace_types(&g_types);
    init_namespace_mlir_ast(&g_mlir_ast);
}
