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

// Required namespace: mlir-ast
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

extern Namespace_mlir_ast g_mlir_ast;
void init_namespace_mlir_ast(Namespace_mlir_ast* ns);


typedef struct {
    int32_t (*test_simple_expr)();
    int32_t (*test_op_form)();
    int32_t (*test_block_form)();
    int32_t (*main_fn)();
} Namespace_parse_test_files;

Namespace_parse_test_files g_parse_test_files;

static int32_t test_simple_expr();
static int32_t test_op_form();
static int32_t test_block_form();
static int32_t main_fn();

void init_namespace_parse_test_files(Namespace_parse_test_files* ns) {
    ns->test_simple_expr = &test_simple_expr;
    ns->test_op_form = &test_op_form;
    ns->test_block_form = &test_block_form;
    ns->main_fn = &main_fn;
}

static int32_t test_simple_expr() {
    printf("=== Test 1: Parse simple list (foo bar) ===\n");
    ({ Token* tokens = (Token*)((Token*)malloc(128)); Token* t0 = (Token*)tokens; t0->type = TokenType_LeftParen; t0->text = "("; t0->length = 1; ({ Token* t1 = (Token*)((Token*)(((long long)tokens) + 24)); t1->type = TokenType_Symbol; t1->text = "foo"; t1->length = 3; ({ Token* t2 = (Token*)((Token*)(((long long)tokens) + 48)); t2->type = TokenType_Symbol; t2->text = "bar"; t2->length = 3; ({ Token* t3 = (Token*)((Token*)(((long long)tokens) + 72)); t3->type = TokenType_RightParen; t3->text = ")"; t3->length = 1; ({ Parser* p = (Parser*)g_parser.make_parser(tokens, 4); Value* result = (Value*)g_parser.parse_value(p); printf("Parsed: "); g_parser.print_value_ptr(result); printf("\n\n"); 0; }); }); }); }); });
    return 0;
}
static int32_t test_op_form() {
    printf("=== Test 2: Parse and inspect op form ===\n");
    printf("Input: (op \"arith.constant\" [\"i32\"] [] {} [])\n");
    return ({ Value* nil_val = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_nil(); __tmp_ptr; }); Value* regions = (Value*)g_types.make_empty_vector(); Value* rest5 = (Value*)g_types.make_cons(regions, nil_val); Value* attrs = (Value*)g_types.make_empty_map(); Value* rest4 = (Value*)g_types.make_cons(attrs, rest5); Value* operands = (Value*)g_types.make_empty_vector(); Value* rest3 = (Value*)g_types.make_cons(operands, rest4); Value* result_types = (Value*)g_types.make_empty_vector(); Value* rest2 = (Value*)g_types.make_cons(result_types, rest3); Value* name = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_string("arith.constant"); __tmp_ptr; }); Value* rest1 = (Value*)g_types.make_cons(name, rest2); Value* op_sym = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_symbol("op"); __tmp_ptr; }); Value* op_form = (Value*)g_types.make_cons(op_sym, rest1); ({ int32_t is_op_result = g_mlir_ast.is_op(op_form); printf("Is this an op form? %s\n", ((is_op_result == 1) ? "YES" : "NO")); }); ({ Value* extracted_name = (Value*)g_mlir_ast.get_op_name(op_form); ValueTag name_tag = extracted_name->tag; ((name_tag == ValueTag_String) ? ({ uint8_t* name_str = (uint8_t*)extracted_name->str_val; printf("Op name: \"%s\"\n", name_str); }) : printf("ERROR: Could not extract op name\n")); }); ({ OpNode* op_node = (OpNode*)g_mlir_ast.parse_op(op_form); ((((int64_t)op_node) != 0) ? ({ uint8_t* parsed_name = (uint8_t*)op_node->name; printf("Parsed OpNode with name: \"%s\"\n", parsed_name); }) : printf("ERROR: parse-op failed\n")); }); printf("\n"); 0; });
}
static int32_t test_block_form() {
    printf("=== Test 3: Parse and inspect block form ===\n");
    printf("Input: (block [] [])\n");
    return ({ Value* nil_val = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_nil(); __tmp_ptr; }); Value* operations = (Value*)g_types.make_empty_vector(); Value* rest2 = (Value*)g_types.make_cons(operations, nil_val); Value* block_args = (Value*)g_types.make_empty_vector(); Value* rest1 = (Value*)g_types.make_cons(block_args, rest2); Value* block_sym = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_symbol("block"); __tmp_ptr; }); Value* block_form = (Value*)g_types.make_cons(block_sym, rest1); ({ int32_t is_block_result = g_mlir_ast.is_block(block_form); printf("Is this a block form? %s\n", ((is_block_result == 1) ? "YES" : "NO")); }); ({ BlockNode* block_node = (BlockNode*)g_mlir_ast.parse_block(block_form); ((((int64_t)block_node) != 0) ? printf("Successfully parsed BlockNode\n") : printf("ERROR: parse-block failed\n")); }); printf("\n"); 0; });
}
static int32_t main_fn() {
    printf("=== MLIR AST Parser Demo ===\n\n");
    printf("This demo shows parsing of MLIR-style op and block forms\n");
    printf("using the modular parser and AST libraries.\n\n");
    g_parse_test_files.test_simple_expr();
    g_parse_test_files.test_op_form();
    g_parse_test_files.test_block_form();
    printf("=== Demo Complete ===\n");
    return 0;
}
int main() {
    init_namespace_types(&g_types);
    init_namespace_mlir_ast(&g_mlir_ast);
    init_namespace_parse_test_files(&g_parse_test_files);
    // namespace parse-test-files
    // require [types :as types]
    // require [parser :as parser]
    // require [mlir-ast :as ast]
    g_parse_test_files.main_fn();
    return 0;
}
