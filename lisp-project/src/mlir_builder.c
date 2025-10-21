#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "stdint.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/ExecutionEngine.h"
#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
#include "mlir-c/RegisterEverything.h"

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
    int32_t (*print_indent)(int32_t);
    int32_t (*process_vector_elements)(Value*, int32_t);
    int32_t (*parse_and_print_recursive)(Value*, int32_t);
    BlockNode* (*parse_block_recursive)(Value*);
    OpNode* (*parse_op_recursive)(Value*);
    int32_t (*main_fn)();
} Namespace_mlir_ast;

extern Namespace_mlir_ast g_mlir_ast;
void init_namespace_mlir_ast(Namespace_mlir_ast* ns);

// Required namespace: tokenizer
typedef struct Tokenizer Tokenizer;
struct Tokenizer {
    uint8_t* input;
    int32_t position;
    int32_t length;
};

typedef struct {
    Tokenizer* (*make_tokenizer)(uint8_t*);
    int32_t (*peek_char)(Tokenizer*);
    int32_t (*advance)(Tokenizer*);
    int32_t (*skip_to_eol)(Tokenizer*);
    int32_t (*skip_whitespace)(Tokenizer*);
    Token (*make_token)(TokenType, uint8_t*, int32_t);
    Token (*next_token)(Tokenizer*);
    Token (*read_symbol)(Tokenizer*);
    Token (*read_string)(Tokenizer*);
    Token (*read_keyword)(Tokenizer*);
    int32_t (*main_fn)();
} Namespace_tokenizer;

extern Namespace_tokenizer g_tokenizer;
void init_namespace_tokenizer(Namespace_tokenizer* ns);

// Required namespace: parser
typedef struct Parser Parser;
struct Parser {
    Token* tokens;
    int32_t position;
    int32_t count;
};

typedef struct {
    Parser* (*make_parser)(Token*, int32_t);
    Token (*peek_token)(Parser*);
    int32_t (*advance_parser)(Parser*);
    Value* (*parse_list)(Parser*);
    int32_t (*parse_vector_elements)(Parser*, Value*, int32_t);
    Value* (*parse_vector)(Parser*);
    int32_t (*parse_map_elements)(Parser*, Value*, int32_t);
    Value* (*parse_map)(Parser*);
    Value* (*parse_value)(Parser*);
    int32_t (*print_vector_contents)(Value*, int32_t);
    int32_t (*print_map_contents)(Value*, int32_t);
    int32_t (*print_value_ptr)(Value*);
    int32_t (*print_list_contents)(Value*);
    int32_t (*main_fn)();
} Namespace_parser;

extern Namespace_parser g_parser;
void init_namespace_parser(Namespace_parser* ns);

// Local type definitions
typedef struct ValueMapEntry ValueMapEntry;
typedef struct MLIRBuilderContext MLIRBuilderContext;
typedef struct ValueTracker ValueTracker;
struct ValueMapEntry {
    uint8_t* name;
    MlirValue value;
    ValueMapEntry* next;
};

struct MLIRBuilderContext {
    MlirContext ctx;
    MlirLocation loc;
    MlirModule mod;
    MlirType i32Type;
    MlirType i64Type;
};

struct ValueTracker {
    ValueMapEntry* head;
};


typedef struct {
    void (*die)(uint8_t*);
    ValueTracker* (*value_tracker_create)();
    ValueMapEntry* (*value_tracker_find_entry)(ValueMapEntry*, uint8_t*);
    int32_t (*value_tracker_register)(ValueTracker*, uint8_t*, MlirValue);
    MlirValue (*value_tracker_missing)(uint8_t*);
    MlirValue (*value_tracker_lookup)(ValueTracker*, uint8_t*);
    MLIRBuilderContext* (*mlir_builder_init)();
    int32_t (*mlir_builder_destroy)(MLIRBuilderContext*);
    MlirType (*parse_type_string_helper)(MLIRBuilderContext*, uint8_t*);
    MlirType (*parse_type_string)(MLIRBuilderContext*, uint8_t*);
    int64_t (*parse_int_attr_value)(uint8_t*);
    MlirAttribute (*parse_integer_value_attr)(MLIRBuilderContext*, Value*);
    MlirAttribute (*parse_function_type_attr)(MLIRBuilderContext*, Value*);
    MlirAttribute (*create_attribute_from_value)(MLIRBuilderContext*, uint8_t*, Value*);
    MlirNamedAttribute (*create_named_attribute_from_value)(MLIRBuilderContext*, uint8_t*, Value*);
    MlirRegion (*build_mlir_region)(MLIRBuilderContext*, Value*, ValueTracker*);
    Value* (*vector_element)(Value*, int32_t);
    MlirBlock (*create_block_with_args)(MLIRBuilderContext*, ValueTracker*, MlirLocation, Value*);
    MlirBlock (*build_mlir_block)(MLIRBuilderContext*, Value*, ValueTracker*);
    uint8_t* (*result_type_string)(Value*);
    int32_t (*add_result_types_to_state)(MLIRBuilderContext*, MlirOperationState*, Value*);
    int32_t (*add_operands_to_state)(MLIRBuilderContext*, MlirOperationState*, Value*, ValueTracker*);
    int32_t (*add_attributes_to_state)(MLIRBuilderContext*, MlirOperationState*, Value*);
    int32_t (*add_regions_to_state)(MLIRBuilderContext*, MlirOperationState*, Value*, ValueTracker*);
    int32_t (*run_lowering_passes)(MLIRBuilderContext*);
    int32_t (*jit_run_function_i32)(MLIRBuilderContext*, uint8_t*, int32_t);
    MlirOperation (*build_mlir_operation)(MLIRBuilderContext*, OpNode*, ValueTracker*, MlirBlock);
    int32_t (*compile_op_to_mlir)(MLIRBuilderContext*, Value*);
    uint8_t* (*read_file)(uint8_t*);
    Token* (*tokenize_file)(uint8_t*);
    int32_t (*count_tokens)(Token*);
    int32_t (*compile_file)(MLIRBuilderContext*, uint8_t*);
    int32_t (*main_fn)();
} Namespace_mlir_builder;

Namespace_mlir_builder g_mlir_builder;

static void die(uint8_t*);
static ValueTracker* value_tracker_create();
static ValueMapEntry* value_tracker_find_entry(ValueMapEntry*, uint8_t*);
static int32_t value_tracker_register(ValueTracker*, uint8_t*, MlirValue);
static MlirValue value_tracker_missing(uint8_t*);
static MlirValue value_tracker_lookup(ValueTracker*, uint8_t*);
static MLIRBuilderContext* mlir_builder_init();
static int32_t mlir_builder_destroy(MLIRBuilderContext*);
static MlirType parse_type_string_helper(MLIRBuilderContext*, uint8_t*);
static MlirType parse_type_string(MLIRBuilderContext*, uint8_t*);
static int64_t parse_int_attr_value(uint8_t*);
static MlirAttribute parse_integer_value_attr(MLIRBuilderContext*, Value*);
static MlirAttribute parse_function_type_attr(MLIRBuilderContext*, Value*);
static MlirAttribute create_attribute_from_value(MLIRBuilderContext*, uint8_t*, Value*);
static MlirNamedAttribute create_named_attribute_from_value(MLIRBuilderContext*, uint8_t*, Value*);
static MlirRegion build_mlir_region(MLIRBuilderContext*, Value*, ValueTracker*);
static Value* vector_element(Value*, int32_t);
static MlirBlock create_block_with_args(MLIRBuilderContext*, ValueTracker*, MlirLocation, Value*);
static MlirBlock build_mlir_block(MLIRBuilderContext*, Value*, ValueTracker*);
static uint8_t* result_type_string(Value*);
static int32_t add_result_types_to_state(MLIRBuilderContext*, MlirOperationState*, Value*);
static int32_t add_operands_to_state(MLIRBuilderContext*, MlirOperationState*, Value*, ValueTracker*);
static int32_t add_attributes_to_state(MLIRBuilderContext*, MlirOperationState*, Value*);
static int32_t add_regions_to_state(MLIRBuilderContext*, MlirOperationState*, Value*, ValueTracker*);
static int32_t run_lowering_passes(MLIRBuilderContext*);
static int32_t jit_run_function_i32(MLIRBuilderContext*, uint8_t*, int32_t);
static MlirOperation build_mlir_operation(MLIRBuilderContext*, OpNode*, ValueTracker*, MlirBlock);
static int32_t compile_op_to_mlir(MLIRBuilderContext*, Value*);
static uint8_t* read_file(uint8_t*);
static Token* tokenize_file(uint8_t*);
static int32_t count_tokens(Token*);
static int32_t compile_file(MLIRBuilderContext*, uint8_t*);
static int32_t main_fn();

void init_namespace_mlir_builder(Namespace_mlir_builder* ns) {
    ns->die = &die;
    ns->value_tracker_create = &value_tracker_create;
    ns->value_tracker_find_entry = &value_tracker_find_entry;
    ns->value_tracker_register = &value_tracker_register;
    ns->value_tracker_missing = &value_tracker_missing;
    ns->value_tracker_lookup = &value_tracker_lookup;
    ns->mlir_builder_init = &mlir_builder_init;
    ns->mlir_builder_destroy = &mlir_builder_destroy;
    ns->parse_type_string_helper = &parse_type_string_helper;
    ns->parse_type_string = &parse_type_string;
    ns->parse_int_attr_value = &parse_int_attr_value;
    ns->parse_integer_value_attr = &parse_integer_value_attr;
    ns->parse_function_type_attr = &parse_function_type_attr;
    ns->create_attribute_from_value = &create_attribute_from_value;
    ns->create_named_attribute_from_value = &create_named_attribute_from_value;
    ns->build_mlir_region = &build_mlir_region;
    ns->vector_element = &vector_element;
    ns->create_block_with_args = &create_block_with_args;
    ns->build_mlir_block = &build_mlir_block;
    ns->result_type_string = &result_type_string;
    ns->add_result_types_to_state = &add_result_types_to_state;
    ns->add_operands_to_state = &add_operands_to_state;
    ns->add_attributes_to_state = &add_attributes_to_state;
    ns->add_regions_to_state = &add_regions_to_state;
    ns->run_lowering_passes = &run_lowering_passes;
    ns->jit_run_function_i32 = &jit_run_function_i32;
    ns->build_mlir_operation = &build_mlir_operation;
    ns->compile_op_to_mlir = &compile_op_to_mlir;
    ns->read_file = &read_file;
    ns->tokenize_file = &tokenize_file;
    ns->count_tokens = &count_tokens;
    ns->compile_file = &compile_file;
    ns->main_fn = &main_fn;
}

static void die(uint8_t* msg) {
    printf("ERROR: %s\n", msg);
    fflush(stderr);
    exit(1);
}
static ValueTracker* value_tracker_create() {
    return ({ uint8_t* tracker_bytes = (uint8_t*)malloc(8); ValueTracker* tracker = (ValueTracker*)((ValueTracker*)tracker_bytes); tracker->head = ((ValueMapEntry*)0); tracker; });
}
static ValueMapEntry* value_tracker_find_entry(ValueMapEntry* entry, uint8_t* name) {
    return ({ ValueMapEntry* __if_result; if ((((int64_t)entry) == 0)) { __if_result = ((ValueMapEntry*)0); } else { __if_result = ({ uint8_t* entry_name = (uint8_t*)entry->name; int32_t cmp = strcmp(entry_name, name); ((cmp == 0) ? entry : g_mlir_builder.value_tracker_find_entry(entry->next, name)); }); } __if_result; });
}
static int32_t value_tracker_register(ValueTracker* tracker, uint8_t* name, MlirValue value) {
    return ({ ValueMapEntry* head = (ValueMapEntry*)tracker->head; ValueMapEntry* existing = (ValueMapEntry*)g_mlir_builder.value_tracker_find_entry(head, name); ({ long long __if_result; if ((((int64_t)existing) != 0)) { __if_result = ({ existing->value = value; 0; }); } else { __if_result = ({ uint8_t* entry_bytes = (uint8_t*)malloc(32); ValueMapEntry* new_entry = (ValueMapEntry*)((ValueMapEntry*)entry_bytes); new_entry->name = name; new_entry->value = value; new_entry->next = head; tracker->head = new_entry; 0; }); } __if_result; }); });
}
static MlirValue value_tracker_missing(uint8_t* name) {
    return ({ printf("ERROR: Unknown SSA value '%s'\n", name); g_mlir_builder.die("SSA lookup failed"); ({ MlirValue* tmp = (MlirValue*)((MlirValue*)malloc(16)); (*tmp); }); });
}
static MlirValue value_tracker_lookup(ValueTracker* tracker, uint8_t* name) {
    return ({ ValueMapEntry* head = (ValueMapEntry*)tracker->head; ValueMapEntry* entry = (ValueMapEntry*)g_mlir_builder.value_tracker_find_entry(head, name); ((((int64_t)entry) == 0) ? g_mlir_builder.value_tracker_missing(name) : entry->value); });
}
static MLIRBuilderContext* mlir_builder_init() {
    return ({ MlirContext ctx = mlirContextCreate(); MlirDialectRegistry registry = mlirDialectRegistryCreate(); mlirRegisterAllDialects(registry); mlirContextAppendDialectRegistry(ctx, registry); mlirDialectRegistryDestroy(registry); mlirContextLoadAllAvailableDialects(ctx); mlirRegisterAllPasses(); mlirRegisterAllLLVMTranslations(ctx); ({ MlirLocation loc = mlirLocationUnknownGet(ctx); MlirModule mod = mlirModuleCreateEmpty(loc); MlirType i32Type = mlirIntegerTypeGet(ctx, 32); MlirType i64Type = mlirIntegerTypeGet(ctx, 64); MLIRBuilderContext* builder = (MLIRBuilderContext*)((MLIRBuilderContext*)malloc(40)); builder->ctx = ctx; builder->loc = loc; builder->mod = mod; builder->i32Type = i32Type; builder->i64Type = i64Type; builder; }); });
}
static int32_t mlir_builder_destroy(MLIRBuilderContext* builder) {
    return ({ MlirContext ctx = builder->ctx; mlirContextDestroy(ctx); 0; });
}
static MlirType parse_type_string_helper(MLIRBuilderContext* builder, uint8_t* type_str) {
    return ({ int32_t cmp_i64 = strcmp(type_str, "i64"); ({ MlirType __if_result; if ((cmp_i64 == 0)) { __if_result = builder->i64Type; } else { __if_result = ({ int32_t cmp_i1 = strcmp(type_str, "i1"); ({ MlirType __if_result; if ((cmp_i1 == 0)) { __if_result = ({ MlirContext ctx = builder->ctx; mlirIntegerTypeGet(ctx, 1); }); } else { __if_result = ({ printf("Unknown type: %s, defaulting to i32\n", type_str); builder->i32Type; }); } __if_result; }); }); } __if_result; }); });
}
static MlirType parse_type_string(MLIRBuilderContext* builder, uint8_t* type_str) {
    return ({ int32_t cmp_i32 = strcmp(type_str, "i32"); ((cmp_i32 == 0) ? builder->i32Type : g_mlir_builder.parse_type_string_helper(builder, type_str)); });
}
static int64_t parse_int_attr_value(uint8_t* value_str) {
    return ((int64_t)atoi(value_str));
}
static MlirAttribute parse_integer_value_attr(MLIRBuilderContext* builder, Value* value_list) {
    return ({ Value* first_elem = (Value*)g_types.car(value_list); ({ Value* rest = (Value*)g_types.cdr(value_list); ({ ValueTag rest_tag = rest->tag; ({ MlirAttribute __if_result; if ((rest_tag == ValueTag_List)) { __if_result = ({ Value* second_elem = (Value*)g_types.car(rest); int64_t int_val = first_elem->num_val; uint8_t* type_str = (uint8_t*)second_elem->str_val; MlirType mlir_type = g_mlir_builder.parse_type_string(builder, type_str); mlirIntegerAttrGet(mlir_type, int_val); }); } else { __if_result = ({ MlirContext ctx = builder->ctx; printf("    ERROR: Invalid integer value format, rest-tag=%d\n", ((int32_t)rest_tag)); mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("ERROR")); }); } __if_result; }); }); }); });
}
static MlirAttribute parse_function_type_attr(MLIRBuilderContext* builder, Value* value_list) {
    return ({ MlirContext ctx = builder->ctx; MlirType input_types[8]; MlirType result_types[8]; int32_t input_count = 0; int32_t result_count = 0; Value* args_list = (Value*)g_types.cdr(value_list); Value* inputs_val = (Value*)((((int64_t)args_list) == 0) ? ((Value*)0) : g_types.car(args_list)); Value* outputs_list = (Value*)((((int64_t)args_list) == 0) ? ((Value*)0) : g_types.cdr(args_list)); Value* outputs_val = (Value*)((((int64_t)outputs_list) == 0) ? ((Value*)0) : g_types.car(outputs_list)); ({ long long __if_result; if ((((int64_t)inputs_val) != 0)) { __if_result = ({ ValueTag inputs_tag = inputs_val->tag; ({ long long __if_result; if ((inputs_tag == ValueTag_Vector)) { __if_result = ({ Vector* vec_struct = (Vector*)((Vector*)inputs_val->vec_val); int32_t vec_count = vec_struct->count; int32_t limit = ((vec_count > 8) ? 8 : vec_count); int32_t idx = 0; input_count = limit; ({ while ((idx < limit)) { ({ Value* elem = (Value*)g_mlir_builder.vector_element(inputs_val, idx); ValueTag elem_tag = elem->tag; uint8_t* type_str = (uint8_t*)((elem_tag == ValueTag_Symbol) ? elem->str_val : ((elem_tag == ValueTag_String) ? elem->str_val : "i32")); MlirType mlir_type = g_mlir_builder.parse_type_string(builder, type_str); (input_types[idx] = mlir_type); idx = (idx + 1); 0; }); } }); 0; }); } else { __if_result = 0; } __if_result; }); }); } else { __if_result = 0; } __if_result; }); ({ long long __if_result; if ((((int64_t)outputs_val) != 0)) { __if_result = ({ ValueTag outputs_tag = outputs_val->tag; ({ long long __if_result; if ((outputs_tag == ValueTag_Vector)) { __if_result = ({ Vector* vec_struct = (Vector*)((Vector*)outputs_val->vec_val); int32_t vec_count = vec_struct->count; int32_t limit = ((vec_count > 8) ? 8 : vec_count); int32_t idx = 0; result_count = limit; ({ while ((idx < limit)) { ({ Value* elem = (Value*)g_mlir_builder.vector_element(outputs_val, idx); ValueTag elem_tag = elem->tag; uint8_t* type_str = (uint8_t*)((elem_tag == ValueTag_Symbol) ? elem->str_val : ((elem_tag == ValueTag_String) ? elem->str_val : "i32")); MlirType mlir_type = g_mlir_builder.parse_type_string(builder, type_str); (result_types[idx] = mlir_type); idx = (idx + 1); 0; }); } }); 0; }); } else { __if_result = 0; } __if_result; }); }); } else { __if_result = 0; } __if_result; }); ({ MlirType* input_ptr = (MlirType*)((input_count > 0) ? (&input_types[0]) : ((MlirType*)0)); MlirType* result_ptr = (MlirType*)((result_count > 0) ? (&result_types[0]) : ((MlirType*)0)); MlirType fn_type = mlirFunctionTypeGet(ctx, ((int64_t)input_count), input_ptr, ((int64_t)result_count), result_ptr); mlirTypeAttrGet(fn_type); }); });
}
static MlirAttribute create_attribute_from_value(MLIRBuilderContext* builder, uint8_t* key, Value* value_val) {
    return ({ MlirContext ctx = builder->ctx; ValueTag value_tag = value_val->tag; printf("  Creating attribute: %s (tag=%d)\n", key, ((int32_t)value_tag)); ({ MlirAttribute __if_result; if ((value_tag == ValueTag_String)) { __if_result = ({ uint8_t* value_str = (uint8_t*)value_val->str_val; ({ MlirAttribute __if_result; if ((strcmp(key, "callee") == 0)) { __if_result = ({ uint8_t first_char = (*value_str); uint8_t* symbol_name = (uint8_t*)((first_char == ((uint8_t)64)) ? ((uint8_t*)(((long long)value_str) + 1)) : value_str); MlirStringRef sym_ref = mlirStringRefCreateFromCString(symbol_name); mlirFlatSymbolRefAttrGet(ctx, sym_ref); }); } else { __if_result = ({ MlirAttribute __if_result; if ((strcmp(key, "predicate") == 0)) { __if_result = ({ int64_t pred_val = ((strcmp(value_str, "sle") == 0) ? 3 : ((strcmp(value_str, "eq") == 0) ? 0 : ((strcmp(value_str, "ne") == 0) ? 1 : ((strcmp(value_str, "slt") == 0) ? 2 : ((strcmp(value_str, "sgt") == 0) ? 4 : ((strcmp(value_str, "sge") == 0) ? 5 : ((strcmp(value_str, "ult") == 0) ? 6 : ((strcmp(value_str, "ule") == 0) ? 7 : ((strcmp(value_str, "ugt") == 0) ? 8 : ((strcmp(value_str, "uge") == 0) ? 9 : 0)))))))))); MlirType i64_type = builder->i64Type; mlirIntegerAttrGet(i64_type, pred_val); }); } else { __if_result = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString(value_str)); } __if_result; }); } __if_result; }); }); } else { __if_result = ({ MlirAttribute __if_result; if ((value_tag == ValueTag_List)) { __if_result = ((strcmp(key, "function_type") == 0) ? g_mlir_builder.parse_function_type_attr(builder, value_val) : ((strcmp(key, "value") == 0) ? g_mlir_builder.parse_integer_value_attr(builder, value_val) : ({ MlirAttribute __if_result; if ((strcmp(key, "callee") == 0)) { __if_result = ({ Value* first_elem = (Value*)g_types.car(value_val); uint8_t* str_val = (uint8_t*)first_elem->str_val; mlirStringAttrGet(ctx, mlirStringRefCreateFromCString(str_val)); }); } else { __if_result = ({ printf("    WARNING: Unknown list attribute type\n"); mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("TODO")); }); } __if_result; }))); } else { __if_result = ({ printf("    ERROR: Unknown attribute value type\n"); mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("ERROR")); }); } __if_result; }); } __if_result; }); });
}
static MlirNamedAttribute create_named_attribute_from_value(MLIRBuilderContext* builder, uint8_t* key, Value* value_val) {
    return ({ MlirContext ctx = builder->ctx; MlirIdentifier name_id = mlirIdentifierGet(ctx, mlirStringRefCreateFromCString(key)); MlirAttribute attr = g_mlir_builder.create_attribute_from_value(builder, key, value_val); mlirNamedAttributeGet(name_id, attr); });
}
static MlirRegion build_mlir_region(MLIRBuilderContext* builder, Value* region_vec, ValueTracker* tracker) {
    return ({ MlirRegion region = mlirRegionCreate(); ValueTag tag = region_vec->tag; ({ MlirRegion __if_result; if ((tag == ValueTag_Vector)) { __if_result = ({ uint8_t* vec_ptr = (uint8_t*)region_vec->vec_val; Vector* vector_struct = (Vector*)((Vector*)vec_ptr); int32_t count = vector_struct->count; uint8_t* data = (uint8_t*)vector_struct->data; int32_t idx = 0; ({ while ((idx < count)) { ({ int64_t elem_offset = (((long long)idx) * 8); uint8_t* elem_ptr_loc = (uint8_t*)((uint8_t*)(((int64_t)data) + elem_offset)); Value** elem_ptr_ptr = (Value**)((Value**)elem_ptr_loc); Value* block_form = (Value*)(*elem_ptr_ptr); MlirBlock mlir_block = g_mlir_builder.build_mlir_block(builder, block_form, tracker); mlirRegionAppendOwnedBlock(region, mlir_block); idx = (idx + 1); }); } }); region; }); } else { __if_result = region; } __if_result; }); });
}
static Value* vector_element(Value* vec_val, int32_t idx) {
    return ({ uint8_t* vec_ptr = (uint8_t*)vec_val->vec_val; Vector* vector_struct = (Vector*)((Vector*)vec_ptr); uint8_t* data = (uint8_t*)vector_struct->data; int64_t elem_offset = (((long long)idx) * 8); uint8_t* elem_ptr_loc = (uint8_t*)((uint8_t*)(((int64_t)data) + elem_offset)); Value** elem_ptr_ptr = (Value**)((Value**)elem_ptr_loc); (*elem_ptr_ptr); });
}
static MlirBlock create_block_with_args(MLIRBuilderContext* builder, ValueTracker* tracker, MlirLocation loc, Value* args_val) {
    return ({ ValueTag tag = args_val->tag; ({ MlirBlock __if_result; if ((tag == ValueTag_Vector)) { __if_result = ({ uint8_t* vec_ptr = (uint8_t*)args_val->vec_val; Vector* vector_struct = (Vector*)((Vector*)vec_ptr); int32_t count = vector_struct->count; ({ MlirBlock __if_result; if ((count > 0)) { __if_result = ({ MlirType arg_types[16]; MlirLocation arg_locs[16]; int32_t idx = 0; ({ while ((idx < count)) { ({ Value* pair = (Value*)g_mlir_builder.vector_element(args_val, idx); ValueTag pair_tag = pair->tag; MlirType mlir_type = ({ MlirType __if_result; if ((pair_tag == ValueTag_Vector)) { __if_result = ({ Value* type_elem = (Value*)g_mlir_builder.vector_element(pair, 1); uint8_t* type_str = (uint8_t*)type_elem->str_val; g_mlir_builder.parse_type_string(builder, type_str); }); } else { __if_result = builder->i32Type; } __if_result; }); (arg_types[idx] = mlir_type); (arg_locs[idx] = loc); idx = (idx + 1); }); } }); ({ MlirBlock block = mlirBlockCreate(((int64_t)count), (&arg_types[0]), (&arg_locs[0])); int32_t reg_idx = 0; ({ while ((reg_idx < count)) { ({ Value* pair = (Value*)g_mlir_builder.vector_element(args_val, reg_idx); ValueTag pair_tag = pair->tag; ({ long long __if_result; if ((pair_tag == ValueTag_Vector)) { __if_result = ({ Value* name_elem = (Value*)g_mlir_builder.vector_element(pair, 0); uint8_t* name_str = (uint8_t*)name_elem->str_val; MlirValue arg_val = mlirBlockGetArgument(block, ((int64_t)reg_idx)); g_mlir_builder.value_tracker_register(tracker, name_str, arg_val); }); } else { __if_result = 0; } __if_result; }); reg_idx = (reg_idx + 1); }); } }); block; }); }); } else { __if_result = mlirBlockCreate(0, ((MlirType*)0), ((MlirLocation*)0)); } __if_result; }); }); } else { __if_result = mlirBlockCreate(0, ((MlirType*)0), ((MlirLocation*)0)); } __if_result; }); });
}
static MlirBlock build_mlir_block(MLIRBuilderContext* builder, Value* block_form, ValueTracker* tracker) {
    return ({ BlockNode* block_node = (BlockNode*)g_mlir_ast.parse_block(block_form); ({ MlirBlock __if_result; if ((((int64_t)block_node) != 0)) { __if_result = ({ MlirLocation loc = builder->loc; Value* args_val = (Value*)block_node->args; MlirBlock block = g_mlir_builder.create_block_with_args(builder, tracker, loc, args_val); Value* operations = (Value*)block_node->operations; ValueTag ops_tag = operations->tag; ({ long long __if_result; if ((ops_tag == ValueTag_Vector)) { __if_result = ({ uint8_t* vec_ptr = (uint8_t*)operations->vec_val; Vector* vector_struct = (Vector*)((Vector*)vec_ptr); int32_t count = vector_struct->count; int32_t idx = 0; ({ while ((idx < count)) { ({ Value* op_form = (Value*)g_mlir_builder.vector_element(operations, idx); OpNode* op_node = (OpNode*)g_mlir_ast.parse_op(op_form); ({ long long __if_result; if ((((int64_t)op_node) != 0)) { __if_result = ({ g_mlir_builder.build_mlir_operation(builder, op_node, tracker, block); 0; }); } else { __if_result = 0; } __if_result; }); idx = (idx + 1); 0; }); } }); 0; }); } else { __if_result = 0; } __if_result; }); block; }); } else { __if_result = mlirBlockCreate(0, ((MlirType*)0), ((MlirLocation*)0)); } __if_result; }); });
}
static uint8_t* result_type_string(Value* elem) {
    return ({ ValueTag elem_tag = elem->tag; ({ uint8_t* __if_result; if ((elem_tag == ValueTag_Vector)) { __if_result = ({ Value* type_elem = (Value*)g_mlir_builder.vector_element(elem, 1); type_elem->str_val; }); } else { __if_result = ((elem_tag == ValueTag_String) ? elem->str_val : ((elem_tag == ValueTag_Symbol) ? elem->str_val : ((uint8_t*)0))); } __if_result; }); });
}
static int32_t add_result_types_to_state(MLIRBuilderContext* builder, MlirOperationState* state_ptr, Value* result_types_val) {
    return ({ ValueTag tag = result_types_val->tag; ({ long long __if_result; if ((tag == ValueTag_Vector)) { __if_result = ({ uint8_t* vec_ptr = (uint8_t*)result_types_val->vec_val; Vector* vector_struct = (Vector*)((Vector*)vec_ptr); int32_t raw_count = vector_struct->count; ({ long long __if_result; if ((raw_count > 0)) { __if_result = ({ Value* first_elem = (Value*)g_mlir_builder.vector_element(result_types_val, 0); ValueTag first_tag = first_elem->tag; int32_t result_count = ((first_tag == ValueTag_Vector) ? raw_count : ((int32_t)(((double)raw_count) / 2))); ({ long long __if_result; if ((result_count > 0)) { __if_result = ({ MlirType result_types_array[8]; int32_t idx = 0; ({ while ((idx < result_count)) { ({ Value* type_val = (Value*)({ Value* __if_result; if ((first_tag == ValueTag_Vector)) { __if_result = ({ Value* elem = (Value*)g_mlir_builder.vector_element(result_types_val, idx); g_mlir_builder.vector_element(elem, 1); }); } else { __if_result = ({ int32_t type_idx = ((idx * 2) + 1); g_mlir_builder.vector_element(result_types_val, type_idx); }); } __if_result; }); uint8_t* type_str = (uint8_t*)g_mlir_builder.result_type_string(type_val); MlirType mlir_type = ((((int64_t)type_str) != 0) ? g_mlir_builder.parse_type_string(builder, type_str) : builder->i32Type); (result_types_array[idx] = mlir_type); idx = (idx + 1); }); } }); mlirOperationStateAddResults(state_ptr, ((int64_t)result_count), (&result_types_array[0])); result_count; }); } else { __if_result = 0; } __if_result; }); }); } else { __if_result = 0; } __if_result; }); }); } else { __if_result = 0; } __if_result; }); });
}
static int32_t add_operands_to_state(MLIRBuilderContext* builder, MlirOperationState* state_ptr, Value* operands_val, ValueTracker* tracker) {
    return ({ ValueTag tag = operands_val->tag; ({ long long __if_result; if ((tag == ValueTag_Vector)) { __if_result = ({ uint8_t* vec_ptr = (uint8_t*)operands_val->vec_val; Vector* vector_struct = (Vector*)((Vector*)vec_ptr); int32_t count = vector_struct->count; ({ long long __if_result; if ((count > 0)) { __if_result = ({ MlirValue operands_array[8]; int32_t idx = 0; ({ while ((idx < count)) { ({ Value* elem = (Value*)g_mlir_builder.vector_element(operands_val, idx); ValueTag elem_tag = elem->tag; uint8_t* operand_name = (uint8_t*)((elem_tag == ValueTag_String) ? elem->str_val : ((elem_tag == ValueTag_Symbol) ? elem->str_val : ((uint8_t*)0))); ({ long long __if_result; if ((((int64_t)operand_name) != 0)) { __if_result = ({ MlirValue operand_val = g_mlir_builder.value_tracker_lookup(tracker, operand_name); (operands_array[idx] = operand_val); 0; }); } else { __if_result = 0; } __if_result; }); idx = (idx + 1); }); } }); mlirOperationStateAddOperands(state_ptr, ((int64_t)count), (&operands_array[0])); count; }); } else { __if_result = 0; } __if_result; }); }); } else { __if_result = 0; } __if_result; }); });
}
static int32_t add_attributes_to_state(MLIRBuilderContext* builder, MlirOperationState* state_ptr, Value* attributes_val) {
    return ({ ValueTag tag = attributes_val->tag; ({ long long __if_result; if ((tag == ValueTag_Map)) { __if_result = ({ uint8_t* vec_ptr = (uint8_t*)attributes_val->vec_val; Vector* vector_struct = (Vector*)((Vector*)vec_ptr); int32_t vec_count = vector_struct->count; int32_t attr_count = ((int32_t)(((double)vec_count) / 2)); ({ long long __if_result; if ((attr_count > 0)) { __if_result = ({ uint8_t* data = (uint8_t*)vector_struct->data; MlirNamedAttribute attrs_array[16]; int32_t idx = 0; ({ while ((idx < attr_count)) { ({ int32_t key_idx = (idx * 2); int32_t val_idx = ((idx * 2) + 1); int64_t key_offset = (((long long)key_idx) * 8); uint8_t* key_ptr_loc = (uint8_t*)((uint8_t*)(((int64_t)data) + key_offset)); Value** key_ptr_ptr = (Value**)((Value**)key_ptr_loc); Value* key_val = (Value*)(*key_ptr_ptr); int64_t val_offset = (((long long)val_idx) * 8); uint8_t* val_ptr_loc = (uint8_t*)((uint8_t*)(((int64_t)data) + val_offset)); Value** val_ptr_ptr = (Value**)((Value**)val_ptr_loc); Value* val_val = (Value*)(*val_ptr_ptr); ({ uint8_t* key_str = (uint8_t*)key_val->str_val; MlirNamedAttribute named_attr = g_mlir_builder.create_named_attribute_from_value(builder, key_str, val_val); (attrs_array[idx] = named_attr); idx = (idx + 1); }); }); } }); mlirOperationStateAddAttributes(state_ptr, ((int64_t)attr_count), (&attrs_array[0])); attr_count; }); } else { __if_result = 0; } __if_result; }); }); } else { __if_result = 0; } __if_result; }); });
}
static int32_t add_regions_to_state(MLIRBuilderContext* builder, MlirOperationState* state_ptr, Value* regions_val, ValueTracker* tracker) {
    return ({ ValueTag tag = regions_val->tag; ({ long long __if_result; if ((tag == ValueTag_Vector)) { __if_result = ({ uint8_t* vec_ptr = (uint8_t*)regions_val->vec_val; Vector* vector_struct = (Vector*)((Vector*)vec_ptr); int32_t count = vector_struct->count; ({ long long __if_result; if ((count > 0)) { __if_result = ({ uint8_t* data = (uint8_t*)vector_struct->data; MlirRegion regions_array[4]; int32_t idx = 0; int32_t out_count = 0; ({ while ((idx < count)) { ({ int64_t elem_offset = (((long long)idx) * 8); uint8_t* elem_ptr_loc = (uint8_t*)((uint8_t*)(((int64_t)data) + elem_offset)); Value** elem_ptr_ptr = (Value**)((Value**)elem_ptr_loc); Value* region_vec = (Value*)(*elem_ptr_ptr); ValueTag region_tag = region_vec->tag; ({ long long __if_result; if ((region_tag == ValueTag_Vector)) { __if_result = ({ MlirRegion mlir_region = g_mlir_builder.build_mlir_region(builder, region_vec, tracker); (regions_array[out_count] = mlir_region); out_count = (out_count + 1); 0; }); } else { __if_result = 0; } __if_result; }); idx = (idx + 1); }); } }); ({ long long __if_result; if ((out_count > 0)) { __if_result = ({ mlirOperationStateAddOwnedRegions(state_ptr, ((int64_t)out_count), (&regions_array[0])); 0; }); } else { __if_result = 0; } __if_result; }); out_count; }); } else { __if_result = 0; } __if_result; }); }); } else { __if_result = 0; } __if_result; }); });
}
static int32_t run_lowering_passes(MLIRBuilderContext* builder) {
    return ({ MlirContext ctx = builder->ctx; MlirModule mod = builder->mod; uint8_t* first_pipeline = (uint8_t*)"builtin.module(func.func(convert-scf-to-cf))"; uint8_t* second_pipeline = (uint8_t*)"builtin.module(func.func(convert-arith-to-llvm),convert-cf-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)"; ({ MlirPassManager pm1 = mlirPassManagerCreate(ctx); MlirOpPassManager opm1 = mlirPassManagerGetAsOpPassManager(pm1); MlirLogicalResult parse1 = mlirParsePassPipeline(opm1, mlirStringRefCreateFromCString(first_pipeline), ((void*)0), ((void*)0)); ({ long long __if_result; if ((mlirLogicalResultIsFailure(parse1) == 1)) { __if_result = ({ printf("ERROR: Failed to parse SCF lowering pipeline\n"); mlirPassManagerDestroy(pm1); 1; }); } else { __if_result = ({ MlirLogicalResult run1 = mlirPassManagerRunOnOp(pm1, mlirModuleGetOperation(mod)); mlirPassManagerDestroy(pm1); ({ long long __if_result; if ((mlirLogicalResultIsFailure(run1) == 1)) { __if_result = ({ printf("ERROR: SCF lowering pipeline failed\n"); 1; }); } else { __if_result = ({ MlirPassManager pm2 = mlirPassManagerCreate(ctx); MlirOpPassManager opm2 = mlirPassManagerGetAsOpPassManager(pm2); MlirLogicalResult parse2 = mlirParsePassPipeline(opm2, mlirStringRefCreateFromCString(second_pipeline), ((void*)0), ((void*)0)); ({ long long __if_result; if ((mlirLogicalResultIsFailure(parse2) == 1)) { __if_result = ({ printf("ERROR: Failed to parse LLVM lowering pipeline\n"); mlirPassManagerDestroy(pm2); 1; }); } else { __if_result = ({ MlirLogicalResult run2 = mlirPassManagerRunOnOp(pm2, mlirModuleGetOperation(mod)); mlirPassManagerDestroy(pm2); ({ long long __if_result; if ((mlirLogicalResultIsFailure(run2) == 1)) { __if_result = ({ printf("ERROR: LLVM lowering pipeline failed\n"); 1; }); } else { __if_result = 0; } __if_result; }); }); } __if_result; }); }); } __if_result; }); }); } __if_result; }); }); });
}
static int32_t jit_run_function_i32(MLIRBuilderContext* builder, uint8_t* fn_name, int32_t arg0) {
    return ({ MlirModule mod = builder->mod; MlirExecutionEngine engine = mlirExecutionEngineCreate(mod, 3, 0, ((void*)0), 0); ({ int32_t __if_result; if ((mlirExecutionEngineIsNull(engine) == 1)) { __if_result = ({ printf("ERROR: Failed to create execution engine\n"); ((int32_t)-1); }); } else { __if_result = ({ void* fn_ptr = (void*)mlirExecutionEngineLookup(engine, mlirStringRefCreateFromCString(fn_name)); ({ int32_t __if_result; if ((((int64_t)fn_ptr) == 0)) { __if_result = ({ printf("ERROR: Failed to lookup function %s\n", fn_name); mlirExecutionEngineDestroy(engine); ((int32_t)-1); }); } else { __if_result = ({ int32_t (*callable)(int32_t) = (int32_t (*)(int32_t))((int32_t (*)(int32_t))fn_ptr); int32_t result = (*callable)(arg0); printf("JIT result %s(%d) = %d\n", fn_name, arg0, result); mlirExecutionEngineDestroy(engine); result; }); } __if_result; }); }); } __if_result; }); });
}
static MlirOperation build_mlir_operation(MLIRBuilderContext* builder, OpNode* op_node, ValueTracker* tracker, MlirBlock parent_block) {
    return ({ MlirContext ctx = builder->ctx; MlirLocation loc = builder->loc; uint8_t* name_str = (uint8_t*)op_node->name; MlirStringRef name_ref = mlirStringRefCreateFromCString(name_str); MlirOperationState state = mlirOperationStateGet(name_ref, loc); MlirOperationState* state_ptr = (MlirOperationState*)({ MlirOperationState* __tmp_ptr = malloc(sizeof(MlirOperationState)); *__tmp_ptr = state; __tmp_ptr; }); printf("Building operation: %s\n", name_str); ({ Value* result_types = (Value*)op_node->result_types; g_mlir_builder.add_result_types_to_state(builder, state_ptr, result_types); ({ Value* operands = (Value*)op_node->operands; g_mlir_builder.add_operands_to_state(builder, state_ptr, operands, tracker); ({ Value* attributes = (Value*)op_node->attributes; g_mlir_builder.add_attributes_to_state(builder, state_ptr, attributes); ({ Value* regions = (Value*)op_node->regions; g_mlir_builder.add_regions_to_state(builder, state_ptr, regions, tracker); ({ MlirOperation op = mlirOperationCreate(state_ptr); mlirBlockAppendOwnedOperation(parent_block, op); ({ Value* result_types_val = (Value*)op_node->result_types; ValueTag result_tag = result_types_val->tag; ({ MlirOperation __if_result; if ((result_tag == ValueTag_Vector)) { __if_result = ({ uint8_t* vec_ptr = (uint8_t*)result_types_val->vec_val; Vector* vector_struct = (Vector*)((Vector*)vec_ptr); int32_t raw_count = vector_struct->count; ({ MlirOperation __if_result; if ((raw_count > 0)) { __if_result = ({ Value* first_elem = (Value*)g_mlir_builder.vector_element(result_types_val, 0); ValueTag first_tag = first_elem->tag; int32_t result_count = ((first_tag == ValueTag_Vector) ? raw_count : ((int32_t)(((double)raw_count) / 2))); int32_t idx = 0; ({ while ((idx < result_count)) { ({ MlirValue result_val = mlirOperationGetResult(op, ((int64_t)idx)); Value* name_val = (Value*)({ Value* __if_result; if ((first_tag == ValueTag_Vector)) { __if_result = ({ Value* elem = (Value*)g_mlir_builder.vector_element(result_types_val, idx); g_mlir_builder.vector_element(elem, 0); }); } else { __if_result = ({ int32_t name_idx = (idx * 2); g_mlir_builder.vector_element(result_types_val, name_idx); }); } __if_result; }); uint8_t* name_str = (uint8_t*)name_val->str_val; g_mlir_builder.value_tracker_register(tracker, name_str, result_val); idx = (idx + 1); }); } }); op; }); } else { __if_result = op; } __if_result; }); }); } else { __if_result = op; } __if_result; }); }); }); }); }); }); }); });
}
static int32_t compile_op_to_mlir(MLIRBuilderContext* builder, Value* op_form) {
    printf("=== Starting MLIR Compilation ===\n");
    return ({ OpNode* op_node = (OpNode*)g_mlir_ast.parse_op(op_form); ({ long long __if_result; if ((((int64_t)op_node) != 0)) { __if_result = ({ MlirModule mod = builder->mod; MlirOperation mod_op = mlirModuleGetOperation(mod); MlirRegion mod_region = mlirOperationGetRegion(mod_op, 0); MlirBlock mod_block = mlirRegionGetFirstBlock(mod_region); ValueTracker* tracker = (ValueTracker*)g_mlir_builder.value_tracker_create(); g_mlir_builder.build_mlir_operation(builder, op_node, tracker, mod_block); printf("=== Dumping MLIR Module ===\n"); mlirOperationDump(mod_op); printf("\n"); 0; }); } else { __if_result = ({ printf("ERROR: Failed to parse op\n"); 1; }); } __if_result; }); });
}
static uint8_t* read_file(uint8_t* filename) {
    return ({ uint8_t* file = (uint8_t*)fopen(filename, "r"); ({ uint8_t* __if_result; if ((((int64_t)file) == 0)) { __if_result = ({ printf("Error: Could not open file %s\n", filename); ((uint8_t*)0); }); } else { __if_result = ({ fseek(file, 0, 2); int32_t size = ftell(file); rewind(file); uint8_t* buffer = (uint8_t*)malloc((size + 1)); int32_t read_count = fread(buffer, 1, size, file); fclose(file); ({ int64_t null_pos = (((int64_t)buffer) + ((int64_t)size)); uint8_t* null_ptr = (uint8_t*)((uint8_t*)null_pos); (*null_ptr = ((uint8_t)0)); buffer; }); }); } __if_result; }); });
}
static Token* tokenize_file(uint8_t* content) {
    return ({ Tokenizer* tok = (Tokenizer*)g_tokenizer.make_tokenizer(content); int32_t max_tokens = 1000; int32_t token_size = 24; Token* tokens = (Token*)((Token*)malloc((max_tokens * token_size))); int32_t count = 0; ({ while ((count < max_tokens)) { ({ Token token = g_tokenizer.next_token(tok); TokenType token_type = token.type; int64_t token_offset = (((int64_t)count) * ((int64_t)token_size)); Token* token_ptr = (Token*)((Token*)(((int64_t)tokens) + token_offset)); token_ptr->type = token.type; token_ptr->text = token.text; token_ptr->length = token.length; count = (count + 1); ({ if ((token_type == TokenType_EOF)) { count = max_tokens; } else { count = count; } }); }); } }); tokens; });
}
static int32_t count_tokens(Token* tokens) {
    return ({ int32_t token_count = 0; int32_t found_eof = 0; ({ while (((token_count < 1000) && (found_eof == 0))) { ({ int64_t token_offset = (((long long)token_count) * 24); Token* token_ptr = (Token*)((Token*)(((int64_t)tokens) + token_offset)); TokenType token_type = token_ptr->type; token_count = (token_count + 1); ({ if ((token_type == TokenType_EOF)) { found_eof = 1; } else { found_eof = 0; } }); }); } }); token_count; });
}
static int32_t compile_file(MLIRBuilderContext* builder, uint8_t* filename) {
    printf("=== Compiling: %s ===\n", filename);
    return ({ uint8_t* content = (uint8_t*)g_mlir_builder.read_file(filename); ({ int32_t __if_result; if ((((int64_t)content) == 0)) { __if_result = 1; } else { __if_result = ({ Token* tokens = (Token*)g_mlir_builder.tokenize_file(content); int32_t token_count = g_mlir_builder.count_tokens(tokens); printf("Found %d tokens\n", token_count); Parser* p = (Parser*)g_parser.make_parser(tokens, token_count); ({ Value* result = (Value*)g_parser.parse_value(p); g_mlir_builder.compile_op_to_mlir(builder, result); }); }); } __if_result; }); });
}
static int32_t main_fn() {
    printf("=== MLIR Builder - File Compilation Test ===\n\n");
    return ({ MLIRBuilderContext* builder = (MLIRBuilderContext*)g_mlir_builder.mlir_builder_init(); printf("Builder initialized successfully\n\n"); ({ int32_t compile_res = g_mlir_builder.compile_file(builder, "tests/fib.lisp"); ({ int32_t __if_result; if ((compile_res == 0)) { __if_result = ({ printf("Running lowering pipeline...\n"); int32_t lower_res = g_mlir_builder.run_lowering_passes(builder); ({ int32_t __if_result; if ((lower_res == 0)) { __if_result = ({ printf("=== Lowered MLIR Module ===\n"); MlirModule mod = builder->mod; MlirOperation mod_op = mlirModuleGetOperation(mod); mlirOperationDump(mod_op); int32_t jit_res = g_mlir_builder.jit_run_function_i32(builder, "fib", 10); g_mlir_builder.mlir_builder_destroy(builder); printf("\nDone!\n"); ((jit_res == ((int32_t)-1)) ? ((int32_t)1) : 0); }); } else { __if_result = ({ g_mlir_builder.mlir_builder_destroy(builder); lower_res; }); } __if_result; }); }); } else { __if_result = ({ g_mlir_builder.mlir_builder_destroy(builder); compile_res; }); } __if_result; }); }); });
}
int main() {
    init_namespace_types(&g_types);
    init_namespace_mlir_ast(&g_mlir_ast);
    init_namespace_tokenizer(&g_tokenizer);
    init_namespace_parser(&g_parser);
    init_namespace_mlir_builder(&g_mlir_builder);
    // namespace mlir-builder
    // require [types :as types]
    // require [mlir-ast :as ast]
    // require [tokenizer :as tokenizer]
    // require [parser :as parser]
    g_mlir_builder.main_fn();
    return 0;
}
