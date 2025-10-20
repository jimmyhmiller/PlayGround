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
typedef struct MLIRBuilderContext MLIRBuilderContext;
typedef struct ValueTracker ValueTracker;
struct MLIRBuilderContext {
    MlirContext ctx;
    MlirLocation loc;
    MlirModule mod;
    MlirType i32Type;
    MlirType i64Type;
};

struct ValueTracker {
    MlirValue values[256];
    int32_t count;
};

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

typedef struct {
    void (*die)(uint8_t*);
    MLIRBuilderContext* (*mlir_builder_init)();
    int32_t (*mlir_builder_destroy)(MLIRBuilderContext*);
    int32_t MAX_VALUES;
    ValueTracker* (*value_tracker_create)();
    int32_t (*value_tracker_register)(ValueTracker*, MlirValue);
    MlirValue (*value_tracker_lookup)(ValueTracker*, int32_t);
    MlirType (*parse_type_string_helper)(MLIRBuilderContext*, uint8_t*);
    MlirType (*parse_type_string)(MLIRBuilderContext*, uint8_t*);
    int64_t (*parse_int_attr_value)(uint8_t*);
    MlirAttribute (*create_attribute)(MLIRBuilderContext*, uint8_t*, uint8_t*);
    MlirNamedAttribute (*create_named_attribute)(MLIRBuilderContext*, uint8_t*, uint8_t*);
    MlirRegion (*build_mlir_region)(MLIRBuilderContext*, Value*, ValueTracker*);
    MlirBlock (*build_mlir_block)(MLIRBuilderContext*, Value*, ValueTracker*);
    int32_t (*add_result_types_to_state)(MLIRBuilderContext*, MlirOperationState*, Value*);
    int32_t (*add_operands_to_state)(MLIRBuilderContext*, MlirOperationState*, Value*, ValueTracker*);
    int32_t (*add_attributes_to_state)(MLIRBuilderContext*, MlirOperationState*, Value*);
    int32_t (*add_regions_to_state)(MLIRBuilderContext*, MlirOperationState*, Value*, ValueTracker*);
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
static MLIRBuilderContext* mlir_builder_init();
static int32_t mlir_builder_destroy(MLIRBuilderContext*);
static ValueTracker* value_tracker_create();
static int32_t value_tracker_register(ValueTracker*, MlirValue);
static MlirValue value_tracker_lookup(ValueTracker*, int32_t);
static MlirType parse_type_string_helper(MLIRBuilderContext*, uint8_t*);
static MlirType parse_type_string(MLIRBuilderContext*, uint8_t*);
static int64_t parse_int_attr_value(uint8_t*);
static MlirAttribute create_attribute(MLIRBuilderContext*, uint8_t*, uint8_t*);
static MlirNamedAttribute create_named_attribute(MLIRBuilderContext*, uint8_t*, uint8_t*);
static MlirRegion build_mlir_region(MLIRBuilderContext*, Value*, ValueTracker*);
static MlirBlock build_mlir_block(MLIRBuilderContext*, Value*, ValueTracker*);
static int32_t add_result_types_to_state(MLIRBuilderContext*, MlirOperationState*, Value*);
static int32_t add_operands_to_state(MLIRBuilderContext*, MlirOperationState*, Value*, ValueTracker*);
static int32_t add_attributes_to_state(MLIRBuilderContext*, MlirOperationState*, Value*);
static int32_t add_regions_to_state(MLIRBuilderContext*, MlirOperationState*, Value*, ValueTracker*);
static MlirOperation build_mlir_operation(MLIRBuilderContext*, OpNode*, ValueTracker*, MlirBlock);
static int32_t compile_op_to_mlir(MLIRBuilderContext*, Value*);
static uint8_t* read_file(uint8_t*);
static Token* tokenize_file(uint8_t*);
static int32_t count_tokens(Token*);
static int32_t compile_file(MLIRBuilderContext*, uint8_t*);
static int32_t main_fn();

void init_namespace_mlir_builder(Namespace_mlir_builder* ns) {
    ns->die = &die;
    ns->mlir_builder_init = &mlir_builder_init;
    ns->mlir_builder_destroy = &mlir_builder_destroy;
    ns->MAX_VALUES = 256;
    ns->value_tracker_create = &value_tracker_create;
    ns->value_tracker_register = &value_tracker_register;
    ns->value_tracker_lookup = &value_tracker_lookup;
    ns->parse_type_string_helper = &parse_type_string_helper;
    ns->parse_type_string = &parse_type_string;
    ns->parse_int_attr_value = &parse_int_attr_value;
    ns->create_attribute = &create_attribute;
    ns->create_named_attribute = &create_named_attribute;
    ns->build_mlir_region = &build_mlir_region;
    ns->build_mlir_block = &build_mlir_block;
    ns->add_result_types_to_state = &add_result_types_to_state;
    ns->add_operands_to_state = &add_operands_to_state;
    ns->add_attributes_to_state = &add_attributes_to_state;
    ns->add_regions_to_state = &add_regions_to_state;
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
static MLIRBuilderContext* mlir_builder_init() {
    return ({ MlirContext ctx = mlirContextCreate(); MlirDialectRegistry registry = mlirDialectRegistryCreate(); mlirRegisterAllDialects(registry); mlirContextAppendDialectRegistry(ctx, registry); mlirDialectRegistryDestroy(registry); mlirContextLoadAllAvailableDialects(ctx); mlirRegisterAllPasses(); mlirRegisterAllLLVMTranslations(ctx); ({ MlirLocation loc = mlirLocationUnknownGet(ctx); MlirModule mod = mlirModuleCreateEmpty(loc); MlirType i32Type = mlirIntegerTypeGet(ctx, 32); MlirType i64Type = mlirIntegerTypeGet(ctx, 64); MLIRBuilderContext* builder = (MLIRBuilderContext*)((MLIRBuilderContext*)malloc(40)); builder->ctx = ctx; builder->loc = loc; builder->mod = mod; builder->i32Type = i32Type; builder->i64Type = i64Type; builder; }); });
}
static int32_t mlir_builder_destroy(MLIRBuilderContext* builder) {
    return ({ MlirContext ctx = builder->ctx; mlirContextDestroy(ctx); 0; });
}
static ValueTracker* value_tracker_create() {
    return ({ ValueTracker* tracker = (ValueTracker*)((ValueTracker*)malloc(2056)); tracker->count = 0; tracker; });
}
static int32_t value_tracker_register(ValueTracker* tracker, MlirValue value) {
    return ({ int32_t count = tracker->count; ({ MlirValue* values_ptr = (MlirValue*)(&tracker->values[0]); ({ MlirValue* target_ptr = (MlirValue*)((MlirValue*)(((long long)values_ptr) + (((long long)count) * 8))); (*target_ptr = value); tracker->count = (count + 1); count; }); }); });
}
static MlirValue value_tracker_lookup(ValueTracker* tracker, int32_t idx) {
    return ({ MlirValue* values_ptr = (MlirValue*)(&tracker->values[0]); MlirValue* target_ptr = (MlirValue*)((MlirValue*)(((long long)values_ptr) + (((long long)idx) * 8))); (*target_ptr); });
}
static MlirType parse_type_string_helper(MLIRBuilderContext* builder, uint8_t* type_str) {
    return ({ int32_t cmp_i64 = strcmp(type_str, "i64"); ((cmp_i64 == 0) ? builder->i64Type : ({ printf("Unknown type: %s, defaulting to i32\n", type_str); builder->i32Type; })); });
}
static MlirType parse_type_string(MLIRBuilderContext* builder, uint8_t* type_str) {
    return ({ int32_t cmp_i32 = strcmp(type_str, "i32"); ((cmp_i32 == 0) ? builder->i32Type : g_mlir_builder.parse_type_string_helper(builder, type_str)); });
}
static int64_t parse_int_attr_value(uint8_t* value_str) {
    return ((int64_t)atoi(value_str));
}
static MlirAttribute create_attribute(MLIRBuilderContext* builder, uint8_t* key, uint8_t* value) {
    return ({ MlirContext ctx = builder->ctx; mlirStringAttrGet(ctx, mlirStringRefCreateFromCString(value)); });
}
static MlirNamedAttribute create_named_attribute(MLIRBuilderContext* builder, uint8_t* key, uint8_t* value) {
    return ({ MlirContext ctx = builder->ctx; MlirIdentifier name_id = mlirIdentifierGet(ctx, mlirStringRefCreateFromCString(key)); MlirAttribute attr = g_mlir_builder.create_attribute(builder, key, value); mlirNamedAttributeGet(name_id, attr); });
}
static MlirRegion build_mlir_region(MLIRBuilderContext* builder, Value* region_vec, ValueTracker* tracker) {
    return ({ MlirRegion region = mlirRegionCreate(); ValueTag tag = region_vec->tag; ((tag == ValueTag_Vector) ? ({ uint8_t* vec_ptr = (uint8_t*)region_vec->vec_val; Vector* vector_struct = (Vector*)((Vector*)vec_ptr); int32_t count = vector_struct->count; uint8_t* data = (uint8_t*)vector_struct->data; int32_t idx = 0; ({ while ((idx < count)) { ({ int64_t elem_offset = (((long long)idx) * 8); uint8_t* elem_ptr_loc = (uint8_t*)((uint8_t*)(((int64_t)data) + elem_offset)); Value** elem_ptr_ptr = (Value**)((Value**)elem_ptr_loc); Value* block_form = (Value*)(*elem_ptr_ptr); MlirBlock mlir_block = g_mlir_builder.build_mlir_block(builder, block_form, tracker); mlirRegionAppendOwnedBlock(region, mlir_block); idx = (idx + 1); }); } }); region; }) : region); });
}
static MlirBlock build_mlir_block(MLIRBuilderContext* builder, Value* block_form, ValueTracker* tracker) {
    return ({ BlockNode* block_node = (BlockNode*)g_mlir_ast.parse_block(block_form); ((((int64_t)block_node) != 0) ? ({ MlirLocation loc = builder->loc; MlirBlock block = mlirBlockCreate(0, ((MlirType*)0), ((MlirLocation*)0)); Value* operations = (Value*)block_node->operations; ValueTag ops_tag = operations->tag; ((ops_tag == ValueTag_Vector) ? ({ uint8_t* vec_ptr = (uint8_t*)operations->vec_val; Vector* vector_struct = (Vector*)((Vector*)vec_ptr); int32_t count = vector_struct->count; uint8_t* data = (uint8_t*)vector_struct->data; int32_t idx = 0; ({ while ((idx < count)) { ({ int64_t elem_offset = (((long long)idx) * 8); uint8_t* elem_ptr_loc = (uint8_t*)((uint8_t*)(((int64_t)data) + elem_offset)); Value** elem_ptr_ptr = (Value**)((Value**)elem_ptr_loc); Value* op_form = (Value*)(*elem_ptr_ptr); OpNode* op_node = (OpNode*)g_mlir_ast.parse_op(op_form); ({ if ((((int64_t)op_node) != 0)) { ({ MlirOperation mlir_op = g_mlir_builder.build_mlir_operation(builder, op_node, tracker, block); idx = (idx + 1); }); } else { idx = (idx + 1); } }); }); } }); block; }) : block); }) : mlirBlockCreate(0, ((MlirType*)0), ((MlirLocation*)0))); });
}
static int32_t add_result_types_to_state(MLIRBuilderContext* builder, MlirOperationState* state_ptr, Value* result_types_val) {
    return ({ ValueTag tag = result_types_val->tag; ((tag == ValueTag_Vector) ? ({ uint8_t* vec_ptr = (uint8_t*)result_types_val->vec_val; Vector* vector_struct = (Vector*)((Vector*)vec_ptr); int32_t count = vector_struct->count; ((count > 0) ? ({ uint8_t* data = (uint8_t*)vector_struct->data; MlirType result_types_array[8]; int32_t idx = 0; ({ while ((idx < count)) { ({ int64_t elem_offset = (((long long)idx) * 8); uint8_t* elem_ptr_loc = (uint8_t*)((uint8_t*)(((int64_t)data) + elem_offset)); Value** elem_ptr_ptr = (Value**)((Value**)elem_ptr_loc); Value* elem = (Value*)(*elem_ptr_ptr); ValueTag elem_tag = elem->tag; ({ if ((elem_tag == ValueTag_String)) { ({ uint8_t* type_str = (uint8_t*)elem->str_val; MlirType mlir_type = g_mlir_builder.parse_type_string(builder, type_str); (result_types_array[idx] = mlir_type); idx = (idx + 1); }); } else { idx = (idx + 1); } }); }); } }); mlirOperationStateAddResults(state_ptr, ((int64_t)count), (&result_types_array[0])); count; }) : 0); }) : 0); });
}
static int32_t add_operands_to_state(MLIRBuilderContext* builder, MlirOperationState* state_ptr, Value* operands_val, ValueTracker* tracker) {
    return ({ ValueTag tag = operands_val->tag; ((tag == ValueTag_Vector) ? ({ uint8_t* vec_ptr = (uint8_t*)operands_val->vec_val; Vector* vector_struct = (Vector*)((Vector*)vec_ptr); int32_t count = vector_struct->count; ((count > 0) ? ({ uint8_t* data = (uint8_t*)vector_struct->data; MlirValue operands_array[8]; int32_t idx = 0; ({ while ((idx < count)) { ({ int64_t elem_offset = (((long long)idx) * 8); uint8_t* elem_ptr_loc = (uint8_t*)((uint8_t*)(((int64_t)data) + elem_offset)); Value** elem_ptr_ptr = (Value**)((Value**)elem_ptr_loc); Value* elem = (Value*)(*elem_ptr_ptr); ValueTag elem_tag = elem->tag; ({ if ((elem_tag == ValueTag_String)) { ({ uint8_t* operand_str = (uint8_t*)elem->str_val; int32_t operand_idx = atoi(operand_str); MlirValue operand_val = g_mlir_builder.value_tracker_lookup(tracker, operand_idx); (operands_array[idx] = operand_val); idx = (idx + 1); }); } else { idx = (idx + 1); } }); }); } }); mlirOperationStateAddOperands(state_ptr, ((int64_t)count), (&operands_array[0])); count; }) : 0); }) : 0); });
}
static int32_t add_attributes_to_state(MLIRBuilderContext* builder, MlirOperationState* state_ptr, Value* attributes_val) {
    return ({ ValueTag tag = attributes_val->tag; ((tag == ValueTag_Map) ? ({ uint8_t* vec_ptr = (uint8_t*)attributes_val->vec_val; Vector* vector_struct = (Vector*)((Vector*)vec_ptr); int32_t vec_count = vector_struct->count; int32_t attr_count = ((int32_t)(((double)vec_count) / 2)); ((attr_count > 0) ? ({ uint8_t* data = (uint8_t*)vector_struct->data; MlirNamedAttribute attrs_array[16]; int32_t idx = 0; ({ while ((idx < attr_count)) { ({ int32_t key_idx = (idx * 2); int32_t val_idx = ((idx * 2) + 1); int64_t key_offset = (((long long)key_idx) * 8); uint8_t* key_ptr_loc = (uint8_t*)((uint8_t*)(((int64_t)data) + key_offset)); Value** key_ptr_ptr = (Value**)((Value**)key_ptr_loc); Value* key_val = (Value*)(*key_ptr_ptr); int64_t val_offset = (((long long)val_idx) * 8); uint8_t* val_ptr_loc = (uint8_t*)((uint8_t*)(((int64_t)data) + val_offset)); Value** val_ptr_ptr = (Value**)((Value**)val_ptr_loc); Value* val_val = (Value*)(*val_ptr_ptr); ({ uint8_t* key_str = (uint8_t*)key_val->str_val; uint8_t* val_str = (uint8_t*)val_val->str_val; MlirNamedAttribute named_attr = g_mlir_builder.create_named_attribute(builder, key_str, val_str); (attrs_array[idx] = named_attr); idx = (idx + 1); }); }); } }); mlirOperationStateAddAttributes(state_ptr, ((int64_t)attr_count), (&attrs_array[0])); attr_count; }) : 0); }) : 0); });
}
static int32_t add_regions_to_state(MLIRBuilderContext* builder, MlirOperationState* state_ptr, Value* regions_val, ValueTracker* tracker) {
    return ({ ValueTag tag = regions_val->tag; ((tag == ValueTag_Vector) ? ({ uint8_t* vec_ptr = (uint8_t*)regions_val->vec_val; Vector* vector_struct = (Vector*)((Vector*)vec_ptr); int32_t count = vector_struct->count; ((count > 0) ? ({ uint8_t* data = (uint8_t*)vector_struct->data; MlirRegion regions_array[4]; int32_t idx = 0; ({ while ((idx < count)) { ({ int64_t elem_offset = (((long long)idx) * 8); uint8_t* elem_ptr_loc = (uint8_t*)((uint8_t*)(((int64_t)data) + elem_offset)); Value** elem_ptr_ptr = (Value**)((Value**)elem_ptr_loc); Value* region_vec = (Value*)(*elem_ptr_ptr); MlirRegion mlir_region = g_mlir_builder.build_mlir_region(builder, region_vec, tracker); (regions_array[idx] = mlir_region); idx = (idx + 1); }); } }); mlirOperationStateAddOwnedRegions(state_ptr, ((int64_t)count), (&regions_array[0])); count; }) : 0); }) : 0); });
}
static MlirOperation build_mlir_operation(MLIRBuilderContext* builder, OpNode* op_node, ValueTracker* tracker, MlirBlock parent_block) {
    return ({ MlirContext ctx = builder->ctx; MlirLocation loc = builder->loc; uint8_t* name_str = (uint8_t*)op_node->name; MlirStringRef name_ref = mlirStringRefCreateFromCString(name_str); MlirOperationState state = mlirOperationStateGet(name_ref, loc); MlirOperationState* state_ptr = (MlirOperationState*)({ MlirOperationState* __tmp_ptr = malloc(sizeof(MlirOperationState)); __tmp_ptr; }); printf("Building operation: %s\n", name_str); ({ Value* result_types = (Value*)op_node->result_types; g_mlir_builder.add_result_types_to_state(builder, state_ptr, result_types); ({ Value* operands = (Value*)op_node->operands; g_mlir_builder.add_operands_to_state(builder, state_ptr, operands, tracker); ({ Value* attributes = (Value*)op_node->attributes; g_mlir_builder.add_attributes_to_state(builder, state_ptr, attributes); ({ Value* regions = (Value*)op_node->regions; g_mlir_builder.add_regions_to_state(builder, state_ptr, regions, tracker); ({ MlirOperation op = mlirOperationCreate(state_ptr); mlirBlockAppendOwnedOperation(parent_block, op); ({ Value* result_types_val = (Value*)op_node->result_types; ValueTag result_tag = result_types_val->tag; ((result_tag == ValueTag_Vector) ? ({ uint8_t* vec_ptr = (uint8_t*)result_types_val->vec_val; Vector* vector_struct = (Vector*)((Vector*)vec_ptr); int32_t result_count = vector_struct->count; int32_t idx = 0; ({ while ((idx < result_count)) { ({ MlirValue result_val = mlirOperationGetResult(op, ((int64_t)idx)); g_mlir_builder.value_tracker_register(tracker, result_val); idx = (idx + 1); }); } }); op; }) : op); }); }); }); }); }); }); });
}
static int32_t compile_op_to_mlir(MLIRBuilderContext* builder, Value* op_form) {
    printf("=== Starting MLIR Compilation ===\n");
    return ({ OpNode* op_node = (OpNode*)g_mlir_ast.parse_op(op_form); ((((int64_t)op_node) != 0) ? ({ MlirModule mod = builder->mod; MlirOperation mod_op = mlirModuleGetOperation(mod); MlirRegion mod_region = mlirOperationGetRegion(mod_op, 0); MlirBlock mod_block = mlirRegionGetFirstBlock(mod_region); ValueTracker* tracker = (ValueTracker*)g_mlir_builder.value_tracker_create(); g_mlir_builder.build_mlir_operation(builder, op_node, tracker, mod_block); printf("=== Dumping MLIR Module ===\n"); mlirOperationDump(mod_op); printf("\n"); 0; }) : ({ printf("ERROR: Failed to parse op\n"); 1; })); });
}
static uint8_t* read_file(uint8_t* filename) {
    return ({ uint8_t* file = (uint8_t*)fopen(filename, "r"); ((((int64_t)file) == 0) ? ({ printf("Error: Could not open file %s\n", filename); ((uint8_t*)0); }) : ({ fseek(file, 0, 2); int32_t size = ftell(file); rewind(file); uint8_t* buffer = (uint8_t*)malloc((size + 1)); int32_t read_count = fread(buffer, 1, size, file); fclose(file); ({ int64_t null_pos = (((int64_t)buffer) + ((int64_t)size)); uint8_t* null_ptr = (uint8_t*)((uint8_t*)null_pos); (*null_ptr = ((uint8_t)0)); buffer; }); })); });
}
static Token* tokenize_file(uint8_t* content) {
    return ({ Tokenizer* tok = (Tokenizer*)g_tokenizer.make_tokenizer(content); int32_t max_tokens = 1000; int32_t token_size = 24; Token* tokens = (Token*)((Token*)malloc((max_tokens * token_size))); int32_t count = 0; ({ while ((count < max_tokens)) { ({ Token token = g_tokenizer.next_token(tok); TokenType token_type = token.type; int64_t token_offset = (((int64_t)count) * ((int64_t)token_size)); Token* token_ptr = (Token*)((Token*)(((int64_t)tokens) + token_offset)); token_ptr->type = token.type; token_ptr->text = token.text; token_ptr->length = token.length; count = (count + 1); ({ if ((token_type == TokenType_EOF)) { count = max_tokens; } else { count = count; } }); }); } }); tokens; });
}
static int32_t count_tokens(Token* tokens) {
    return ({ int32_t token_count = 0; int32_t found_eof = 0; ({ while (((token_count < 1000) && (found_eof == 0))) { ({ int64_t token_offset = (((long long)token_count) * 24); Token* token_ptr = (Token*)((Token*)(((int64_t)tokens) + token_offset)); TokenType token_type = token_ptr->type; token_count = (token_count + 1); ({ if ((token_type == TokenType_EOF)) { found_eof = 1; } else { found_eof = 0; } }); }); } }); token_count; });
}
static int32_t compile_file(MLIRBuilderContext* builder, uint8_t* filename) {
    printf("=== Compiling: %s ===\n", filename);
    return ({ uint8_t* content = (uint8_t*)g_mlir_builder.read_file(filename); ((((int64_t)content) == 0) ? 1 : ({ Token* tokens = (Token*)g_mlir_builder.tokenize_file(content); int32_t token_count = g_mlir_builder.count_tokens(tokens); printf("Found %d tokens\n", token_count); Parser* p = (Parser*)g_parser.make_parser(tokens, token_count); ({ Value* result = (Value*)g_parser.parse_value(p); g_mlir_builder.compile_op_to_mlir(builder, result); }); })); });
}
static int32_t main_fn() {
    printf("=== MLIR Builder - File Compilation Test ===\n\n");
    return ({ MLIRBuilderContext* builder = (MLIRBuilderContext*)g_mlir_builder.mlir_builder_init(); printf("Builder initialized successfully\n\n"); ({ int32_t result = g_mlir_builder.compile_file(builder, "tests/simple.lisp"); g_mlir_builder.mlir_builder_destroy(builder); printf("\nDone!\n"); result; }); });
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
