#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

// Local type definitions
typedef struct Cons Cons;
typedef struct Token Token;
typedef struct Vector Vector;
typedef struct Value Value;
struct Cons {
    uint8_t* car;
    uint8_t* cdr;
};

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

struct Token {
    TokenType type;
    uint8_t* text;
    int32_t length;
};

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

#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "ctype.h"

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

Namespace_types g_types;

static Value make_nil();
static Value make_number(int64_t);
static Value make_symbol(uint8_t*);
static Value make_string(uint8_t*);
static Value* make_cons(Value*, Value*);
static Value* make_empty_vector();
static Value* make_vector_with_capacity(int32_t, int32_t);
static Value* make_empty_map();
static Value* car(Value*);
static Value* cdr(Value*);
static uint8_t* copy_string(uint8_t*, int32_t);
static int32_t is_number_token(Token);
static int32_t vector_set(Value*, int32_t, Value*);

void init_namespace_types(Namespace_types* ns) {
    ns->make_nil = &make_nil;
    ns->make_number = &make_number;
    ns->make_symbol = &make_symbol;
    ns->make_string = &make_string;
    ns->make_cons = &make_cons;
    ns->make_empty_vector = &make_empty_vector;
    ns->make_vector_with_capacity = &make_vector_with_capacity;
    ns->make_empty_map = &make_empty_map;
    ns->car = &car;
    ns->cdr = &cdr;
    ns->copy_string = &copy_string;
    ns->is_number_token = &is_number_token;
    ns->vector_set = &vector_set;
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
static Value* make_cons(Value* car_val, Value* cdr_val) {
    return ({ Cons* cons_cell = (Cons*)((Cons*)malloc(16)); cons_cell->car = ((uint8_t*)car_val); cons_cell->cdr = ((uint8_t*)cdr_val); ({ Value* val = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_nil(); __tmp_ptr; }); val->tag = ValueTag_List; val->cons_val = ((uint8_t*)cons_cell); val; }); });
}
static Value* make_empty_vector() {
    return ({ Vector* vec = (Vector*)((Vector*)malloc(16)); uint8_t* data = (uint8_t*)malloc(8); vec->data = data; vec->count = 0; vec->capacity = 1; ({ Value* val = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_nil(); __tmp_ptr; }); val->tag = ValueTag_Vector; val->vec_val = ((uint8_t*)vec); val; }); });
}
static Value* make_vector_with_capacity(int32_t count, int32_t capacity) {
    return ({ Vector* vec = (Vector*)((Vector*)malloc(16)); uint8_t* data = (uint8_t*)malloc((capacity * 8)); vec->data = data; vec->count = count; vec->capacity = capacity; ({ Value* val = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_nil(); __tmp_ptr; }); val->tag = ValueTag_Vector; val->vec_val = ((uint8_t*)vec); val; }); });
}
static Value* make_empty_map() {
    return ({ Vector* vec = (Vector*)((Vector*)malloc(16)); uint8_t* data = (uint8_t*)malloc(8); vec->data = data; vec->count = 0; vec->capacity = 1; ({ Value* val = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_nil(); __tmp_ptr; }); val->tag = ValueTag_Map; val->vec_val = ((uint8_t*)vec); val; }); });
}
static Value* car(Value* v) {
    return ({ uint8_t* cons_ptr = (uint8_t*)v->cons_val; Cons* cons_cell = (Cons*)((Cons*)cons_ptr); ((Value*)cons_cell->car); });
}
static Value* cdr(Value* v) {
    return ({ uint8_t* cons_ptr = (uint8_t*)v->cons_val; Cons* cons_cell = (Cons*)((Cons*)cons_ptr); ((Value*)cons_cell->cdr); });
}
static uint8_t* copy_string(uint8_t* src, int32_t len) {
    return ({ uint8_t* dest = (uint8_t*)malloc((len + 1)); strncpy(dest, src, len); ({ uint8_t* null_pos = (uint8_t*)((uint8_t*)(((int64_t)dest) + ((int64_t)len))); (*((uint8_t*)null_pos) = ((uint8_t)0)); }); dest; });
}
static int32_t is_number_token(Token tok) {
    return ((tok.length == 0) ? 0 : ({ uint8_t first_char = (*tok.text); int32_t is_digit = isdigit(((int32_t)first_char)); ((is_digit != 0) ? 1 : (((first_char == ((uint8_t)45)) && (tok.length > 1)) ? ({ uint8_t* second_char_ptr = (uint8_t*)((uint8_t*)(((long long)tok.text) + 1)); uint8_t second_char = (*second_char_ptr); int32_t is_digit2 = isdigit(((int32_t)second_char)); ((is_digit2 != 0) ? 1 : 0); }) : 0)); }));
}
static int32_t vector_set(Value* vec_val, int32_t index, Value* elem) {
    return ({ uint8_t* vec_ptr = (uint8_t*)vec_val->vec_val; Vector* vec = (Vector*)((Vector*)vec_ptr); uint8_t* data = (uint8_t*)vec->data; uint8_t* elem_loc = (uint8_t*)((uint8_t*)(((long long)data) + (index * 8))); Value** elem_ptr = (Value**)((Value**)elem_loc); (*elem_ptr = elem); 0; });
}
void lisp_main(void) {
    init_namespace_types(&g_types);
}
