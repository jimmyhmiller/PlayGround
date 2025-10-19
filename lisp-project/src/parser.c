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

Namespace_parser g_parser;

static Parser* make_parser(Token*, int32_t);
static Token peek_token(Parser*);
static int32_t advance_parser(Parser*);
static Value* parse_list(Parser*);
static int32_t parse_vector_elements(Parser*, Value*, int32_t);
static Value* parse_vector(Parser*);
static int32_t parse_map_elements(Parser*, Value*, int32_t);
static Value* parse_map(Parser*);
static Value* parse_value(Parser*);
static int32_t print_vector_contents(Value*, int32_t);
static int32_t print_map_contents(Value*, int32_t);
static int32_t print_value_ptr(Value*);
static int32_t print_list_contents(Value*);
static int32_t main_fn();

void init_namespace_parser(Namespace_parser* ns) {
    init_namespace_types(&g_types);
    ns->make_parser = &make_parser;
    ns->peek_token = &peek_token;
    ns->advance_parser = &advance_parser;
    ns->parse_list = &parse_list;
    ns->parse_vector_elements = &parse_vector_elements;
    ns->parse_vector = &parse_vector;
    ns->parse_map_elements = &parse_map_elements;
    ns->parse_map = &parse_map;
    ns->parse_value = &parse_value;
    ns->print_vector_contents = &print_vector_contents;
    ns->print_map_contents = &print_map_contents;
    ns->print_value_ptr = &print_value_ptr;
    ns->print_list_contents = &print_list_contents;
    ns->main_fn = &main_fn;
}

static Parser* make_parser(Token* tokens, int32_t count) {
    return ({ Parser* p = (Parser*)({ Parser* __tmp_ptr = malloc(sizeof(Parser)); *__tmp_ptr = (Parser){NULL, 0, 0}; __tmp_ptr; }); p->tokens = tokens; p->position = 0; p->count = count; p; });
}
static Token peek_token(Parser* p) {
    return ({ int32_t pos = p->position; int32_t count = p->count; ((pos >= count) ? (Token){TokenType_EOF, NULL, 0} : ({ Token* tokens = (Token*)p->tokens; ({ Token* token_ptr = (Token*)((Token*)(((long long)tokens) + (((long long)pos) * 24))); TokenType ttype = token_ptr->type; uint8_t* ttext = (uint8_t*)token_ptr->text; int32_t tlen = token_ptr->length; (Token){ttype, ttext, tlen}; }); })); });
}
static int32_t advance_parser(Parser* p) {
    return ({ int32_t pos = p->position; p->position = (pos + 1); 0; });
}
static Value* parse_list(Parser* p) {
    return ({ Token tok = g_parser.peek_token(p); ((tok.type == TokenType_RightParen) ? ({ g_parser.advance_parser(p); ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_nil(); __tmp_ptr; }); }) : ({ Value* first = (Value*)g_parser.parse_value(p); Value* rest = (Value*)g_parser.parse_list(p); g_types.make_cons(first, rest); })); });
}
static int32_t parse_vector_elements(Parser* p, Value* vec, int32_t index) {
    return ({ Token tok = g_parser.peek_token(p); ((tok.type == TokenType_RightBracket) ? ({ g_parser.advance_parser(p); index; }) : ({ Value* elem = (Value*)g_parser.parse_value(p); g_types.vector_set(vec, index, elem); g_parser.parse_vector_elements(p, vec, (index + 1)); })); });
}
static Value* parse_vector(Parser* p) {
    return ({ Value* vec = (Value*)g_types.make_vector_with_capacity(16, 16); int32_t final_count = g_parser.parse_vector_elements(p, vec, 0); ({ uint8_t* vec_ptr = (uint8_t*)vec->vec_val; Vector* vec_struct = (Vector*)((Vector*)vec_ptr); vec_struct->count = final_count; vec; }); });
}
static int32_t parse_map_elements(Parser* p, Value* map_vec, int32_t index) {
    return ({ Token tok = g_parser.peek_token(p); ((tok.type == TokenType_RightBrace) ? ({ g_parser.advance_parser(p); index; }) : ({ Value* elem = (Value*)g_parser.parse_value(p); g_types.vector_set(map_vec, index, elem); g_parser.parse_map_elements(p, map_vec, (index + 1)); })); });
}
static Value* parse_map(Parser* p) {
    return ({ Value* map_vec = (Value*)g_types.make_vector_with_capacity(32, 32); int32_t final_count = g_parser.parse_map_elements(p, map_vec, 0); ({ uint8_t* vec_ptr = (uint8_t*)map_vec->vec_val; Vector* vec_struct = (Vector*)((Vector*)vec_ptr); vec_struct->count = final_count; ({ Value* map_val = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_nil(); __tmp_ptr; }); map_val->tag = ValueTag_Map; map_val->vec_val = vec_ptr; map_val; }); }); });
}
static Value* parse_value(Parser* p) {
    return ({ Token tok = g_parser.peek_token(p); ((tok.type == TokenType_LeftParen) ? ({ g_parser.advance_parser(p); g_parser.parse_list(p); }) : ((tok.type == TokenType_LeftBracket) ? ({ g_parser.advance_parser(p); g_parser.parse_vector(p); }) : ((tok.type == TokenType_LeftBrace) ? ({ g_parser.advance_parser(p); g_parser.parse_map(p); }) : ((tok.type == TokenType_Symbol) ? ({ g_parser.advance_parser(p); ((g_types.is_number_token(tok) != 0) ? ({ uint8_t* str = (uint8_t*)g_types.copy_string(tok.text, tok.length); int64_t num = atoll(str); ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_number(num); __tmp_ptr; }); }) : ({ uint8_t* str = (uint8_t*)g_types.copy_string(tok.text, tok.length); ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_symbol(str); __tmp_ptr; }); })); }) : ((tok.type == TokenType_String) ? ({ g_parser.advance_parser(p); ({ uint8_t* str_start = (uint8_t*)((uint8_t*)(((long long)tok.text) + 1)); int32_t str_len = (tok.length - 2); uint8_t* str = (uint8_t*)g_types.copy_string(str_start, str_len); ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = (Value){ValueTag_String, 0, str, NULL, NULL}; __tmp_ptr; }); }); }) : ((tok.type == TokenType_Keyword) ? ({ g_parser.advance_parser(p); ({ uint8_t* str = (uint8_t*)g_types.copy_string(tok.text, tok.length); ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = (Value){ValueTag_Keyword, 0, str, NULL, NULL}; __tmp_ptr; }); }); }) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_types.make_nil(); __tmp_ptr; }))))))); });
}
static int32_t print_vector_contents(Value* vec_val, int32_t index) {
    return ({ uint8_t* vec_ptr = (uint8_t*)vec_val->vec_val; Vector* vec = (Vector*)((Vector*)vec_ptr); int32_t count = vec->count; ((index >= count) ? printf("]") : ({ uint8_t* data = (uint8_t*)vec->data; uint8_t* elem_loc = (uint8_t*)((uint8_t*)(((long long)data) + (index * 8))); Value** elem_ptr_ptr = (Value**)((Value**)elem_loc); Value* elem = (Value*)(*elem_ptr_ptr); g_parser.print_value_ptr(elem); (((index + 1) < count) ? ({ printf(" "); g_parser.print_vector_contents(vec_val, (index + 1)); }) : printf("]")); })); });
}
static int32_t print_map_contents(Value* map_val, int32_t index) {
    return ({ uint8_t* vec_ptr = (uint8_t*)map_val->vec_val; Vector* vec = (Vector*)((Vector*)vec_ptr); int32_t count = vec->count; ((index >= count) ? printf("}") : ({ uint8_t* data = (uint8_t*)vec->data; uint8_t* elem_loc = (uint8_t*)((uint8_t*)(((long long)data) + (index * 8))); Value** elem_ptr_ptr = (Value**)((Value**)elem_loc); Value* elem = (Value*)(*elem_ptr_ptr); g_parser.print_value_ptr(elem); (((index + 1) < count) ? ({ printf(" "); g_parser.print_map_contents(map_val, (index + 1)); }) : printf("}")); })); });
}
static int32_t print_value_ptr(Value* v) {
    return ({ ValueTag tag = v->tag; ((tag == ValueTag_Nil) ? printf("nil") : ((tag == ValueTag_Number) ? printf("%lld", v->num_val) : ((tag == ValueTag_Symbol) ? printf("%s", v->str_val) : ((tag == ValueTag_String) ? printf("\"%s\"", v->str_val) : ((tag == ValueTag_Keyword) ? printf("%s", v->str_val) : ((tag == ValueTag_List) ? ({ printf("("); g_parser.print_list_contents(v); }) : ((tag == ValueTag_Vector) ? ({ printf("["); g_parser.print_vector_contents(v, 0); }) : ((tag == ValueTag_Map) ? ({ printf("{"); g_parser.print_map_contents(v, 0); }) : printf("<unknown>"))))))))); });
}
static int32_t print_list_contents(Value* v) {
    return ({ ValueTag tag = v->tag; ((tag == ValueTag_Nil) ? printf(")") : ((tag == ValueTag_List) ? ({ uint8_t* cons_ptr = (uint8_t*)v->cons_val; Cons* cons_cell = (Cons*)((Cons*)cons_ptr); Value* car_ptr = (Value*)((Value*)cons_cell->car); Value* cdr_ptr = (Value*)((Value*)cons_cell->cdr); g_parser.print_value_ptr(car_ptr); ({ ValueTag cdr_tag = cdr_ptr->tag; ((cdr_tag == ValueTag_Nil) ? printf(")") : ({ printf(" "); g_parser.print_list_contents(cdr_ptr); })); }); }) : ({ printf(". "); g_parser.print_value_ptr(v); printf(")"); }))); });
}
static int32_t main_fn() {
    printf("Parser test - building Value structures from tokens\n\n");
    printf("Test 1: (foo bar)\n");
    ({ Token* tokens = (Token*)((Token*)malloc(128)); Token* t0 = (Token*)tokens; t0->type = TokenType_LeftParen; t0->text = "("; t0->length = 1; ({ Token* t1 = (Token*)((Token*)(((long long)tokens) + 24)); t1->type = TokenType_Symbol; t1->text = "foo"; t1->length = 3; ({ Token* t2 = (Token*)((Token*)(((long long)tokens) + 48)); t2->type = TokenType_Symbol; t2->text = "bar"; t2->length = 3; ({ Token* t3 = (Token*)((Token*)(((long long)tokens) + 72)); t3->type = TokenType_RightParen; t3->text = ")"; t3->length = 1; ({ Parser* parser = (Parser*)g_parser.make_parser(tokens, 4); Value* result = (Value*)g_parser.parse_value(parser); printf("  Result: "); g_parser.print_value_ptr(result); printf("\n\n"); }); }); }); }); });
    return 0;
}
void lisp_main(void) {
    init_namespace_types(&g_types);
    init_namespace_parser(&g_parser);
}
