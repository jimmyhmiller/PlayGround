#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "ctype.h"
typedef enum {
    TokenType_LeftParen,
    TokenType_RightParen,
    TokenType_LeftBracket,
    TokenType_RightBracket,
    TokenType_Number,
    TokenType_Symbol,
    TokenType_String,
    TokenType_EOF,
} TokenType;

typedef struct {
    TokenType type;
    uint8_t* text;
    int32_t length;
} Token;

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
    uint8_t* data;
    int32_t count;
    int32_t capacity;
} Vector;

typedef struct {
    Token* tokens;
    int32_t position;
    int32_t count;
} Parser;


typedef struct {
    Value (*make_nil)();
    Value (*make_number)(int64_t);
    Value (*make_symbol)(uint8_t*);
    Value* (*make_cons)(Value*, Value*);
    uint8_t* (*copy_string)(uint8_t*, int32_t);
    int32_t (*is_number_token)(Token);
    Parser* (*make_parser)(Token*, int32_t);
    Token (*peek_token)(Parser*);
    int32_t (*advance_parser)(Parser*);
    Value* (*parse_list)(Parser*);
    int32_t (*vector_set)(Value*, int32_t, Value*);
    Value* (*make_vector_with_capacity)(int32_t, int32_t);
    int32_t (*parse_vector_elements)(Parser*, Value*, int32_t);
    Value* (*parse_vector)(Parser*);
    Value* (*parse_value)(Parser*);
    int32_t (*print_vector_contents)(Value*, int32_t);
    int32_t (*print_value_ptr)(Value*);
    int32_t (*print_list_contents)(Value*);
    int32_t (*main_fn)();
} Namespace_user;

Namespace_user g_user;

static Value make_nil();
static Value make_number(int64_t);
static Value make_symbol(uint8_t*);
static Value* make_cons(Value*, Value*);
static uint8_t* copy_string(uint8_t*, int32_t);
static int32_t is_number_token(Token);
static Parser* make_parser(Token*, int32_t);
static Token peek_token(Parser*);
static int32_t advance_parser(Parser*);
static Value* parse_list(Parser*);
static int32_t vector_set(Value*, int32_t, Value*);
static Value* make_vector_with_capacity(int32_t, int32_t);
static int32_t parse_vector_elements(Parser*, Value*, int32_t);
static Value* parse_vector(Parser*);
static Value* parse_value(Parser*);
static int32_t print_vector_contents(Value*, int32_t);
static int32_t print_value_ptr(Value*);
static int32_t print_list_contents(Value*);
static int32_t main_fn();

void init_namespace_user(Namespace_user* ns) {
    ns->make_nil = &make_nil;
    ns->make_number = &make_number;
    ns->make_symbol = &make_symbol;
    ns->make_cons = &make_cons;
    ns->copy_string = &copy_string;
    ns->is_number_token = &is_number_token;
    ns->make_parser = &make_parser;
    ns->peek_token = &peek_token;
    ns->advance_parser = &advance_parser;
    ns->parse_list = &parse_list;
    ns->vector_set = &vector_set;
    ns->make_vector_with_capacity = &make_vector_with_capacity;
    ns->parse_vector_elements = &parse_vector_elements;
    ns->parse_vector = &parse_vector;
    ns->parse_value = &parse_value;
    ns->print_vector_contents = &print_vector_contents;
    ns->print_value_ptr = &print_value_ptr;
    ns->print_list_contents = &print_list_contents;
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
static Value* make_cons(Value* car_val, Value* cdr_val) {
    return ({ Cons* cons_cell = (Cons*)((Cons*)malloc(16)); cons_cell->car = ((uint8_t*)car_val); cons_cell->cdr = ((uint8_t*)cdr_val); ({ Value* val = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; }); val->tag = ValueTag_List; val->cons_val = ((uint8_t*)cons_cell); val; }); });
}
static uint8_t* copy_string(uint8_t* src, int32_t len) {
    return ({ uint8_t* dest = (uint8_t*)malloc((len + 1)); strncpy(dest, src, len); ({ uint8_t* null_pos = (uint8_t*)((uint8_t*)(((int64_t)dest) + ((int64_t)len))); (*((uint8_t*)null_pos) = ((uint8_t)0)); }); dest; });
}
static int32_t is_number_token(Token tok) {
    return ((tok.length == 0) ? 0 : ({ uint8_t first_char = (*tok.text); ({ int32_t is_digit = isdigit(((int32_t)first_char)); ((is_digit != 0) ? 1 : (((first_char == ((uint8_t)45)) && (tok.length > 1)) ? ({ uint8_t* second_char_ptr = (uint8_t*)((uint8_t*)(((long long)tok.text) + 1)); ({ uint8_t second_char = (*second_char_ptr); ({ int32_t is_digit2 = isdigit(((int32_t)second_char)); ((is_digit2 != 0) ? 1 : 0); }); }); }) : 0)); }); }));
}
static Parser* make_parser(Token* tokens, int32_t count) {
    return ({ Parser* p = (Parser*)({ Parser* __tmp_ptr = malloc(sizeof(Parser)); *__tmp_ptr = (Parser){NULL, 0, 0}; __tmp_ptr; }); p->tokens = tokens; p->position = 0; p->count = count; p; });
}
static Token peek_token(Parser* p) {
    return ({ int32_t pos = p->position; ({ int32_t count = p->count; ((pos >= count) ? (Token){TokenType_EOF, NULL, 0} : ({ Token* tokens = (Token*)p->tokens; ({ Token* token_ptr = (Token*)((Token*)(((long long)tokens) + (((long long)pos) * 24))); ({ TokenType ttype = token_ptr->type; ({ uint8_t* ttext = (uint8_t*)token_ptr->text; ({ int32_t tlen = token_ptr->length; (Token){ttype, ttext, tlen}; }); }); }); }); })); }); });
}
static int32_t advance_parser(Parser* p) {
    return ({ int32_t pos = p->position; p->position = (pos + 1); 0; });
}
static Value* parse_list(Parser* p) {
    return ({ Token tok = g_user.peek_token(p); ((tok.type == TokenType_RightParen) ? ({ int32_t _ = g_user.advance_parser(p); ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; }); }) : ({ Value* first = (Value*)g_user.parse_value(p); ({ Value* rest = (Value*)g_user.parse_list(p); g_user.make_cons(first, rest); }); })); });
}
static int32_t vector_set(Value* vec_val, int32_t index, Value* elem) {
    return ({ uint8_t* vec_ptr = (uint8_t*)vec_val->vec_val; ({ Vector* vec = (Vector*)((Vector*)vec_ptr); ({ uint8_t* data = (uint8_t*)vec->data; ({ uint8_t* elem_loc = (uint8_t*)((uint8_t*)(((long long)data) + (index * 8))); ({ Value** elem_ptr = (Value**)((Value**)elem_loc); (*elem_ptr = elem); 0; }); }); }); }); });
}
static Value* make_vector_with_capacity(int32_t count, int32_t capacity) {
    return ({ Vector* vec = (Vector*)((Vector*)malloc(16)); ({ uint8_t* data = (uint8_t*)malloc((capacity * 8)); vec->data = data; vec->count = count; vec->capacity = capacity; ({ Value* val = (Value*)({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; }); val->tag = ValueTag_Vector; val->vec_val = ((uint8_t*)vec); val; }); }); });
}
static int32_t parse_vector_elements(Parser* p, Value* vec, int32_t index) {
    return ({ Token tok = g_user.peek_token(p); ((tok.type == TokenType_RightBracket) ? ({ int32_t _ = g_user.advance_parser(p); index; }) : ({ Value* elem = (Value*)g_user.parse_value(p); g_user.vector_set(vec, index, elem); g_user.parse_vector_elements(p, vec, (index + 1)); })); });
}
static Value* parse_vector(Parser* p) {
    return ({ Value* vec = (Value*)g_user.make_vector_with_capacity(16, 16); ({ int32_t final_count = g_user.parse_vector_elements(p, vec, 0); ({ uint8_t* vec_ptr = (uint8_t*)vec->vec_val; ({ Vector* vec_struct = (Vector*)((Vector*)vec_ptr); vec_struct->count = final_count; vec; }); }); }); });
}
static Value* parse_value(Parser* p) {
    return ({ Token tok = g_user.peek_token(p); ((tok.type == TokenType_LeftParen) ? ({ int32_t _ = g_user.advance_parser(p); g_user.parse_list(p); }) : ((tok.type == TokenType_LeftBracket) ? ({ int32_t _ = g_user.advance_parser(p); g_user.parse_vector(p); }) : ((tok.type == TokenType_Symbol) ? ({ int32_t _ = g_user.advance_parser(p); ((g_user.is_number_token(tok) != 0) ? ({ uint8_t* str = (uint8_t*)g_user.copy_string(tok.text, tok.length); ({ int64_t num = atoll(str); printf("[DEBUG] Parsed number: %lld\n", num); ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_number(num); __tmp_ptr; }); }); }) : ({ uint8_t* str = (uint8_t*)g_user.copy_string(tok.text, tok.length); printf("[DEBUG] Copied symbol: '%s'\n", str); ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_symbol(str); __tmp_ptr; }); })); }) : ((tok.type == TokenType_String) ? ({ int32_t _ = g_user.advance_parser(p); ({ uint8_t* str_start = (uint8_t*)((uint8_t*)(((long long)tok.text) + 1)); ({ int32_t str_len = (tok.length - 2); ({ uint8_t* str = (uint8_t*)g_user.copy_string(str_start, str_len); printf("[DEBUG] Copied string: \"%s\"\n", str); ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = (Value){ValueTag_String, 0, str, NULL, NULL}; __tmp_ptr; }); }); }); }); }) : ({ Value* __tmp_ptr = malloc(sizeof(Value)); *__tmp_ptr = g_user.make_nil(); __tmp_ptr; }))))); });
}
static int32_t print_vector_contents(Value* vec_val, int32_t index) {
    return ({ uint8_t* vec_ptr = (uint8_t*)vec_val->vec_val; ({ Vector* vec = (Vector*)((Vector*)vec_ptr); ({ int32_t count = vec->count; ((index >= count) ? printf("]") : ({ uint8_t* data = (uint8_t*)vec->data; ({ uint8_t* elem_loc = (uint8_t*)((uint8_t*)(((long long)data) + (index * 8))); ({ Value** elem_ptr_ptr = (Value**)((Value**)elem_loc); ({ Value* elem = (Value*)(*elem_ptr_ptr); g_user.print_value_ptr(elem); (((index + 1) < count) ? ({ int32_t _ = printf(" "); g_user.print_vector_contents(vec_val, (index + 1)); }) : printf("]")); }); }); }); })); }); }); });
}
static int32_t print_value_ptr(Value* v) {
    return ({ ValueTag tag = v->tag; ((tag == ValueTag_Nil) ? printf("nil") : ((tag == ValueTag_Number) ? printf("%lld", v->num_val) : ((tag == ValueTag_Symbol) ? printf("%s", v->str_val) : ((tag == ValueTag_String) ? printf("\"%s\"", v->str_val) : ((tag == ValueTag_List) ? ({ int32_t _ = printf("("); g_user.print_list_contents(v); }) : ((tag == ValueTag_Vector) ? ({ int32_t _ = printf("["); g_user.print_vector_contents(v, 0); }) : printf("<unknown>"))))))); });
}
static int32_t print_list_contents(Value* v) {
    return ({ ValueTag tag = v->tag; ((tag == ValueTag_Nil) ? printf(")") : ((tag == ValueTag_List) ? ({ uint8_t* cons_ptr = (uint8_t*)v->cons_val; ({ Cons* cons_cell = (Cons*)((Cons*)cons_ptr); ({ Value* car_ptr = (Value*)((Value*)cons_cell->car); ({ Value* cdr_ptr = (Value*)((Value*)cons_cell->cdr); g_user.print_value_ptr(car_ptr); ({ ValueTag cdr_tag = cdr_ptr->tag; ((cdr_tag == ValueTag_Nil) ? printf(")") : ({ int32_t _ = printf(" "); g_user.print_list_contents(cdr_ptr); })); }); }); }); }); }) : ({ int32_t _ = printf(". "); ({ int32_t _ = g_user.print_value_ptr(v); printf(")"); }); }))); });
}
static int32_t main_fn() {
    printf("Parser test - building Value structures from tokens\n\n");
    printf("Test 1: (foo bar)\n");
    ({ Token* tokens = (Token*)((Token*)malloc(128)); ({ Token* t0 = (Token*)tokens; t0->type = TokenType_LeftParen; t0->text = "("; t0->length = 1; ({ Token* t1 = (Token*)((Token*)(((long long)tokens) + 24)); t1->type = TokenType_Symbol; t1->text = "foo"; t1->length = 3; ({ Token* t2 = (Token*)((Token*)(((long long)tokens) + 48)); t2->type = TokenType_Symbol; t2->text = "bar"; t2->length = 3; ({ Token* t3 = (Token*)((Token*)(((long long)tokens) + 72)); t3->type = TokenType_RightParen; t3->text = ")"; t3->length = 1; ({ Parser* parser = (Parser*)g_user.make_parser(tokens, 4); ({ Value* result = (Value*)g_user.parse_value(parser); printf("  Result: "); g_user.print_value_ptr(result); printf("\n\n"); }); }); }); }); }); }); });
    printf("Test 2: [a b c]\n");
    ({ Token* tokens2 = (Token*)((Token*)malloc(256)); ({ Token* t0 = (Token*)tokens2; t0->type = TokenType_LeftBracket; t0->text = "["; t0->length = 1; ({ Token* t1 = (Token*)((Token*)(((long long)tokens2) + 24)); t1->type = TokenType_Symbol; t1->text = "a"; t1->length = 1; ({ Token* t2 = (Token*)((Token*)(((long long)tokens2) + 48)); t2->type = TokenType_Symbol; t2->text = "b"; t2->length = 1; ({ Token* t3 = (Token*)((Token*)(((long long)tokens2) + 72)); t3->type = TokenType_Symbol; t3->text = "c"; t3->length = 1; ({ Token* t4 = (Token*)((Token*)(((long long)tokens2) + 96)); t4->type = TokenType_RightBracket; t4->text = "]"; t4->length = 1; ({ Parser* parser = (Parser*)g_user.make_parser(tokens2, 5); ({ Value* result = (Value*)g_user.parse_value(parser); printf("  Result: "); g_user.print_value_ptr(result); printf("\n\n"); }); }); }); }); }); }); }); });
    printf("Test 3: (foo [bar baz])\n");
    ({ Token* tokens3 = (Token*)((Token*)malloc(256)); ({ Token* t0 = (Token*)tokens3; t0->type = TokenType_LeftParen; ({ Token* t1 = (Token*)((Token*)(((long long)tokens3) + 24)); t1->type = TokenType_Symbol; t1->text = "foo"; t1->length = 3; ({ Token* t2 = (Token*)((Token*)(((long long)tokens3) + 48)); t2->type = TokenType_LeftBracket; ({ Token* t3 = (Token*)((Token*)(((long long)tokens3) + 72)); t3->type = TokenType_Symbol; t3->text = "bar"; t3->length = 3; ({ Token* t4 = (Token*)((Token*)(((long long)tokens3) + 96)); t4->type = TokenType_Symbol; t4->text = "baz"; t4->length = 3; ({ Token* t5 = (Token*)((Token*)(((long long)tokens3) + 120)); t5->type = TokenType_RightBracket; ({ Token* t6 = (Token*)((Token*)(((long long)tokens3) + 144)); t6->type = TokenType_RightParen; ({ Parser* parser = (Parser*)g_user.make_parser(tokens3, 7); ({ Value* result = (Value*)g_user.parse_value(parser); printf("  Result: "); g_user.print_value_ptr(result); printf("\n\n"); }); }); }); }); }); }); }); }); }); });
    printf("Test 4: (add 1 2)\n");
    ({ Token* tokens4 = (Token*)((Token*)malloc(256)); ({ Token* t0 = (Token*)tokens4; t0->type = TokenType_LeftParen; t0->text = "("; t0->length = 1; ({ Token* t1 = (Token*)((Token*)(((long long)tokens4) + 24)); t1->type = TokenType_Symbol; t1->text = "add"; t1->length = 3; ({ Token* t2 = (Token*)((Token*)(((long long)tokens4) + 48)); t2->type = TokenType_Symbol; t2->text = "1"; t2->length = 1; ({ Token* t3 = (Token*)((Token*)(((long long)tokens4) + 72)); t3->type = TokenType_Symbol; t3->text = "2"; t3->length = 1; ({ Token* t4 = (Token*)((Token*)(((long long)tokens4) + 96)); t4->type = TokenType_RightParen; t4->text = ")"; t4->length = 1; ({ Parser* parser = (Parser*)g_user.make_parser(tokens4, 5); ({ Value* result = (Value*)g_user.parse_value(parser); printf("  Result: "); g_user.print_value_ptr(result); printf("\n\n"); }); }); }); }); }); }); }); });
    printf("Test 5: [42 -5 100]\n");
    ({ Token* tokens5 = (Token*)((Token*)malloc(256)); ({ Token* t0 = (Token*)tokens5; t0->type = TokenType_LeftBracket; t0->text = "["; t0->length = 1; ({ Token* t1 = (Token*)((Token*)(((long long)tokens5) + 24)); t1->type = TokenType_Symbol; t1->text = "42"; t1->length = 2; ({ Token* t2 = (Token*)((Token*)(((long long)tokens5) + 48)); t2->type = TokenType_Symbol; t2->text = "-5"; t2->length = 2; ({ Token* t3 = (Token*)((Token*)(((long long)tokens5) + 72)); t3->type = TokenType_Symbol; t3->text = "100"; t3->length = 3; ({ Token* t4 = (Token*)((Token*)(((long long)tokens5) + 96)); t4->type = TokenType_RightBracket; t4->text = "]"; t4->length = 1; ({ Parser* parser = (Parser*)g_user.make_parser(tokens5, 5); ({ Value* result = (Value*)g_user.parse_value(parser); printf("  Result: "); g_user.print_value_ptr(result); printf("\n\n"); }); }); }); }); }); }); }); });
    printf("Test 6: (println \"hello world\")\n");
    ({ Token* tokens6 = (Token*)((Token*)malloc(256)); ({ Token* t0 = (Token*)tokens6; t0->type = TokenType_LeftParen; t0->text = "("; t0->length = 1; ({ Token* t1 = (Token*)((Token*)(((long long)tokens6) + 24)); t1->type = TokenType_Symbol; t1->text = "println"; t1->length = 7; ({ Token* t2 = (Token*)((Token*)(((long long)tokens6) + 48)); t2->type = TokenType_String; t2->text = "\"hello world\""; t2->length = 13; ({ Token* t3 = (Token*)((Token*)(((long long)tokens6) + 72)); t3->type = TokenType_RightParen; t3->text = ")"; t3->length = 1; ({ Parser* parser = (Parser*)g_user.make_parser(tokens6, 4); ({ Value* result = (Value*)g_user.parse_value(parser); printf("  Result: "); g_user.print_value_ptr(result); printf("\n"); }); }); }); }); }); }); });
    return 0;
}
int main() {
    init_namespace_user(&g_user);
    g_user.main_fn();
    return 0;
}
