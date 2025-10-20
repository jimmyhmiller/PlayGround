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
typedef struct Tokenizer Tokenizer;
struct Tokenizer {
    uint8_t* input;
    int32_t position;
    int32_t length;
};

#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "ctype.h"

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

Namespace_tokenizer g_tokenizer;

static Tokenizer* make_tokenizer(uint8_t*);
static int32_t peek_char(Tokenizer*);
static int32_t advance(Tokenizer*);
static int32_t skip_to_eol(Tokenizer*);
static int32_t skip_whitespace(Tokenizer*);
static Token make_token(TokenType, uint8_t*, int32_t);
static Token next_token(Tokenizer*);
static Token read_symbol(Tokenizer*);
static Token read_string(Tokenizer*);
static Token read_keyword(Tokenizer*);
static int32_t main_fn();

void init_namespace_tokenizer(Namespace_tokenizer* ns) {
    init_namespace_types(&g_types);
    ns->make_tokenizer = &make_tokenizer;
    ns->peek_char = &peek_char;
    ns->advance = &advance;
    ns->skip_to_eol = &skip_to_eol;
    ns->skip_whitespace = &skip_whitespace;
    ns->make_token = &make_token;
    ns->next_token = &next_token;
    ns->read_symbol = &read_symbol;
    ns->read_string = &read_string;
    ns->read_keyword = &read_keyword;
    ns->main_fn = &main_fn;
}

static Tokenizer* make_tokenizer(uint8_t* input) {
    return ({ Tokenizer* tok = (Tokenizer*)({ Tokenizer* __tmp_ptr = malloc(sizeof(Tokenizer)); *__tmp_ptr = (Tokenizer){NULL, 0, 0}; __tmp_ptr; }); tok->input = input; tok->position = 0; tok->length = strlen(input); tok; });
}
static int32_t peek_char(Tokenizer* tok) {
    return ({ int32_t pos = tok->position; int32_t len = tok->length; ((pos >= len) ? ((int32_t)0) : ({ uint8_t* input = (uint8_t*)tok->input; int64_t char_loc = (((int64_t)input) + ((int64_t)pos)); uint8_t* char_ptr = (uint8_t*)((uint8_t*)char_loc); uint8_t byte_val = (*char_ptr); ((int32_t)byte_val); })); });
}
static int32_t advance(Tokenizer* tok) {
    return ({ int32_t pos = tok->position; tok->position = (pos + 1); 0; });
}
static int32_t skip_to_eol(Tokenizer* tok) {
    return ({ int32_t c = g_tokenizer.peek_char(tok); (((c != 0) && (c != 10)) ? ({ int32_t _1 = g_tokenizer.advance(tok); g_tokenizer.skip_to_eol(tok); }) : 0); });
}
static int32_t skip_whitespace(Tokenizer* tok) {
    return ({ int32_t c = g_tokenizer.peek_char(tok); ((c == 59) ? ({ int32_t _1 = g_tokenizer.skip_to_eol(tok); g_tokenizer.skip_whitespace(tok); }) : (((c != 0) && (isspace(c) != 0)) ? ({ int32_t _1 = g_tokenizer.advance(tok); g_tokenizer.skip_whitespace(tok); }) : 0)); });
}
static Token make_token(TokenType type, uint8_t* text, int32_t length) {
    return (Token){type, text, length};
}
static Token next_token(Tokenizer* tok) {
    g_tokenizer.skip_whitespace(tok);
    return ({ int32_t c = g_tokenizer.peek_char(tok); ((c == 0) ? g_tokenizer.make_token(TokenType_EOF, ((uint8_t*)0), 0) : ((c == 40) ? ({ int32_t _1 = g_tokenizer.advance(tok); g_tokenizer.make_token(TokenType_LeftParen, "(", 1); }) : ((c == 41) ? ({ int32_t _1 = g_tokenizer.advance(tok); g_tokenizer.make_token(TokenType_RightParen, ")", 1); }) : ((c == 91) ? ({ int32_t _1 = g_tokenizer.advance(tok); g_tokenizer.make_token(TokenType_LeftBracket, "[", 1); }) : ((c == 93) ? ({ int32_t _1 = g_tokenizer.advance(tok); g_tokenizer.make_token(TokenType_RightBracket, "]", 1); }) : ((c == 123) ? ({ int32_t _1 = g_tokenizer.advance(tok); g_tokenizer.make_token(TokenType_LeftBrace, "{", 1); }) : ((c == 125) ? ({ int32_t _1 = g_tokenizer.advance(tok); g_tokenizer.make_token(TokenType_RightBrace, "}", 1); }) : ((c == 34) ? g_tokenizer.read_string(tok) : ((c == 58) ? g_tokenizer.read_keyword(tok) : g_tokenizer.read_symbol(tok)))))))))); });
}
static Token read_symbol(Tokenizer* tok) {
    return ({ int32_t start_pos = tok->position; uint8_t* input = (uint8_t*)tok->input; uint8_t* start_ptr = (uint8_t*)((uint8_t*)(((int64_t)input) + ((int64_t)start_pos))); int32_t len = 0; ({ while (((g_tokenizer.peek_char(tok) != 0) && ((isspace(g_tokenizer.peek_char(tok)) == 0) && ((g_tokenizer.peek_char(tok) != 40) && ((g_tokenizer.peek_char(tok) != 41) && ((g_tokenizer.peek_char(tok) != 91) && (g_tokenizer.peek_char(tok) != 93))))))) { g_tokenizer.advance(tok); len = (len + 1); } }); g_tokenizer.make_token(TokenType_Symbol, start_ptr, len); });
}
static Token read_string(Tokenizer* tok) {
    return ({ int32_t start_pos = tok->position; uint8_t* input = (uint8_t*)tok->input; uint8_t* start_ptr = (uint8_t*)((uint8_t*)(((int64_t)input) + ((int64_t)start_pos))); ({ g_tokenizer.advance(tok); int32_t len = 1; ({ while (((g_tokenizer.peek_char(tok) != 0) && (g_tokenizer.peek_char(tok) != 34))) { ({ if ((g_tokenizer.peek_char(tok) == 92)) { ({ g_tokenizer.advance(tok); g_tokenizer.advance(tok); len = (len + 2); }); } else { ({ g_tokenizer.advance(tok); len = (len + 1); }); } }); } }); ({ g_tokenizer.advance(tok); g_tokenizer.make_token(TokenType_String, start_ptr, (len + 1)); }); }); });
}
static Token read_keyword(Tokenizer* tok) {
    return ({ int32_t start_pos = tok->position; uint8_t* input = (uint8_t*)tok->input; uint8_t* start_ptr = (uint8_t*)((uint8_t*)(((int64_t)input) + ((int64_t)start_pos))); int32_t len = 0; ({ while (((g_tokenizer.peek_char(tok) != 0) && ((isspace(g_tokenizer.peek_char(tok)) == 0) && ((g_tokenizer.peek_char(tok) != 40) && ((g_tokenizer.peek_char(tok) != 41) && ((g_tokenizer.peek_char(tok) != 91) && ((g_tokenizer.peek_char(tok) != 93) && ((g_tokenizer.peek_char(tok) != 123) && (g_tokenizer.peek_char(tok) != 125))))))))) { g_tokenizer.advance(tok); len = (len + 1); } }); g_tokenizer.make_token(TokenType_Keyword, start_ptr, len); });
}
static int32_t main_fn() {
    printf("Testing tokenizer:\n");
    ({ Tokenizer* tok = (Tokenizer*)g_tokenizer.make_tokenizer(";; comment\n(foo bar)"); Token t1 = g_tokenizer.next_token(tok); printf("Token 1: type=%d text='%.*s'\n", ((int32_t)t1.type), t1.length, t1.text); ({ Token t2 = g_tokenizer.next_token(tok); printf("Token 2: type=%d text='%.*s'\n", ((int32_t)t2.type), t2.length, t2.text); ({ Token t3 = g_tokenizer.next_token(tok); printf("Token 3: type=%d text='%.*s'\n", ((int32_t)t3.type), t3.length, t3.text); ({ Token t4 = g_tokenizer.next_token(tok); printf("Token 4: type=%d text='%.*s'\n", ((int32_t)t4.type), t4.length, t4.text); ({ Token t5 = g_tokenizer.next_token(tok); printf("Token 5: type=%d text='%.*s'\n", ((int32_t)t5.type), t5.length, t5.text); ({ Token t6 = g_tokenizer.next_token(tok); printf("Token 6: type=%d text='%.*s'\n", ((int32_t)t6.type), t6.length, t6.text); ({ Token t7 = g_tokenizer.next_token(tok); printf("Token 7: type=%d text='%.*s'\n", ((int32_t)t7.type), t7.length, t7.text); ({ Token t8 = g_tokenizer.next_token(tok); printf("Token 8: type=%d text='%.*s'\n", ((int32_t)t8.type), t8.length, t8.text); 0; }); }); }); }); }); }); }); });
    return 0;
}
void lisp_main(void) {
    init_namespace_types(&g_types);
    init_namespace_tokenizer(&g_tokenizer);
}
