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
    TokenType_EOF,
} TokenType;

typedef struct {
    TokenType type;
    uint8_t* text;
    int32_t length;
} Token;

typedef struct {
    uint8_t* input;
    int32_t position;
    int32_t length;
} Tokenizer;


typedef struct {
    Tokenizer* (*make_tokenizer)(uint8_t*);
    int32_t (*peek_char)(Tokenizer*);
    int32_t (*advance)(Tokenizer*);
    int32_t (*skip_whitespace)(Tokenizer*);
    Token (*make_token)(TokenType, uint8_t*, int32_t);
    Token (*next_token)(Tokenizer*);
    Token (*read_symbol)(Tokenizer*);
    int32_t (*main_fn)();
} Namespace_user;

Namespace_user g_user;

static Tokenizer* make_tokenizer(uint8_t*);
static int32_t peek_char(Tokenizer*);
static int32_t advance(Tokenizer*);
static int32_t skip_whitespace(Tokenizer*);
static Token make_token(TokenType, uint8_t*, int32_t);
static Token next_token(Tokenizer*);
static Token read_symbol(Tokenizer*);
static int32_t main_fn();

void init_namespace_user(Namespace_user* ns) {
    ns->make_tokenizer = &make_tokenizer;
    ns->peek_char = &peek_char;
    ns->advance = &advance;
    ns->skip_whitespace = &skip_whitespace;
    ns->make_token = &make_token;
    ns->next_token = &next_token;
    ns->read_symbol = &read_symbol;
    ns->main_fn = &main_fn;
}

static Tokenizer* make_tokenizer(uint8_t* input) {
    return ({ Tokenizer* tok = (Tokenizer*)({ Tokenizer* __tmp_ptr = malloc(sizeof(Tokenizer)); *__tmp_ptr = (Tokenizer){NULL, 0, 0}; __tmp_ptr; }); tok->input = input; tok->position = 0; tok->length = strlen(input); tok; });
}
static int32_t peek_char(Tokenizer* tok) {
    return ({ int32_t pos = tok->position; ({ int32_t len = tok->length; ((pos >= len) ? ((int32_t)0) : ({ uint8_t* input = (uint8_t*)tok->input; ({ int64_t char_loc = (((int64_t)input) + ((int64_t)pos)); ({ uint8_t* char_ptr = (uint8_t*)((uint8_t*)char_loc); ({ uint8_t byte_val = (*char_ptr); ((int32_t)byte_val); }); }); }); })); }); });
}
static int32_t advance(Tokenizer* tok) {
    return ({ int32_t pos = tok->position; tok->position = (pos + 1); 0; });
}
static int32_t skip_whitespace(Tokenizer* tok) {
    return ({ int32_t c = g_user.peek_char(tok); (((c != 0) && (isspace(c) != 0)) ? ({ int32_t _1 = g_user.advance(tok); g_user.skip_whitespace(tok); }) : 0); });
}
static Token make_token(TokenType type, uint8_t* text, int32_t length) {
    return (Token){type, text, length};
}
static Token next_token(Tokenizer* tok) {
    g_user.skip_whitespace(tok);
    return ({ int32_t c = g_user.peek_char(tok); ((c == 0) ? g_user.make_token(TokenType_EOF, NULL, 0) : ((c == 40) ? ({ int32_t _1 = g_user.advance(tok); g_user.make_token(TokenType_LeftParen, "(", 1); }) : ((c == 41) ? ({ int32_t _1 = g_user.advance(tok); g_user.make_token(TokenType_RightParen, ")", 1); }) : ((c == 91) ? ({ int32_t _1 = g_user.advance(tok); g_user.make_token(TokenType_LeftBracket, "[", 1); }) : ((c == 93) ? ({ int32_t _1 = g_user.advance(tok); g_user.make_token(TokenType_RightBracket, "]", 1); }) : g_user.read_symbol(tok)))))); });
}
static Token read_symbol(Tokenizer* tok) {
    return ({ int32_t start_pos = tok->position; ({ uint8_t* input = (uint8_t*)tok->input; ({ uint8_t* start_ptr = (uint8_t*)((uint8_t*)(((int64_t)input) + ((int64_t)start_pos))); ({ int32_t len = 0; ({ while (((g_user.peek_char(tok) != 0) && ((isspace(g_user.peek_char(tok)) == 0) && ((g_user.peek_char(tok) != 40) && ((g_user.peek_char(tok) != 41) && ((g_user.peek_char(tok) != 91) && (g_user.peek_char(tok) != 93))))))) { g_user.advance(tok); len = (len + 1); } }); g_user.make_token(TokenType_Symbol, start_ptr, len); }); }); }); });
}
static int32_t main_fn() {
    printf("Testing tokenizer:\n");
    ({ Tokenizer* tok = (Tokenizer*)g_user.make_tokenizer("(foo bar [1 2])"); ({ Token t1 = g_user.next_token(tok); printf("Token 1: type=%d text='%.*s'\n", ((int32_t)t1.type), t1.length, t1.text); ({ Token t2 = g_user.next_token(tok); printf("Token 2: type=%d text='%.*s'\n", ((int32_t)t2.type), t2.length, t2.text); ({ Token t3 = g_user.next_token(tok); printf("Token 3: type=%d text='%.*s'\n", ((int32_t)t3.type), t3.length, t3.text); ({ Token t4 = g_user.next_token(tok); printf("Token 4: type=%d text='%.*s'\n", ((int32_t)t4.type), t4.length, t4.text); ({ Token t5 = g_user.next_token(tok); printf("Token 5: type=%d text='%.*s'\n", ((int32_t)t5.type), t5.length, t5.text); ({ Token t6 = g_user.next_token(tok); printf("Token 6: type=%d text='%.*s'\n", ((int32_t)t6.type), t6.length, t6.text); ({ Token t7 = g_user.next_token(tok); printf("Token 7: type=%d text='%.*s'\n", ((int32_t)t7.type), t7.length, t7.text); ({ Token t8 = g_user.next_token(tok); printf("Token 8: type=%d text='%.*s'\n", ((int32_t)t8.type), t8.length, t8.text); 0; }); }); }); }); }); }); }); }); });
    return 0;
}
int main() {
    init_namespace_user(&g_user);
    g_user.main_fn();
    return 0;
}
