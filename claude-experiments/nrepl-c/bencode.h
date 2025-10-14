#ifndef BENCODE_H
#define BENCODE_H

#include <stddef.h>

typedef enum {
    BENCODE_INT,
    BENCODE_STR,
    BENCODE_LIST,
    BENCODE_DICT
} bencode_type_t;

typedef struct bencode_value bencode_value_t;

typedef struct {
    char *key;
    bencode_value_t *value;
} bencode_dict_entry_t;

struct bencode_value {
    bencode_type_t type;
    union {
        long long int_val;
        struct {
            char *data;
            size_t len;
        } str_val;
        struct {
            bencode_value_t **items;
            size_t count;
        } list_val;
        struct {
            bencode_dict_entry_t *entries;
            size_t count;
        } dict_val;
    };
};

// Parse bencode from buffer
bencode_value_t *bencode_parse(const char *data, size_t len, size_t *parsed);

// Encode bencode to buffer
char *bencode_encode(bencode_value_t *value, size_t *out_len);

// Free bencode value
void bencode_free(bencode_value_t *value);

// Helper functions for creating bencode values
bencode_value_t *bencode_create_int(long long val);
bencode_value_t *bencode_create_str(const char *str, size_t len);
bencode_value_t *bencode_create_list(void);
bencode_value_t *bencode_create_dict(void);

// Helper functions for manipulating bencode values
void bencode_list_append(bencode_value_t *list, bencode_value_t *item);
void bencode_dict_set(bencode_value_t *dict, const char *key, bencode_value_t *value);
bencode_value_t *bencode_dict_get(bencode_value_t *dict, const char *key);

#endif
