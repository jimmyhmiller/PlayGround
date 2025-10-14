#define _POSIX_C_SOURCE 200809L
#include "bencode.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>

static bencode_value_t *parse_value(const char *data, size_t len, size_t *pos);

static bencode_value_t *parse_int(const char *data, size_t len, size_t *pos) {
    (*pos)++; // skip 'i'

    long long val = 0;
    int negative = 0;

    if (data[*pos] == '-') {
        negative = 1;
        (*pos)++;
    }

    while (*pos < len && data[*pos] != 'e') {
        if (!isdigit(data[*pos])) return NULL;
        val = val * 10 + (data[*pos] - '0');
        (*pos)++;
    }

    if (*pos >= len) return NULL;
    (*pos)++; // skip 'e'

    return bencode_create_int(negative ? -val : val);
}

static bencode_value_t *parse_str(const char *data, size_t len, size_t *pos) {
    size_t str_len = 0;

    while (*pos < len && isdigit(data[*pos])) {
        str_len = str_len * 10 + (data[*pos] - '0');
        (*pos)++;
    }

    if (*pos >= len || data[*pos] != ':') return NULL;
    (*pos)++; // skip ':'

    if (*pos + str_len > len) return NULL;

    bencode_value_t *result = bencode_create_str(data + *pos, str_len);
    *pos += str_len;

    return result;
}

static bencode_value_t *parse_list(const char *data, size_t len, size_t *pos) {
    (*pos)++; // skip 'l'

    bencode_value_t *list = bencode_create_list();

    while (*pos < len && data[*pos] != 'e') {
        bencode_value_t *item = parse_value(data, len, pos);
        if (!item) {
            bencode_free(list);
            return NULL;
        }
        bencode_list_append(list, item);
    }

    if (*pos >= len) {
        bencode_free(list);
        return NULL;
    }

    (*pos)++; // skip 'e'
    return list;
}

static bencode_value_t *parse_dict(const char *data, size_t len, size_t *pos) {
    (*pos)++; // skip 'd'

    bencode_value_t *dict = bencode_create_dict();

    while (*pos < len && data[*pos] != 'e') {
        // Parse key (must be string)
        if (!isdigit(data[*pos])) {
            bencode_free(dict);
            return NULL;
        }

        bencode_value_t *key_val = parse_str(data, len, pos);
        if (!key_val) {
            bencode_free(dict);
            return NULL;
        }

        // Parse value
        bencode_value_t *value = parse_value(data, len, pos);
        if (!value) {
            bencode_free(key_val);
            bencode_free(dict);
            return NULL;
        }

        bencode_dict_set(dict, key_val->str_val.data, value);
        bencode_free(key_val);
    }

    if (*pos >= len) {
        bencode_free(dict);
        return NULL;
    }

    (*pos)++; // skip 'e'
    return dict;
}

static bencode_value_t *parse_value(const char *data, size_t len, size_t *pos) {
    if (*pos >= len) return NULL;

    char c = data[*pos];

    if (c == 'i') return parse_int(data, len, pos);
    if (c == 'l') return parse_list(data, len, pos);
    if (c == 'd') return parse_dict(data, len, pos);
    if (isdigit(c)) return parse_str(data, len, pos);

    return NULL;
}

bencode_value_t *bencode_parse(const char *data, size_t len, size_t *parsed) {
    size_t pos = 0;
    bencode_value_t *result = parse_value(data, len, &pos);
    if (parsed) *parsed = pos;
    return result;
}

static void encode_int(long long val, char **buf, size_t *len, size_t *cap) {
    char temp[64];
    int n = snprintf(temp, sizeof(temp), "i%llde", val);

    while (*len + n >= *cap) {
        *cap *= 2;
        *buf = realloc(*buf, *cap);
    }

    memcpy(*buf + *len, temp, n);
    *len += n;
}

static void encode_str(const char *str, size_t str_len, char **buf, size_t *len, size_t *cap) {
    char temp[32];
    int n = snprintf(temp, sizeof(temp), "%zu:", str_len);

    while (*len + n + str_len >= *cap) {
        *cap *= 2;
        *buf = realloc(*buf, *cap);
    }

    memcpy(*buf + *len, temp, n);
    *len += n;
    memcpy(*buf + *len, str, str_len);
    *len += str_len;
}

static void encode_value(bencode_value_t *value, char **buf, size_t *len, size_t *cap);

static void encode_list(bencode_value_t *list, char **buf, size_t *len, size_t *cap) {
    while (*len + 2 >= *cap) {
        *cap *= 2;
        *buf = realloc(*buf, *cap);
    }

    (*buf)[(*len)++] = 'l';

    for (size_t i = 0; i < list->list_val.count; i++) {
        encode_value(list->list_val.items[i], buf, len, cap);
    }

    (*buf)[(*len)++] = 'e';
}

static void encode_dict(bencode_value_t *dict, char **buf, size_t *len, size_t *cap) {
    while (*len + 2 >= *cap) {
        *cap *= 2;
        *buf = realloc(*buf, *cap);
    }

    (*buf)[(*len)++] = 'd';

    for (size_t i = 0; i < dict->dict_val.count; i++) {
        encode_str(dict->dict_val.entries[i].key,
                  strlen(dict->dict_val.entries[i].key),
                  buf, len, cap);
        encode_value(dict->dict_val.entries[i].value, buf, len, cap);
    }

    (*buf)[(*len)++] = 'e';
}

static void encode_value(bencode_value_t *value, char **buf, size_t *len, size_t *cap) {
    switch (value->type) {
        case BENCODE_INT:
            encode_int(value->int_val, buf, len, cap);
            break;
        case BENCODE_STR:
            encode_str(value->str_val.data, value->str_val.len, buf, len, cap);
            break;
        case BENCODE_LIST:
            encode_list(value, buf, len, cap);
            break;
        case BENCODE_DICT:
            encode_dict(value, buf, len, cap);
            break;
    }
}

char *bencode_encode(bencode_value_t *value, size_t *out_len) {
    size_t cap = 256;
    size_t len = 0;
    char *buf = malloc(cap);

    encode_value(value, &buf, &len, &cap);

    if (out_len) *out_len = len;
    return buf;
}

void bencode_free(bencode_value_t *value) {
    if (!value) return;

    switch (value->type) {
        case BENCODE_STR:
            free(value->str_val.data);
            break;
        case BENCODE_LIST:
            for (size_t i = 0; i < value->list_val.count; i++) {
                bencode_free(value->list_val.items[i]);
            }
            free(value->list_val.items);
            break;
        case BENCODE_DICT:
            for (size_t i = 0; i < value->dict_val.count; i++) {
                free(value->dict_val.entries[i].key);
                bencode_free(value->dict_val.entries[i].value);
            }
            free(value->dict_val.entries);
            break;
        default:
            break;
    }

    free(value);
}

bencode_value_t *bencode_create_int(long long val) {
    bencode_value_t *v = malloc(sizeof(bencode_value_t));
    v->type = BENCODE_INT;
    v->int_val = val;
    return v;
}

bencode_value_t *bencode_create_str(const char *str, size_t len) {
    bencode_value_t *v = malloc(sizeof(bencode_value_t));
    v->type = BENCODE_STR;
    v->str_val.data = malloc(len + 1);
    memcpy(v->str_val.data, str, len);
    v->str_val.data[len] = '\0';
    v->str_val.len = len;
    return v;
}

bencode_value_t *bencode_create_list(void) {
    bencode_value_t *v = malloc(sizeof(bencode_value_t));
    v->type = BENCODE_LIST;
    v->list_val.items = NULL;
    v->list_val.count = 0;
    return v;
}

bencode_value_t *bencode_create_dict(void) {
    bencode_value_t *v = malloc(sizeof(bencode_value_t));
    v->type = BENCODE_DICT;
    v->dict_val.entries = NULL;
    v->dict_val.count = 0;
    return v;
}

void bencode_list_append(bencode_value_t *list, bencode_value_t *item) {
    list->list_val.items = realloc(list->list_val.items,
                                   (list->list_val.count + 1) * sizeof(bencode_value_t *));
    list->list_val.items[list->list_val.count++] = item;
}

void bencode_dict_set(bencode_value_t *dict, const char *key, bencode_value_t *value) {
    dict->dict_val.entries = realloc(dict->dict_val.entries,
                                     (dict->dict_val.count + 1) * sizeof(bencode_dict_entry_t));
    dict->dict_val.entries[dict->dict_val.count].key = strdup(key);
    dict->dict_val.entries[dict->dict_val.count].value = value;
    dict->dict_val.count++;
}

bencode_value_t *bencode_dict_get(bencode_value_t *dict, const char *key) {
    for (size_t i = 0; i < dict->dict_val.count; i++) {
        if (strcmp(dict->dict_val.entries[i].key, key) == 0) {
            return dict->dict_val.entries[i].value;
        }
    }
    return NULL;
}
