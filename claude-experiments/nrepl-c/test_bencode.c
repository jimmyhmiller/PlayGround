#include "bencode.h"
#include "nrepl.h"
#include <stdio.h>
#include <string.h>

int main() {
    // Test bencode message: d2:op8:describe2:id1:1e
    const char *test_msg = "d2:op8:describe2:id1:1e";
    size_t len = strlen(test_msg);

    printf("Parsing: %s\n", test_msg);

    size_t parsed = 0;
    bencode_value_t *msg = bencode_parse(test_msg, len, &parsed);

    if (!msg) {
        printf("Failed to parse!\n");
        return 1;
    }

    printf("Parsed successfully! Type: %d\n", msg->type);

    // Get op
    bencode_value_t *op = bencode_dict_get(msg, "op");
    if (op) {
        printf("Op: %.*s\n", (int)op->str_val.len, op->str_val.data);
    }

    // Initialize nREPL server
    nrepl_server_t server;
    nrepl_init(&server);

    // Handle message
    bencode_value_t *response = nrepl_handle_message(&server, msg);

    if (!response) {
        printf("No response!\n");
        return 1;
    }

    printf("Response generated!\n");

    // Encode response
    size_t response_len;
    char *response_data = bencode_encode(response, &response_len);

    printf("Response: %.*s\n", (int)response_len, response_data);

    free(response_data);
    bencode_free(response);
    bencode_free(msg);

    return 0;
}
