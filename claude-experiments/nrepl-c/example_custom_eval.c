// Example: Using nrepl-c as a library with custom evaluator

#include "nrepl.h"
#include "bencode.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/wait.h>

// Custom evaluator - you can replace this with your own!
char* my_custom_evaluator(const char *code) {
    printf("EVALUATING: %s\n", code);
    fflush(stdout);

    // Example: Simple string responses
    if (strstr(code, "hello")) {
        return strdup("\"Hello, World!\"");
    } else if (strstr(code, "time")) {
        return strdup("\"Time to code!\"");
    } else {
        return strdup("\"I don't understand that yet!\"");
    }
}

// Global client socket for callback
static int current_client_sock = -1;

void send_response_callback(bencode_value_t *resp) {
    if (current_client_sock >= 0) {
        size_t response_len;
        char *response_data = bencode_encode(resp, &response_len);
        printf("Sending intermediate response: %zd bytes: %.*s\n", response_len, (int)response_len, response_data);
        fflush(stdout);
        send(current_client_sock, response_data, response_len, 0);
        free(response_data);
    }
}

void handle_client(int client_sock, nrepl_server_t *server) {
    char buffer[65536];
    ssize_t bytes_read;

    current_client_sock = client_sock;

    while ((bytes_read = recv(client_sock, buffer, 65536 - 1, 0)) > 0) {
        printf("Received %zd bytes\n", bytes_read);
        fflush(stdout);

        size_t parsed = 0;
        bencode_value_t *msg = bencode_parse(buffer, bytes_read, &parsed);

        if (!msg) {
            fprintf(stderr, "Failed to parse bencode message\n");
            continue;
        }

        // Print the op
        bencode_value_t *op = bencode_dict_get(msg, "op");
        if (op && op->type == BENCODE_STR) {
            printf("OP: %.*s\n", (int)op->str_val.len, op->str_val.data);
        }

        // Print code if it's an eval
        bencode_value_t *code = bencode_dict_get(msg, "code");
        if (code && code->type == BENCODE_STR) {
            printf("CODE: %.*s\n", (int)code->str_val.len, code->str_val.data);
        }
        fflush(stdout);

        bencode_value_t *response = nrepl_handle_message(server, msg);

        if (response) {
            size_t response_len;
            char *response_data = bencode_encode(response, &response_len);
            printf("Sending final response: %zd bytes: %.*s\n", response_len, (int)response_len, response_data);
            fflush(stdout);
            send(client_sock, response_data, response_len, 0);
            free(response_data);
            bencode_free(response);
        }

        bencode_free(msg);
    }

    current_client_sock = -1;
    close(client_sock);
    printf("Client disconnected\n");
    fflush(stdout);
}

int main(int argc, char *argv[]) {
    int port = argc > 1 ? atoi(argv[1]) : 7888;

    // Initialize nREPL server
    nrepl_server_t server;
    nrepl_init(&server);

    // Set your custom evaluator!
    nrepl_set_evaluator(my_custom_evaluator);

    // Set response callback for streaming responses
    nrepl_set_response_callback(send_response_callback);

    // Setup socket
    int server_sock = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(server_sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);

    bind(server_sock, (struct sockaddr *)&address, sizeof(address));
    listen(server_sock, 10);

    printf("Custom nREPL server listening on port %d\n", port);
    printf("Try: (hello) or (time)\n");

    signal(SIGCHLD, SIG_IGN);

    while (1) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_sock = accept(server_sock, (struct sockaddr *)&client_addr, &client_len);

        if (client_sock < 0) continue;

        pid_t pid = fork();
        if (pid == 0) {
            close(server_sock);
            handle_client(client_sock, &server);
            exit(0);
        } else if (pid > 0) {
            close(client_sock);
        }
    }

    close(server_sock);
    return 0;
}
