#include "nrepl.h"
#include "bencode.h"
#include "simple_eval.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/wait.h>

#define PORT 7888
#define BUFFER_SIZE 65536

void handle_client(int client_sock, nrepl_server_t *server) {
    char buffer[BUFFER_SIZE];
    ssize_t bytes_read;

    while ((bytes_read = recv(client_sock, buffer, BUFFER_SIZE - 1, 0)) > 0) {
        printf("Received %zd bytes\n", bytes_read);

        // Parse bencode message
        size_t parsed = 0;
        bencode_value_t *msg = bencode_parse(buffer, bytes_read, &parsed);

        if (!msg) {
            fprintf(stderr, "Failed to parse bencode message\n");
            continue;
        }

        // Handle the message
        bencode_value_t *response = nrepl_handle_message(server, msg);

        if (response) {
            // Encode and send response
            size_t response_len;
            char *response_data = bencode_encode(response, &response_len);

            printf("Sending response: %zd bytes\n", response_len);
            send(client_sock, response_data, response_len, 0);

            free(response_data);
            bencode_free(response);
        }

        bencode_free(msg);
    }

    if (bytes_read < 0) {
        perror("recv failed");
    }

    close(client_sock);
    printf("Client disconnected\n");
}

int main(int argc, char *argv[]) {
    int port = PORT;

    if (argc > 1) {
        port = atoi(argv[1]);
    }

    nrepl_server_t server;
    nrepl_init(&server);

    // Set up the simple evaluator
    nrepl_set_evaluator(simple_eval);

    int server_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (server_sock < 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    int opt = 1;
    if (setsockopt(server_sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        perror("setsockopt failed");
        exit(EXIT_FAILURE);
    }

    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);

    if (bind(server_sock, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

    if (listen(server_sock, 10) < 0) {
        perror("listen failed");
        exit(EXIT_FAILURE);
    }

    printf("nREPL server listening on port %d\n", port);

    // Ignore SIGCHLD to prevent zombie processes
    signal(SIGCHLD, SIG_IGN);

    while (1) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);

        int client_sock = accept(server_sock, (struct sockaddr *)&client_addr, &client_len);
        if (client_sock < 0) {
            perror("accept failed");
            continue;
        }

        printf("Client connected from %s:%d\n",
               inet_ntoa(client_addr.sin_addr),
               ntohs(client_addr.sin_port));

        // Fork to handle client concurrently
        pid_t pid = fork();
        if (pid == 0) {
            // Child process
            close(server_sock);  // Child doesn't need the listening socket
            handle_client(client_sock, &server);
            exit(0);
        } else if (pid > 0) {
            // Parent process
            close(client_sock);  // Parent doesn't need the client socket
        } else {
            perror("fork failed");
            close(client_sock);
        }
    }

    close(server_sock);
    return 0;
}
