#ifndef NREPL_H
#define NREPL_H

#include "bencode.h"
#include <uuid/uuid.h>

#define MAX_SESSIONS 100

typedef struct {
    char id[37];  // UUID string
    int active;
} session_t;

typedef struct {
    session_t sessions[MAX_SESSIONS];
    int session_count;
} nrepl_server_t;

// Evaluator callback type
// Takes code string, returns result string (caller must free)
typedef char* (*nrepl_eval_fn)(const char *code);

// Initialize nREPL server
void nrepl_init(nrepl_server_t *server);

// Set custom evaluator function
void nrepl_set_evaluator(nrepl_eval_fn eval_fn);

// Set callback for sending intermediate responses (for streaming)
void nrepl_set_response_callback(void (*callback)(bencode_value_t *resp));

// Handle incoming nREPL message
bencode_value_t *nrepl_handle_message(nrepl_server_t *server, bencode_value_t *msg);

// Helper to create response message
bencode_value_t *nrepl_create_response(const char *id, const char *session);

// Session management
const char *nrepl_create_session(nrepl_server_t *server);
int nrepl_close_session(nrepl_server_t *server, const char *session_id);

#endif
