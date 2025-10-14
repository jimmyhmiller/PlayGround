#include "nrepl.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// Global evaluator function pointer
static nrepl_eval_fn global_evaluator = NULL;

void nrepl_init(nrepl_server_t *server) {
    server->session_count = 0;
    for (int i = 0; i < MAX_SESSIONS; i++) {
        server->sessions[i].active = 0;
    }
}

void nrepl_set_evaluator(nrepl_eval_fn eval_fn) {
    global_evaluator = eval_fn;
}

const char *nrepl_create_session(nrepl_server_t *server) {
    if (server->session_count >= MAX_SESSIONS) {
        return NULL;
    }

    uuid_t uuid;
    uuid_generate(uuid);

    for (int i = 0; i < MAX_SESSIONS; i++) {
        if (!server->sessions[i].active) {
            uuid_unparse(uuid, server->sessions[i].id);
            server->sessions[i].active = 1;
            server->session_count++;
            return server->sessions[i].id;
        }
    }

    return NULL;
}

int nrepl_close_session(nrepl_server_t *server, const char *session_id) {
    for (int i = 0; i < MAX_SESSIONS; i++) {
        if (server->sessions[i].active &&
            strcmp(server->sessions[i].id, session_id) == 0) {
            server->sessions[i].active = 0;
            server->session_count--;
            return 1;
        }
    }
    return 0;
}

bencode_value_t *nrepl_create_response(const char *id, const char *session) {
    bencode_value_t *resp = bencode_create_dict();

    if (id) {
        bencode_dict_set(resp, "id", bencode_create_str(id, strlen(id)));
    }

    if (session) {
        bencode_dict_set(resp, "session", bencode_create_str(session, strlen(session)));
    }

    return resp;
}

static bencode_value_t *handle_describe(nrepl_server_t *server, bencode_value_t *msg) {
    (void)server; // Unused parameter
    bencode_value_t *id = bencode_dict_get(msg, "id");
    bencode_value_t *session = bencode_dict_get(msg, "session");

    const char *id_str = id ? id->str_val.data : NULL;
    const char *session_str = session ? session->str_val.data : NULL;

    bencode_value_t *resp = nrepl_create_response(id_str, session_str);

    // Add versions
    bencode_value_t *versions = bencode_create_dict();
    bencode_dict_set(versions, "nrepl-c", bencode_create_str("0.1.0", 5));
    bencode_dict_set(resp, "versions", versions);

    // Add ops
    bencode_value_t *ops = bencode_create_dict();
    bencode_dict_set(ops, "describe", bencode_create_dict());
    bencode_dict_set(ops, "clone", bencode_create_dict());
    bencode_dict_set(ops, "close", bencode_create_dict());
    bencode_dict_set(ops, "eval", bencode_create_dict());
    bencode_dict_set(resp, "ops", ops);

    // Add status
    bencode_value_t *status = bencode_create_list();
    bencode_list_append(status, bencode_create_str("done", 4));
    bencode_dict_set(resp, "status", status);

    return resp;
}

static bencode_value_t *handle_clone(nrepl_server_t *server, bencode_value_t *msg) {
    bencode_value_t *id = bencode_dict_get(msg, "id");
    const char *id_str = id ? id->str_val.data : NULL;

    const char *new_session = nrepl_create_session(server);
    if (!new_session) {
        bencode_value_t *resp = nrepl_create_response(id_str, NULL);
        bencode_value_t *status = bencode_create_list();
        bencode_list_append(status, bencode_create_str("error", 5));
        bencode_dict_set(resp, "status", status);
        return resp;
    }

    bencode_value_t *resp = nrepl_create_response(id_str, new_session);
    bencode_dict_set(resp, "new-session", bencode_create_str(new_session, strlen(new_session)));

    bencode_value_t *status = bencode_create_list();
    bencode_list_append(status, bencode_create_str("done", 4));
    bencode_dict_set(resp, "status", status);

    return resp;
}

static bencode_value_t *handle_close(nrepl_server_t *server, bencode_value_t *msg) {
    bencode_value_t *id = bencode_dict_get(msg, "id");
    bencode_value_t *session = bencode_dict_get(msg, "session");

    const char *id_str = id ? id->str_val.data : NULL;
    const char *session_str = session ? session->str_val.data : NULL;

    bencode_value_t *resp = nrepl_create_response(id_str, session_str);

    if (session_str && nrepl_close_session(server, session_str)) {
        bencode_value_t *status = bencode_create_list();
        bencode_list_append(status, bencode_create_str("done", 4));
        bencode_dict_set(resp, "status", status);
    } else {
        bencode_value_t *status = bencode_create_list();
        bencode_list_append(status, bencode_create_str("error", 5));
        bencode_dict_set(resp, "status", status);
    }

    return resp;
}

// Helper to send multiple responses
typedef struct {
    bencode_value_t **responses;
    int count;
} multi_response_t;

static multi_response_t* create_multi_response() {
    multi_response_t *mr = malloc(sizeof(multi_response_t));
    mr->responses = NULL;
    mr->count = 0;
    return mr;
}

static void add_response(multi_response_t *mr, bencode_value_t *resp) {
    mr->responses = realloc(mr->responses, (mr->count + 1) * sizeof(bencode_value_t*));
    mr->responses[mr->count++] = resp;
}

// Store callback for sending multiple responses
static void (*response_callback)(bencode_value_t *resp) = NULL;

void nrepl_set_response_callback(void (*callback)(bencode_value_t *resp)) {
    response_callback = callback;
}

static bencode_value_t *handle_eval(nrepl_server_t *server, bencode_value_t *msg) {
    (void)server; // Unused parameter
    bencode_value_t *id = bencode_dict_get(msg, "id");
    bencode_value_t *session = bencode_dict_get(msg, "session");
    bencode_value_t *code = bencode_dict_get(msg, "code");

    const char *id_str = id ? id->str_val.data : NULL;
    const char *session_str = session ? session->str_val.data : NULL;

    // Send value response first if we have a result
    if (code && code->str_val.data) {
        if (global_evaluator) {
            char *result = global_evaluator(code->str_val.data);
            if (result && response_callback) {
                // Send value response
                bencode_value_t *value_resp = nrepl_create_response(id_str, session_str);
                bencode_dict_set(value_resp, "value", bencode_create_str(result, strlen(result)));
                bencode_dict_set(value_resp, "ns", bencode_create_str("user", 4));
                response_callback(value_resp);
                bencode_free(value_resp);
                free(result);
            } else if (result) {
                // Single response mode
                bencode_value_t *resp = nrepl_create_response(id_str, session_str);
                bencode_dict_set(resp, "value", bencode_create_str(result, strlen(result)));
                bencode_dict_set(resp, "ns", bencode_create_str("user", 4));
                bencode_value_t *status = bencode_create_list();
                bencode_list_append(status, bencode_create_str("done", 4));
                bencode_dict_set(resp, "status", status);
                free(result);
                return resp;
            }
        }
    }

    // Send done response
    bencode_value_t *done_resp = nrepl_create_response(id_str, session_str);
    bencode_value_t *status = bencode_create_list();
    bencode_list_append(status, bencode_create_str("done", 4));
    bencode_dict_set(done_resp, "status", status);

    return done_resp;
}

bencode_value_t *nrepl_handle_message(nrepl_server_t *server, bencode_value_t *msg) {
    if (msg->type != BENCODE_DICT) {
        return NULL;
    }

    bencode_value_t *op = bencode_dict_get(msg, "op");
    if (!op || op->type != BENCODE_STR) {
        return NULL;
    }

    const char *op_str = op->str_val.data;

    if (strcmp(op_str, "describe") == 0) {
        return handle_describe(server, msg);
    } else if (strcmp(op_str, "clone") == 0) {
        return handle_clone(server, msg);
    } else if (strcmp(op_str, "close") == 0) {
        return handle_close(server, msg);
    } else if (strcmp(op_str, "eval") == 0) {
        return handle_eval(server, msg);
    }

    // Unknown operation
    bencode_value_t *id = bencode_dict_get(msg, "id");
    bencode_value_t *session = bencode_dict_get(msg, "session");

    const char *id_str = id ? id->str_val.data : NULL;
    const char *session_str = session ? session->str_val.data : NULL;

    bencode_value_t *resp = nrepl_create_response(id_str, session_str);
    bencode_value_t *status = bencode_create_list();
    bencode_list_append(status, bencode_create_str("error", 5));
    bencode_list_append(status, bencode_create_str("unknown-op", 10));
    bencode_dict_set(resp, "status", status);

    return resp;
}
