#include "llhttp.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define COIL_HTTP_MAX_HEADERS 100
#define COIL_HTTP_MAX_HEADER_BYTES (64 * 1024)
#define COIL_HTTP_MAX_BODY_BYTES (8 * 1024 * 1024)
#define COIL_HTTP_ERROR_LIMIT 1000
#define COIL_HTTP_ERROR_OOM 1001

typedef struct {
  const char* name;
  size_t name_len;
  const char* value;
  size_t value_len;
} coil_http_header;

typedef struct {
  llhttp_t parser;
  llhttp_settings_t settings;
  const char* input;
  size_t input_len;
  const char* method;
  size_t method_len;
  const char* target;
  size_t target_len;
  coil_http_header headers[COIL_HTTP_MAX_HEADERS];
  size_t header_count;
  size_t header_bytes;
  unsigned char* body;
  size_t body_len;
  size_t body_capacity;
  size_t consumed;
  size_t error_offset;
  int status;
  int error_code;
  int complete;
  int keep_alive;
} coil_http_result;

static coil_http_result* result_of(llhttp_t* parser) {
  return (coil_http_result*) parser->data;
}

static int append_span(const char** start, size_t* total,
                       const char* at, size_t length) {
  if (*start == NULL) {
    *start = at;
    *total = length;
    return 0;
  }
  if (*start + *total != at) return HPE_USER;
  *total += length;
  return 0;
}

static int on_method(llhttp_t* parser, const char* at, size_t length) {
  coil_http_result* out = result_of(parser);
  return append_span(&out->method, &out->method_len, at, length);
}

static int on_url(llhttp_t* parser, const char* at, size_t length) {
  coil_http_result* out = result_of(parser);
  return append_span(&out->target, &out->target_len, at, length);
}

static int on_header_field(llhttp_t* parser, const char* at, size_t length) {
  coil_http_result* out = result_of(parser);
  if (out->header_count >= COIL_HTTP_MAX_HEADERS ||
      out->header_bytes + length > COIL_HTTP_MAX_HEADER_BYTES) {
    out->error_code = COIL_HTTP_ERROR_LIMIT;
    return HPE_USER;
  }
  out->header_bytes += length;
  return append_span(&out->headers[out->header_count].name,
                     &out->headers[out->header_count].name_len, at, length);
}

static int on_header_value(llhttp_t* parser, const char* at, size_t length) {
  coil_http_result* out = result_of(parser);
  if (out->header_count >= COIL_HTTP_MAX_HEADERS ||
      out->header_bytes + length > COIL_HTTP_MAX_HEADER_BYTES) {
    out->error_code = COIL_HTTP_ERROR_LIMIT;
    return HPE_USER;
  }
  out->header_bytes += length;
  return append_span(&out->headers[out->header_count].value,
                     &out->headers[out->header_count].value_len, at, length);
}

static int on_header_value_complete(llhttp_t* parser) {
  coil_http_result* out = result_of(parser);
  if (out->header_count >= COIL_HTTP_MAX_HEADERS) {
    out->error_code = COIL_HTTP_ERROR_LIMIT;
    return HPE_USER;
  }
  out->header_count++;
  return 0;
}

static int on_body(llhttp_t* parser, const char* at, size_t length) {
  coil_http_result* out = result_of(parser);
  size_t needed;
  size_t capacity;
  unsigned char* grown;

  if (length > COIL_HTTP_MAX_BODY_BYTES - out->body_len) {
    out->error_code = COIL_HTTP_ERROR_LIMIT;
    return HPE_USER;
  }
  needed = out->body_len + length;
  if (needed > out->body_capacity) {
    capacity = out->body_capacity == 0 ? 256 : out->body_capacity;
    while (capacity < needed) capacity *= 2;
    if (capacity > COIL_HTTP_MAX_BODY_BYTES) capacity = COIL_HTTP_MAX_BODY_BYTES;
    grown = (unsigned char*) realloc(out->body, capacity);
    if (grown == NULL) {
      out->error_code = COIL_HTTP_ERROR_OOM;
      return HPE_USER;
    }
    out->body = grown;
    out->body_capacity = capacity;
  }
  memcpy(out->body + out->body_len, at, length);
  out->body_len = needed;
  return 0;
}

static int on_headers_complete(llhttp_t* parser) {
  result_of(parser)->keep_alive = llhttp_should_keep_alive(parser);
  return 0;
}

static int on_message_complete(llhttp_t* parser) {
  result_of(parser)->complete = 1;
  return HPE_PAUSED;
}

void* coil_llhttp_parse_request(const unsigned char* data, size_t length) {
  coil_http_result* out = (coil_http_result*) calloc(1, sizeof(*out));
  llhttp_errno_t err;
  const char* pos;
  if (out == NULL) return NULL;

  out->input = (const char*) data;
  out->input_len = length;
  llhttp_settings_init(&out->settings);
  out->settings.on_method = on_method;
  out->settings.on_url = on_url;
  out->settings.on_header_field = on_header_field;
  out->settings.on_header_value = on_header_value;
  out->settings.on_header_value_complete = on_header_value_complete;
  out->settings.on_headers_complete = on_headers_complete;
  out->settings.on_body = on_body;
  out->settings.on_message_complete = on_message_complete;
  llhttp_init(&out->parser, HTTP_REQUEST, &out->settings);
  out->parser.data = out;

  err = llhttp_execute(&out->parser, (const char*) data, length);
  pos = llhttp_get_error_pos(&out->parser);
  if (pos != NULL && pos >= (const char*) data && pos <= (const char*) data + length) {
    out->error_offset = (size_t) (pos - (const char*) data);
  } else {
    out->error_offset = length;
  }

  if (out->complete && err == HPE_PAUSED) {
    out->status = 0;
    out->consumed = out->error_offset;
  } else if (err == HPE_OK) {
    out->status = 1;
    out->error_code = HPE_INVALID_EOF_STATE;
  } else {
    out->status = out->error_code >= COIL_HTTP_ERROR_LIMIT ? 3 : 2;
    if (out->error_code == 0) out->error_code = (int) err;
  }
  return out;
}

void coil_llhttp_result_free(void* opaque) {
  coil_http_result* out = (coil_http_result*) opaque;
  if (out == NULL) return;
  free(out->body);
  free(out);
}

int64_t coil_llhttp_status(void* opaque) { return ((coil_http_result*) opaque)->status; }
int64_t coil_llhttp_error_code(void* opaque) { return ((coil_http_result*) opaque)->error_code; }
int64_t coil_llhttp_error_offset(void* opaque) { return (int64_t) ((coil_http_result*) opaque)->error_offset; }
int64_t coil_llhttp_consumed(void* opaque) { return (int64_t) ((coil_http_result*) opaque)->consumed; }
int64_t coil_llhttp_http_major(void* opaque) { return llhttp_get_http_major(&((coil_http_result*) opaque)->parser); }
int64_t coil_llhttp_http_minor(void* opaque) { return llhttp_get_http_minor(&((coil_http_result*) opaque)->parser); }
int64_t coil_llhttp_keep_alive(void* opaque) { return ((coil_http_result*) opaque)->keep_alive; }
int64_t coil_llhttp_header_count(void* opaque) { return (int64_t) ((coil_http_result*) opaque)->header_count; }

const unsigned char* coil_llhttp_method(void* opaque) { return (const unsigned char*) ((coil_http_result*) opaque)->method; }
int64_t coil_llhttp_method_len(void* opaque) { return (int64_t) ((coil_http_result*) opaque)->method_len; }
const unsigned char* coil_llhttp_target(void* opaque) { return (const unsigned char*) ((coil_http_result*) opaque)->target; }
int64_t coil_llhttp_target_len(void* opaque) { return (int64_t) ((coil_http_result*) opaque)->target_len; }
const unsigned char* coil_llhttp_body(void* opaque) { return ((coil_http_result*) opaque)->body; }
int64_t coil_llhttp_body_len(void* opaque) { return (int64_t) ((coil_http_result*) opaque)->body_len; }

const unsigned char* coil_llhttp_header_name(void* opaque, int64_t index) {
  return (const unsigned char*) ((coil_http_result*) opaque)->headers[index].name;
}
int64_t coil_llhttp_header_name_len(void* opaque, int64_t index) {
  return (int64_t) ((coil_http_result*) opaque)->headers[index].name_len;
}
const unsigned char* coil_llhttp_header_value(void* opaque, int64_t index) {
  return (const unsigned char*) ((coil_http_result*) opaque)->headers[index].value;
}
int64_t coil_llhttp_header_value_len(void* opaque, int64_t index) {
  return (int64_t) ((coil_http_result*) opaque)->headers[index].value_len;
}
