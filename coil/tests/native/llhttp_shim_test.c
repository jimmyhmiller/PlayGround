#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

void* coil_llhttp_parse_request(const unsigned char*, size_t);
void coil_llhttp_result_free(void*);
int64_t coil_llhttp_status(void*);
int64_t coil_llhttp_error_code(void*);
int64_t coil_llhttp_consumed(void*);
int64_t coil_llhttp_keep_alive(void*);
int64_t coil_llhttp_header_count(void*);
const unsigned char* coil_llhttp_method(void*);
int64_t coil_llhttp_method_len(void*);
const unsigned char* coil_llhttp_target(void*);
int64_t coil_llhttp_target_len(void*);
const unsigned char* coil_llhttp_body(void*);
int64_t coil_llhttp_body_len(void*);
const unsigned char* coil_llhttp_header_value(void*, int64_t);
int64_t coil_llhttp_header_value_len(void*, int64_t);

static int equals(const unsigned char* actual, int64_t length, const char* expected) {
  return length == (int64_t) strlen(expected) && memcmp(actual, expected, (size_t) length) == 0;
}

static void complete_content_length(void) {
  const char* input =
    "POST /items HTTP/1.1\r\nHost: example.test\r\nContent-Length: 5\r\n\r\nhello"
    "GET /next HTTP/1.1\r\n\r\n";
  void* result = coil_llhttp_parse_request((const unsigned char*) input, strlen(input));
  assert(result != NULL);
  assert(coil_llhttp_status(result) == 0);
  assert(equals(coil_llhttp_method(result), coil_llhttp_method_len(result), "POST"));
  assert(equals(coil_llhttp_target(result), coil_llhttp_target_len(result), "/items"));
  assert(equals(coil_llhttp_body(result), coil_llhttp_body_len(result), "hello"));
  assert(coil_llhttp_header_count(result) == 2);
  assert(equals(coil_llhttp_header_value(result, 0),
                coil_llhttp_header_value_len(result, 0), "example.test"));
  assert(coil_llhttp_keep_alive(result) == 1);
  assert(coil_llhttp_consumed(result) == (int64_t) (strstr(input, "GET /next") - input));
  coil_llhttp_result_free(result);
}

static void complete_chunked(void) {
  const char* input =
    "POST /chunks HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n"
    "4\r\nWiki\r\n5\r\npedia\r\n0\r\nX-Trailer: yes\r\n\r\n";
  void* result = coil_llhttp_parse_request((const unsigned char*) input, strlen(input));
  assert(result != NULL);
  assert(coil_llhttp_status(result) == 0);
  assert(equals(coil_llhttp_body(result), coil_llhttp_body_len(result), "Wikipedia"));
  assert(coil_llhttp_consumed(result) == (int64_t) strlen(input));
  coil_llhttp_result_free(result);
}

static void rejected(const char* input) {
  void* result = coil_llhttp_parse_request((const unsigned char*) input, strlen(input));
  assert(result != NULL);
  assert(coil_llhttp_status(result) == 2);
  assert(coil_llhttp_error_code(result) > 0);
  coil_llhttp_result_free(result);
}

static void incomplete(void) {
  const char* input = "POST / HTTP/1.1\r\nContent-Length: 4\r\n\r\nabc";
  void* result = coil_llhttp_parse_request((const unsigned char*) input, strlen(input));
  assert(result != NULL);
  assert(coil_llhttp_status(result) == 1);
  assert(coil_llhttp_error_code(result) == 14);
  coil_llhttp_result_free(result);
}

int main(void) {
  complete_content_length();
  complete_chunked();
  incomplete();
  rejected("POST / HTTP/1.1\r\nContent-Length: 1\r\nContent-Length: 1\r\n\r\nx");
  rejected("POST / HTTP/1.1\r\nContent-Length: 4\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n");
  rejected("GET / HTTP/1.1\r\nBad Header: value\r\n\r\n");
  puts("llhttp shim: all checks passed");
  return 0;
}
