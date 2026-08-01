#ifndef HARNESS_NATIVE_H
#define HARNESS_NATIVE_H

#include <stddef.h>
#include <stdint.h>

typedef struct HarnessBuffer {
  unsigned char *data;
  int64_t len;
} HarnessBuffer;

typedef struct HarnessHttpResponse {
  int64_t transport_code;
  int64_t status_code;
  HarnessBuffer body;
  HarnessBuffer error;
} HarnessHttpResponse;

typedef size_t (*HarnessChunkCallback)(const unsigned char *data,
                                       size_t len,
                                       void *context);

/*
 * Perform an HTTP POST. Headers is a NULL-terminated array of complete header
 * lines. Response buffers are malloc-owned and must be released with
 * harness_buffer_free. A non-zero transport_code is a libcurl CURLcode.
 */
HarnessHttpResponse harness_http_post(const char *url,
                                      const char *const *headers,
                                      const unsigned char *body,
                                      int64_t body_len,
                                      int64_t timeout_ms,
                                      HarnessChunkCallback on_chunk,
                                      void *callback_context);

void harness_buffer_free(HarnessBuffer buffer);
int64_t harness_now_ms(void);
void harness_sleep_ms(int64_t milliseconds);
const char *harness_getenv(const char *name);
HarnessBuffer harness_current_dir(void);

/* Write one complete monitoring/output record without stdio buffering. */
int64_t harness_write_fd(int64_t fd, const unsigned char *data, int64_t len);

typedef struct HarnessProcess {
  int64_t pid;
  int64_t input_fd;
  int64_t output_fd;
  void *reader;
} HarnessProcess;

typedef struct HarnessProcessResult {
  int64_t code;
  HarnessProcess process;
  HarnessBuffer error;
} HarnessProcessResult;

/* Spawn `codex app-server --listen stdio://` with bidirectional JSONL pipes. */
HarnessProcessResult harness_codex_spawn(void);

/* Writes exactly one JSON line, adding a trailing newline when absent. */
int64_t harness_process_write_line(HarnessProcess *process,
                                   const unsigned char *data,
                                   int64_t len);

/* Returns 1 with a malloc-owned line, 0 on EOF, and a negative errno on error. */
int64_t harness_process_read_line(HarnessProcess *process,
                                  HarnessBuffer *line,
                                  int64_t timeout_ms);

/* Closes pipes, terminates a still-running child, and reaps it. */
int64_t harness_process_close(HarnessProcess *process);

#endif
