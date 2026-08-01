#define _POSIX_C_SOURCE 200809L

#include "harness_native.h"

#include <curl/curl.h>
#include <errno.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

typedef struct HttpWriteContext {
  HarnessBuffer buffer;
  HarnessChunkCallback callback;
  void *callback_context;
  int allocation_failed;
} HttpWriteContext;

static HarnessBuffer copy_text(const char *text) {
  HarnessBuffer result = {0};
  if (text == NULL) {
    return result;
  }
  size_t len = strlen(text);
  result.data = malloc(len + 1);
  if (result.data == NULL) {
    return result;
  }
  memcpy(result.data, text, len + 1);
  result.len = (int64_t)len;
  return result;
}

static size_t collect_http_body(char *data,
                                size_t size,
                                size_t count,
                                void *opaque) {
  HttpWriteContext *context = opaque;
  size_t len = size * count;

  if (context->callback != NULL) {
    size_t consumed = context->callback((const unsigned char *)data,
                                        len,
                                        context->callback_context);
    if (consumed != len) {
      return consumed;
    }
  }

  size_t old_len = (size_t)context->buffer.len;
  unsigned char *next = realloc(context->buffer.data, old_len + len + 1);
  if (next == NULL) {
    context->allocation_failed = 1;
    return 0;
  }
  context->buffer.data = next;
  memcpy(next + old_len, data, len);
  next[old_len + len] = 0;
  context->buffer.len = (int64_t)(old_len + len);
  return len;
}

HarnessHttpResponse harness_http_post(const char *url,
                                      const char *const *headers,
                                      const unsigned char *body,
                                      int64_t body_len,
                                      int64_t timeout_ms,
                                      HarnessChunkCallback on_chunk,
                                      void *callback_context) {
  HarnessHttpResponse result = {0};
  CURL *curl = curl_easy_init();
  if (curl == NULL) {
    result.transport_code = CURLE_FAILED_INIT;
    result.error = copy_text("curl_easy_init failed");
    return result;
  }

  struct curl_slist *header_list = NULL;
  if (headers != NULL) {
    for (size_t i = 0; headers[i] != NULL; ++i) {
      struct curl_slist *next = curl_slist_append(header_list, headers[i]);
      if (next == NULL) {
        curl_slist_free_all(header_list);
        curl_easy_cleanup(curl);
        result.transport_code = CURLE_OUT_OF_MEMORY;
        result.error = copy_text("could not allocate HTTP header list");
        return result;
      }
      header_list = next;
    }
  }

  HttpWriteContext write_context = {
      .callback = on_chunk,
      .callback_context = callback_context,
  };
  char curl_error[CURL_ERROR_SIZE] = {0};

  curl_easy_setopt(curl, CURLOPT_URL, url);
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, header_list);
  curl_easy_setopt(curl, CURLOPT_POST, 1L);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE_LARGE, (curl_off_t)body_len);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, collect_http_body);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &write_context);
  curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, curl_error);
  curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, timeout_ms);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, timeout_ms);

  CURLcode code = curl_easy_perform(curl);
  long status = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);

  result.transport_code = write_context.allocation_failed
                              ? CURLE_OUT_OF_MEMORY
                              : (int64_t)code;
  result.status_code = (int64_t)status;
  result.body = write_context.buffer;
  if (result.transport_code != CURLE_OK) {
    const char *message = curl_error[0] != '\0'
                              ? curl_error
                              : curl_easy_strerror((CURLcode)result.transport_code);
    result.error = copy_text(message);
  }

  curl_slist_free_all(header_list);
  curl_easy_cleanup(curl);
  return result;
}

void harness_buffer_free(HarnessBuffer buffer) {
  free(buffer.data);
}

int64_t harness_now_ms(void) {
  struct timespec now = {0};
  if (clock_gettime(CLOCK_REALTIME, &now) != 0) {
    return 0;
  }
  return (int64_t)now.tv_sec * 1000 + now.tv_nsec / 1000000;
}

void harness_sleep_ms(int64_t milliseconds) {
  if (milliseconds <= 0) return;
  struct timespec requested = {
      .tv_sec = (time_t)(milliseconds / 1000),
      .tv_nsec = (long)((milliseconds % 1000) * 1000000),
  };
  while (nanosleep(&requested, &requested) != 0 && errno == EINTR) {
  }
}

const char *harness_getenv(const char *name) {
  return getenv(name);
}

static int write_all(int fd, const unsigned char *data, size_t len);

int64_t harness_write_fd(int64_t fd,
                         const unsigned char *data,
                         int64_t len) {
  if (fd < 0 || len < 0 || (data == NULL && len != 0)) return -EINVAL;
  return write_all((int)fd, data, (size_t)len);
}

HarnessBuffer harness_current_dir(void) {
  HarnessBuffer result = {0};
  char *directory = getcwd(NULL, 0);
  if (directory == NULL) return result;
  result.data = (unsigned char *)directory;
  result.len = (int64_t)strlen(directory);
  return result;
}

HarnessProcessResult harness_codex_spawn(void) {
  HarnessProcessResult result = {0};
  int input_pipe[2] = {-1, -1};
  int output_pipe[2] = {-1, -1};

  if (pipe(input_pipe) != 0 || pipe(output_pipe) != 0) {
    int saved_errno = errno;
    if (input_pipe[0] >= 0) close(input_pipe[0]);
    if (input_pipe[1] >= 0) close(input_pipe[1]);
    if (output_pipe[0] >= 0) close(output_pipe[0]);
    if (output_pipe[1] >= 0) close(output_pipe[1]);
    result.code = -saved_errno;
    result.error = copy_text(strerror(saved_errno));
    return result;
  }

  pid_t pid = fork();
  if (pid < 0) {
    int saved_errno = errno;
    close(input_pipe[0]);
    close(input_pipe[1]);
    close(output_pipe[0]);
    close(output_pipe[1]);
    result.code = -saved_errno;
    result.error = copy_text(strerror(saved_errno));
    return result;
  }

  if (pid == 0) {
    (void)dup2(input_pipe[0], STDIN_FILENO);
    (void)dup2(output_pipe[1], STDOUT_FILENO);
    close(input_pipe[0]);
    close(input_pipe[1]);
    close(output_pipe[0]);
    close(output_pipe[1]);
    execlp("codex", "codex", "app-server", "--listen", "stdio://", NULL);
    _exit(127);
  }

  close(input_pipe[0]);
  close(output_pipe[1]);
  FILE *reader = fdopen(output_pipe[0], "r");
  if (reader == NULL) {
    int saved_errno = errno;
    close(input_pipe[1]);
    close(output_pipe[0]);
    kill(pid, SIGTERM);
    (void)waitpid(pid, NULL, 0);
    result.code = -saved_errno;
    result.error = copy_text(strerror(saved_errno));
    return result;
  }
  (void)setvbuf(reader, NULL, _IONBF, 0);

  result.process.pid = pid;
  result.process.input_fd = input_pipe[1];
  result.process.output_fd = output_pipe[0];
  result.process.reader = reader;
  return result;
}

static int write_all(int fd, const unsigned char *data, size_t len) {
  while (len > 0) {
    ssize_t written = write(fd, data, len);
    if (written < 0) {
      if (errno == EINTR) continue;
      return -errno;
    }
    data += written;
    len -= (size_t)written;
  }
  return 0;
}

int64_t harness_process_write_line(HarnessProcess *process,
                                   const unsigned char *data,
                                   int64_t len) {
  if (process == NULL || process->input_fd < 0 || len < 0) {
    return -EINVAL;
  }
  int result = write_all((int)process->input_fd, data, (size_t)len);
  if (result != 0) return result;
  if (len == 0 || data[len - 1] != '\n') {
    const unsigned char newline = '\n';
    result = write_all((int)process->input_fd, &newline, 1);
  }
  return result;
}

int64_t harness_process_read_line(HarnessProcess *process,
                                  HarnessBuffer *line,
                                  int64_t timeout_ms) {
  if (process == NULL || line == NULL || process->reader == NULL) {
    return -EINVAL;
  }
  struct pollfd descriptor = {
      .fd = (int)process->output_fd,
      .events = POLLIN | POLLHUP,
  };
  int poll_result;
  do {
    poll_result = poll(&descriptor, 1, (int)timeout_ms);
  } while (poll_result < 0 && errno == EINTR);
  if (poll_result == 0) return -ETIMEDOUT;
  if (poll_result < 0) return -errno;
  char *data = NULL;
  size_t capacity = 0;
  errno = 0;
  ssize_t len = getline(&data, &capacity, (FILE *)process->reader);
  if (len < 0) {
    int result = feof((FILE *)process->reader) ? 0 : -errno;
    free(data);
    return result;
  }
  if (len > 0 && data[len - 1] == '\n') --len;
  data[len] = '\0';
  line->data = (unsigned char *)data;
  line->len = (int64_t)len;
  return 1;
}

int64_t harness_process_close(HarnessProcess *process) {
  if (process == NULL) return -EINVAL;
  if (process->input_fd >= 0) {
    close((int)process->input_fd);
    process->input_fd = -1;
  }
  if (process->reader != NULL) {
    fclose((FILE *)process->reader);
    process->reader = NULL;
    process->output_fd = -1;
  } else if (process->output_fd >= 0) {
    close((int)process->output_fd);
    process->output_fd = -1;
  }

  int status = 0;
  pid_t wait_result = waitpid((pid_t)process->pid, &status, WNOHANG);
  if (wait_result == 0) {
    kill((pid_t)process->pid, SIGTERM);
    wait_result = waitpid((pid_t)process->pid, &status, 0);
  }
  process->pid = -1;
  if (wait_result < 0) return -errno;
  if (WIFEXITED(status)) return WEXITSTATUS(status);
  if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
  return 0;
}
