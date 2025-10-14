# nREPL-C

A minimal nREPL server implementation in C that can be used as a library.

## Features

- ✓ Core nREPL protocol support (describe, clone, close, eval)
- ✓ Bencode encoding/decoding
- ✓ Session management with UUIDs
- ✓ Concurrent client connections (fork-based)
- ✓ Custom evaluator support - plug in your own language!
- ✓ Compatible with standard nREPL clients (rep, reply, official Clojure client)

## Building

```bash
make
```

## Running the Server

The default server includes a simple `(+ num num)` evaluator:

```bash
./nrepl-server          # Runs on port 7888
./nrepl-server 8888     # Runs on port 8888
```

## Testing

Run the comprehensive test suite:

```bash
python3 test_nrepl.py
```

Test the evaluator:

```bash
python3 test_eval.py
```

## Using as a Library

You can use nREPL-C as a library with your own custom evaluator!

### Example: Custom Evaluator

```c
#include "nrepl.h"

// Your custom evaluator function
char* my_evaluator(const char *code) {
    // Evaluate code and return result
    // Must return malloc'd string or NULL
    return strdup("42");
}

int main() {
    nrepl_server_t server;
    nrepl_init(&server);

    // Set your custom evaluator
    nrepl_set_evaluator(my_evaluator);

    // ... setup socket and handle connections ...
}
```

See `example_custom_eval.c` for a complete example.

### Building Your Custom Server

```bash
gcc -o my-server my_server.c nrepl.c bencode.c -luuid
```

## Connecting with Clients

### Using reply

```bash
reply --attach 7888
```

Then try:
```clojure
(+ 2 2)
; => 4
```

### Using rep

```bash
echo '(+ 2 2)' | rep -p 7888
```

### Using Python

```python
import socket

s = socket.socket()
s.connect(("localhost", 7888))

# Send bencode message
msg = b'd2:op4:eval4:code7:(+ 2 2)e'
s.sendall(msg)

# Receive response
print(s.recv(4096))
```

## API Reference

### Core Functions

```c
// Initialize server
void nrepl_init(nrepl_server_t *server);

// Set custom evaluator
typedef char* (*nrepl_eval_fn)(const char *code);
void nrepl_set_evaluator(nrepl_eval_fn eval_fn);

// Handle nREPL message
bencode_value_t *nrepl_handle_message(nrepl_server_t *server, bencode_value_t *msg);

// Session management
const char *nrepl_create_session(nrepl_server_t *server);
int nrepl_close_session(nrepl_server_t *server, const char *session_id);
```

### Evaluator Function

Your custom evaluator should have this signature:

```c
char* my_evaluator(const char *code);
```

- **Input**: `code` - the code string to evaluate
- **Output**: A malloc'd string with the result, or NULL for nil
- **Memory**: The caller (nREPL) will free the returned string

## Project Structure

```
nrepl-c/
├── bencode.c/h         # Bencode encoding/decoding
├── nrepl.c/h           # Core nREPL protocol implementation
├── simple_eval.c/h     # Example evaluator for (+ num num)
├── server.c            # Main server with simple evaluator
├── example_custom_eval.c  # Example custom evaluator
├── test_nrepl.py       # Comprehensive test suite
└── Makefile
```

## Supported nREPL Operations

- **describe** - Returns server capabilities and version
- **clone** - Creates a new session
- **close** - Closes a session
- **eval** - Evaluates code (uses your custom evaluator)

## Implementation Details

- **Bencode**: Full bencode support (integers, strings, lists, dictionaries)
- **Sessions**: UUID-based session IDs, supports up to 100 concurrent sessions
- **Concurrency**: Fork-based client handling (one process per client)
- **Protocol**: Follows nREPL protocol specification

## Future Enhancements

- Thread-based concurrency instead of fork-based
- Additional nREPL middleware (interrupt, load-file, etc.)
- SSL/TLS support
- Logging and debugging middleware

## License

MIT

## Contributing

Feel free to submit issues and pull requests!
