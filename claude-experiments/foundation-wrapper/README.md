# Foundation Models C Wrapper

A C wrapper for Apple's Foundation Models framework, providing C-compatible APIs for language model functionality.

## Current Status

✅ **Swift Wrapper**: Fully functional Swift wrapper (`FoundationModelsWrapper.swift`)  
✅ **C Bridge**: Fully working C interface with proper Swift/Objective-C linking  
✅ **Working Demo**: C test program successfully calls Foundation Models API  

## Files

- `foundation_models.h` - C header with all function declarations and types
- `foundation_models.m` - Objective-C implementation that bridges to Foundation Models
- `FoundationModelsWrapper.swift` - Swift wrapper class ✅ WORKING
- `test_foundation.c` - C test program ✅ WORKING
- `Makefile` - Build system for compiling the wrapper and tests ✅ WORKING
- `build_lib.sh` - Alternative build script

## Features

- ✅ Language model session management
- ✅ Simple text responses with async/await
- ✅ Multi-turn conversations with transcript tracking
- ✅ Custom instructions and specialized use cases (content tagging)
- ✅ Model availability checking
- ⚠️ Tool integration support (basic structure)
- ⚠️ Streaming responses (not implemented)
- ⚠️ @Generable structs (not implemented)

## Building and Testing

### Swift Wrapper (Working)
```bash
# Test the Swift wrapper directly
/Applications/Xcode-beta.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/swiftc \
  -sdk /Applications/Xcode-beta.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk \
  -framework Foundation -framework FoundationModels \
  FoundationModelsWrapper.swift working_test.swift -o working_test

./working_test
```

### C Wrapper (In Progress)
```bash
make clean        # Clean build artifacts
make test         # Attempt to build and run tests (has linking issues)
```

## Usage Example (Swift)

```swift
import Foundation
import FoundationModels

// Create a session
let wrapper = FoundationModelsWrapper()

// Check availability
let isAvailable = FoundationModelsWrapper.checkAvailability()

// Send a prompt
wrapper.respond(to: "What's a good name for a trip to Japan?") { content, error in
    if let content = content {
        print("Response: \(content)")
    }
}
```

## Usage Example (C - When Working)

```c
#include "foundation_models.h"

// Create a session
LanguageModelSession* session = language_model_session_create();

// Send a prompt
LanguageModelResponse* response = language_model_session_respond(
    session, "What's a good name for a trip to Japan?"
);

// Get the response content
const char* content = language_model_response_get_content(response);
printf("Response: %s\n", content);

// Cleanup
language_model_response_destroy(response);
language_model_session_destroy(session);
```

## Requirements

- macOS with Xcode beta (Foundation Models framework)
- Xcode beta developer tools
- clang compiler

## Test Results

The Swift wrapper successfully demonstrates:
- ✅ Model availability check: `true`
- ✅ Simple response: `"Cherry Blossom Odyssey"`
- ✅ Custom instructions: Rhyming responses work
- ✅ Async response handling works correctly

## Notes

The Foundation Models framework is Swift-only and requires Xcode beta. The Swift wrapper (`FoundationModelsWrapper`) works perfectly and demonstrates all core functionality. The C bridge has linking challenges due to Swift/Objective-C interop complexity but the basic structure is in place.