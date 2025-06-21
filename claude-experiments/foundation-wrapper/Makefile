CC = clang
SWIFTC = /Applications/Xcode-beta.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/swiftc
XCODE_PATH = /Applications/Xcode-beta.app/Contents/Developer
SDK_PATH = $(XCODE_PATH)/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
CFLAGS = -Wall -Wextra -std=c99 -isysroot $(SDK_PATH)
OBJCFLAGS = -fobjc-arc -isysroot $(SDK_PATH)
SWIFTFLAGS = -sdk $(SDK_PATH) -framework Foundation -framework FoundationModels
FRAMEWORKS = -framework Foundation -framework FoundationModels
TARGET = test_foundation
SWIFT_LIB = libFoundationModelsWrapper.dylib
SWIFT_HEADER = FoundationModelsWrapper-Swift.h

# Default target
all: $(TARGET)

# Build using the working approach
$(TARGET): $(SWIFT_LIB) foundation_models.o test_foundation.o
	$(CC) $(OBJCFLAGS) $(FRAMEWORKS) -L. -lFoundationModelsWrapper foundation_models.o test_foundation.o -o $@

# Build Swift library with generated header
$(SWIFT_LIB): FoundationModelsWrapper.swift
	$(SWIFTC) $(SWIFTFLAGS) -emit-library -emit-objc-header -emit-objc-header-path $(SWIFT_HEADER) -module-name FoundationModelsWrapper $< -o $@

# Alternative individual compilation (not used by default)
individual: $(OBJECTS)
	$(CC) $(OBJCFLAGS) $(FRAMEWORKS) -o $(TARGET) $^

# Compile Swift wrapper
FoundationModelsWrapper.o: FoundationModelsWrapper.swift
	$(SWIFTC) $(SWIFTFLAGS) $(FRAMEWORKS) -c $< -o $@

# Compile Objective-C implementation (depends on Swift header)
foundation_models.o: foundation_models.m foundation_models.h $(SWIFT_HEADER)
	$(CC) $(CFLAGS) $(OBJCFLAGS) $(FRAMEWORKS) -I. -c $< -o $@

# Compile C test file
test_foundation.o: test_foundation.c foundation_models.h
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f foundation_models.o test_foundation.o $(TARGET) $(SWIFT_LIB) $(SWIFT_HEADER)

# Test with library path
test: $(TARGET)
	DYLD_LIBRARY_PATH=. ./$(TARGET)

# Install headers (optional)
install: foundation_models.h
	mkdir -p /usr/local/include/foundation_wrapper
	cp foundation_models.h /usr/local/include/foundation_wrapper/

# Uninstall headers
uninstall:
	rm -rf /usr/local/include/foundation_wrapper

# Run the test
test: $(TARGET)
	./$(TARGET)

# Debug build
debug: CFLAGS += -g -DDEBUG
debug: $(TARGET)

# Release build
release: CFLAGS += -O2 -DNDEBUG
release: $(TARGET)

# Check syntax
check:
	$(CC) $(CFLAGS) -fsyntax-only foundation_models.m
	$(CC) $(CFLAGS) -fsyntax-only test_foundation.c

.PHONY: all clean install uninstall test debug release check