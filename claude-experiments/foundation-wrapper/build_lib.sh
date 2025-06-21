#!/bin/bash

# Build the Swift wrapper as a dynamic library first
echo "Building Swift wrapper as dynamic library..."

/Applications/Xcode-beta.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/swiftc \
  -sdk /Applications/Xcode-beta.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk \
  -framework Foundation \
  -framework FoundationModels \
  -emit-library \
  -emit-objc-header \
  -emit-objc-header-path FoundationModelsWrapper-Swift.h \
  -module-name FoundationModelsWrapper \
  FoundationModelsWrapper.swift \
  -o libFoundationModelsWrapper.dylib

echo "Swift library built successfully"

# Now build the Objective-C wrapper that uses the Swift library
echo "Building Objective-C wrapper..."

clang \
  -isysroot /Applications/Xcode-beta.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk \
  -framework Foundation \
  -framework FoundationModels \
  -fobjc-arc \
  -I. \
  -L. \
  -lFoundationModelsWrapper \
  -c foundation_models.m \
  -o foundation_models.o

echo "Objective-C wrapper compiled"

# Build the C test
echo "Building C test..."

clang \
  -isysroot /Applications/Xcode-beta.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk \
  -c test_foundation.c \
  -o test_foundation.o

echo "C test compiled"

# Link everything together
echo "Linking final executable..."

clang \
  -isysroot /Applications/Xcode-beta.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk \
  -framework Foundation \
  -framework FoundationModels \
  -fobjc-arc \
  -L. \
  -lFoundationModelsWrapper \
  foundation_models.o \
  test_foundation.o \
  -o test_foundation

echo "Build complete! Run ./test_foundation to test the C wrapper"