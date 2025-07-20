cmake_minimum_required(VERSION 3.20)
project(tensor_ops_dialect)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MLIR installation
# Note: User will need to set MLIR_DIR or install MLIR in standard location
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Set the LLVM and MLIR include directories
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

# Include MLIR and LLVM headers
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Include directories
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/cpp/include)
include_directories(${PROJECT_BINARY_DIR}/cpp/include)

# TableGen for dialect definitions
set(LLVM_TARGET_DEFINITIONS cpp/td/TensorOpsDialect.td)
mlir_tablegen(TensorOpsDialect.h.inc -gen-dialect-decls)
mlir_tablegen(TensorOpsDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(TensorOpsDialectIncGen)

set(LLVM_TARGET_DEFINITIONS cpp/td/TensorOpsOps.td)
mlir_tablegen(TensorOpsOps.h.inc -gen-op-decls)
mlir_tablegen(TensorOpsOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(TensorOpsOpsIncGen)

# TensorOps dialect library
add_mlir_library(TensorOpsDialect
  cpp/lib/TensorOpsDialect.cpp
  cpp/lib/TensorOpsOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/cpp/include/TensorOps

  DEPENDS
  TensorOpsDialectIncGen
  TensorOpsOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)

# C API library
add_mlir_library(TensorOpsAPI
  capi/TensorOpsAPI.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/capi

  DEPENDS
  TensorOpsDialect

  LINK_LIBS PUBLIC
  TensorOpsDialect
  MLIRCAPIIR
  MLIRCAPIRegistration
)

# Create shared library for Rust FFI
add_library(tensor_ops_shared SHARED
  $<TARGET_OBJECTS:TensorOpsDialect>
  $<TARGET_OBJECTS:TensorOpsAPI>
)

target_link_libraries(tensor_ops_shared
  MLIRIR
  MLIRSupport
  MLIRCAPIIR
  MLIRCAPIRegistration
)

# Set output name for easier linking from Rust
set_target_properties(tensor_ops_shared PROPERTIES
  OUTPUT_NAME "tensor_ops"
  PREFIX "lib"
)

# Installation
install(TARGETS tensor_ops_shared
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

install(FILES
  ${PROJECT_BINARY_DIR}/cpp/include/TensorOps/TensorOpsDialect.h.inc
  ${PROJECT_BINARY_DIR}/cpp/include/TensorOps/TensorOpsOps.h.inc
  DESTINATION include/TensorOps
)