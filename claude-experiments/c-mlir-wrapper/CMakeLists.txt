cmake_minimum_required(VERSION 3.20)
project(mlir-introspection-c-api VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find MLIR - try common locations if not specified
if(NOT MLIR_DIR)
    # Try to find MLIR in common locations
    set(MLIR_SEARCH_PATHS
        /usr/local/lib/cmake/mlir
        /usr/lib/cmake/mlir
        /opt/homebrew/opt/llvm/lib/cmake/mlir
        /usr/local/opt/llvm/lib/cmake/mlir
        $ENV{HOME}/llvm-project/build/lib/cmake/mlir
    )

    foreach(SEARCH_PATH ${MLIR_SEARCH_PATHS})
        if(EXISTS ${SEARCH_PATH}/MLIRConfig.cmake)
            set(MLIR_DIR ${SEARCH_PATH})
            message(STATUS "Found MLIR at: ${MLIR_DIR}")
            break()
        endif()
    endforeach()
endif()

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# Main library
add_library(mlir-introspection SHARED
    src/mlir-introspection.cpp
)

target_include_directories(mlir-introspection PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link against MLIR C API and core libraries
target_link_libraries(mlir-introspection PUBLIC
    MLIRIR
    MLIRSupport
    MLIRCAPIIR
    MLIRParser
)

# Example program
add_executable(introspection-example
    examples/introspection-example.c
)

target_link_libraries(introspection-example PRIVATE
    mlir-introspection
    MLIRCAPIIR
    MLIRCAPIFunc
    MLIRCAPIArith
)

# Installation
install(TARGETS mlir-introspection
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)
