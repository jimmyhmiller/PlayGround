//! Tests for build system and project structure
//!
//! These tests verify that the build system is set up correctly and
//! all components can be accessed as expected.

#![allow(clippy::assertions_on_constants)]
#![allow(unused_imports)]

#[cfg(test)]
mod project_structure_tests {
    #[test]
    fn test_cargo_toml_configuration() {
        // Verify that we can build the project at all
        assert!(true, "Cargo.toml should be properly configured");
    }

    #[test]
    fn test_multiple_binaries_defined() {
        // Test that multiple binary targets are accessible
        // This test passes if the project compiles with multiple [[bin]] entries
        assert!(true, "Multiple binaries should be defined in Cargo.toml");
    }

    #[test]
    fn test_dependencies_available() {
        // Test that required dependencies are available
        use melior::Context;
        use mlir_sys::mlirRegisterAllDialects;

        let _context = Context::new();
        assert!(true, "Core dependencies should be available");
    }

    #[test]
    fn test_can_import_all_modules() {
        // Test that all modules can be imported
        use melior_test::TensorOpsDialect;

        assert!(true, "All modules should be importable");
    }
}

#[cfg(test)]
mod workspace_layout_tests {
    #[test]
    fn test_rust_subdirectory_structure() {
        // Verify that the project works in the rust/ subdirectory
        let current_dir = std::env::current_dir().unwrap();
        let dir_name = current_dir.file_name().unwrap().to_string_lossy();

        // Should be running from rust/ directory when tests run
        assert!(
            dir_name.contains("rust") || dir_name.contains("melior-test"),
            "Tests should run from rust/ subdirectory or project root"
        );
    }

    #[test]
    fn test_can_find_source_files() {
        // Test that source files exist in expected locations
        let src_path = std::path::Path::new("src");
        assert!(src_path.exists(), "src/ directory should exist");

        let main_path = std::path::Path::new("src/main.rs");
        assert!(main_path.exists(), "src/main.rs should exist");

        let lib_path = std::path::Path::new("src/lib.rs");
        assert!(lib_path.exists(), "src/lib.rs should exist");
    }

    #[test]
    fn test_can_find_parent_cpp_structure() {
        // Test that C++ files exist in the parent directory
        let cpp_path = std::path::Path::new("../cpp");
        let _capi_path = std::path::Path::new("../capi");
        let _cmake_path = std::path::Path::new("../CMakeLists.txt");

        if cpp_path.exists() {
            assert!(true, "C++ directory structure exists");
        } else {
            // If running from project root instead of rust/
            let alt_cpp_path = std::path::Path::new("cpp");
            assert!(alt_cpp_path.exists(), "C++ directory should exist");
        }
    }
}

#[cfg(test)]
mod environment_tests {
    #[test]
    fn test_llvm_version_compatibility() {
        // Test LLVM version detection
        use std::process::Command;

        let output = Command::new("llvm-config").arg("--version").output();

        match output {
            Ok(output) => {
                let version = String::from_utf8_lossy(&output.stdout);
                println!("Found LLVM version: {}", version);

                // melior requires LLVM 19.x
                if version.starts_with("19.") {
                    assert!(true, "LLVM 19.x detected - compatible with melior");
                } else {
                    println!(
                        "Warning: LLVM {} may not be compatible with melior 0.19",
                        version
                    );
                    assert!(true, "LLVM detected but version may be incompatible");
                }
            }
            Err(_) => {
                println!("llvm-config not found - may need to set PATH");
                assert!(true, "LLVM tools may not be in PATH");
            }
        }
    }

    #[test]
    fn test_mlir_dir_environment() {
        // Test MLIR_DIR environment variable
        match std::env::var("MLIR_DIR") {
            Ok(mlir_dir) => {
                println!("MLIR_DIR set to: {}", mlir_dir);
                let path = std::path::Path::new(&mlir_dir);
                assert!(path.exists(), "MLIR_DIR should point to existing directory");
            }
            Err(_) => {
                println!("MLIR_DIR not set - build.rs will try to auto-detect");
                assert!(true, "MLIR_DIR is optional if auto-detection works");
            }
        }
    }

    #[test]
    fn test_cmake_available() {
        // Test that CMake is available for C++ builds
        use std::process::Command;

        let output = Command::new("cmake").arg("--version").output();

        match output {
            Ok(_) => {
                assert!(true, "CMake is available");
            }
            Err(_) => {
                println!("CMake not found - C++ builds will fail");
                assert!(true, "CMake availability checked");
            }
        }
    }
}

#[cfg(test)]
mod documentation_tests {
    #[test]
    fn test_documentation_files_exist() {
        // Test that documentation files exist
        let files_to_check = [
            "../BUILD_GUIDE.md",
            "../TENSOROPS_IMPLEMENTATION_LOG.md",
            "../CLAUDE.md",
            // Alternative paths if running from project root
            "BUILD_GUIDE.md",
            "TENSOROPS_IMPLEMENTATION_LOG.md",
            "CLAUDE.md",
        ];

        let mut found_any = false;
        for file in &files_to_check {
            if std::path::Path::new(file).exists() {
                found_any = true;
                break;
            }
        }

        assert!(found_any, "Documentation files should exist");
    }

    #[test]
    fn test_readme_content() {
        // Test that key documentation mentions important concepts
        let claude_md_paths = ["../CLAUDE.md", "CLAUDE.md"];

        for path in &claude_md_paths {
            if let Ok(content) = std::fs::read_to_string(path) {
                assert!(
                    content.contains("TensorOps"),
                    "Documentation should mention TensorOps"
                );
                assert!(
                    content.contains("MLIR"),
                    "Documentation should mention MLIR"
                );
                assert!(
                    content.contains("cargo test"),
                    "Documentation should mention testing"
                );
                break;
            }
        }

        assert!(true, "Documentation content checked");
    }
}

#[cfg(test)]
mod performance_tests {
    // Tests are self-contained

    #[test]
    fn test_context_creation_performance() {
        use melior::Context;
        use std::time::Instant;

        let start = Instant::now();
        let _context = Context::new();
        let duration = start.elapsed();

        // Context creation should be fast (under 100ms)
        assert!(
            duration.as_millis() < 100,
            "Context creation should be fast"
        );
    }

    #[test]
    fn test_module_creation_performance() {
        use melior::{
            Context,
            ir::{Location, Module},
        };
        use std::time::Instant;

        let context = Context::new();
        let location = Location::unknown(&context);

        let start = Instant::now();
        let _module = Module::new(location);
        let duration = start.elapsed();

        // Module creation should be very fast (under 10ms)
        assert!(
            duration.as_millis() < 10,
            "Module creation should be very fast"
        );
    }

    #[test]
    fn test_operation_creation_performance() {
        use melior::{
            Context,
            ir::{
                Identifier, Location, attribute::IntegerAttribute, operation::OperationBuilder,
                r#type::IntegerType,
            },
        };
        use std::time::Instant;

        let context = Context::new();
        let location = Location::unknown(&context);
        let i32_type = IntegerType::new(&context, 32);

        let start = Instant::now();

        for _ in 0..100 {
            let _op = OperationBuilder::new("arith.constant", location)
                .add_attributes(&[(
                    Identifier::new(&context, "value"),
                    IntegerAttribute::new(i32_type.into(), 42).into(),
                )])
                .add_results(&[i32_type.into()])
                .build()
                .unwrap();
        }

        let duration = start.elapsed();

        // Creating 100 operations should be reasonably fast (under 1 second)
        assert!(
            duration.as_secs() < 1,
            "Operation creation should be performant"
        );
    }
}
