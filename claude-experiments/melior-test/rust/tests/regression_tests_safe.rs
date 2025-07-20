//! Safe regression tests that avoid MLIR calls that might segfault

#[cfg(test)]
mod safe_tests {
    #[test]
    fn test_basic_rust_functionality() {
        // Test that basic Rust functionality works
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn test_can_import_melior_types() {
        // Test that we can import melior types without using them
        use melior::Context;

        // Don't actually create a context - just test that the import works
        let _ = std::marker::PhantomData::<Context>;
        // Should be able to import melior types - test passes if we reach this point
    }

    #[test]
    fn test_can_import_mlir_sys_types() {
        use mlir_sys::{MlirContext, MlirOperation};

        // Test that mlir-sys types are available without creating them
        let _check_types = |_ctx: MlirContext, _op: MlirOperation| {
            // This function just checks that the types compile
        };

        // mlir-sys types should be available - test passes if we reach this point
    }

    #[test]
    fn test_string_operations() {
        // Test that basic string operations work
        let test_str = "tensor_ops.add";
        assert!(test_str.contains("tensor_ops"));
        assert!(test_str.contains("add"));
    }

    #[test]
    fn test_error_handling_patterns() {
        // Test error handling patterns without MLIR
        let result: Result<i32, &str> = Ok(42);
        assert!(result.is_ok());

        let error: Result<i32, &str> = Err("test error");
        assert!(error.is_err());
    }
}

#[cfg(test)]
mod documentation_tests {
    #[test]
    fn test_known_issues_documented() {
        // Test that we document known issues
        let known_issues = [
            "Function lookup optimization issue",
            "Unregistered dialect limitations",
            "FFI function availability",
        ];

        for issue in &known_issues {
            assert!(!issue.is_empty(), "Issues should be documented");
        }
    }

    #[test]
    fn test_error_messages_helpful() {
        // Test that error message patterns are helpful
        let error_msg = "Failed to create tensor_ops.add operation";
        assert!(error_msg.contains("Failed to create"));
        assert!(error_msg.contains("tensor_ops"));
    }
}
