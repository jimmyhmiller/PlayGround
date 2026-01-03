//! Pipeline builder for constructing MLIR pass pipelines from compilation specs.
//!
//! This module handles the conversion from CompilationGpu and CompilationCpu
//! AST nodes to MLIR pipeline strings with proper nesting.
//!
//! The ordering of passes is critical. GPU module passes (gpu.module scope)
//! must appear in their correct position relative to module passes - e.g.,
//! convert-gpu-to-rocdl must run BEFORE gpu-module-to-binary.

use std::collections::HashMap;

use crate::ast::{CompilationCpu, CompilationGpu, PassScope};

/// Build an MLIR pipeline string from a GPU compilation specification.
///
/// The pipeline is constructed with proper nesting and ordering:
/// - Func-scoped passes are wrapped with `func.func(...)` individually and kept in position
/// - GPU module passes are collected into `gpu.module(...)` at their position
/// - Module passes stay at their position
///
/// This preserves the original pass ordering, which is critical for correctness.
/// For example, `scf-parallel-loop-tiling` must run AFTER `convert-linalg-to-parallel-loops`.
pub fn build_gpu_pipeline(
    spec: &CompilationGpu,
    runtime_attrs: &HashMap<String, String>,
) -> String {
    let mut before_gpu = Vec::new();
    let mut gpu_passes = Vec::new();
    let mut after_gpu = Vec::new();
    let mut seen_gpu_pass = false;

    // Always add llvm-request-c-wrappers first (wrapped in func.func)
    before_gpu.push("func.func(llvm-request-c-wrappers)".to_string());

    for pass in &spec.passes {
        let pass_str = pass.to_pipeline_string(runtime_attrs);
        match pass.scope {
            PassScope::Func => {
                // Wrap func-scoped passes individually and keep them in position
                let wrapped = format!("func.func({})", pass_str);
                if seen_gpu_pass {
                    after_gpu.push(wrapped);
                } else {
                    before_gpu.push(wrapped);
                }
            }
            PassScope::GpuModule => {
                seen_gpu_pass = true;
                gpu_passes.push(pass_str);
            }
            PassScope::Module => {
                if seen_gpu_pass {
                    after_gpu.push(pass_str);
                } else {
                    before_gpu.push(pass_str);
                }
            }
        }
    }

    build_pipeline_string_simple(before_gpu, gpu_passes, after_gpu)
}

/// Build an MLIR pipeline string from a CPU compilation specification.
///
/// The pipeline is constructed with proper nesting:
/// - `func.func(...)` for function-scoped passes
/// - Module-level passes at the top level
/// - No GPU module passes (would be an error in syntax)
pub fn build_cpu_pipeline(
    spec: &CompilationCpu,
    runtime_attrs: &HashMap<String, String>,
) -> String {
    let mut module_passes = Vec::new();
    let mut func_passes = Vec::new();

    for pass in &spec.passes {
        let pass_str = pass.to_pipeline_string(runtime_attrs);
        match pass.scope {
            PassScope::Module => module_passes.push(pass_str),
            PassScope::Func => func_passes.push(pass_str),
            PassScope::GpuModule => {
                // This shouldn't happen if parser validates correctly
                eprintln!(
                    "Warning: GPU module pass '{}' in CPU compilation spec, treating as module pass",
                    pass.name
                );
                module_passes.push(pass_str);
            }
        }
    }

    // No GPU passes for CPU compilation
    build_pipeline_string_ordered(func_passes, module_passes, Vec::new(), Vec::new())
}

/// Simple pipeline builder that takes pre-formatted pass groups.
/// Used by build_gpu_pipeline where func passes are already wrapped individually.
fn build_pipeline_string_simple(
    before_gpu: Vec<String>,
    gpu_passes: Vec<String>,
    after_gpu: Vec<String>,
) -> String {
    let mut parts = before_gpu;

    // gpu.module() scope
    if !gpu_passes.is_empty() {
        parts.push(format!("gpu.module({})", gpu_passes.join(",")));
    }

    // Module-level passes AFTER gpu.module
    parts.extend(after_gpu);

    format!("builtin.module({})", parts.join(","))
}

/// Construct the nested MLIR pipeline string with correct ordering.
///
/// Pipeline structure:
/// builtin.module(func.func(...), before_gpu..., gpu.module(...), after_gpu...)
fn build_pipeline_string_ordered(
    func_passes: Vec<String>,
    before_gpu: Vec<String>,
    gpu_passes: Vec<String>,
    after_gpu: Vec<String>,
) -> String {
    let mut parts = Vec::new();

    // func.func() scope - always include llvm-request-c-wrappers first
    if func_passes.is_empty() {
        parts.push("func.func(llvm-request-c-wrappers)".to_string());
    } else {
        let mut all_func_passes = vec!["llvm-request-c-wrappers".to_string()];
        all_func_passes.extend(func_passes);
        parts.push(format!("func.func({})", all_func_passes.join(",")));
    }

    // Module-level passes BEFORE gpu.module
    parts.extend(before_gpu);

    // gpu.module() scope - in the correct position
    if !gpu_passes.is_empty() {
        parts.push(format!("gpu.module({})", gpu_passes.join(",")));
    }

    // Module-level passes AFTER gpu.module
    parts.extend(after_gpu);

    format!("builtin.module({})", parts.join(","))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Pass;

    #[test]
    fn test_build_gpu_pipeline_basic() {
        let mut spec = CompilationGpu::new("rocm");
        spec.passes.push(Pass::with_scope(
            "convert-linalg-to-parallel-loops",
            PassScope::Module,
        ));
        spec.passes.push(Pass::with_scope(
            "gpu-map-parallel-loops",
            PassScope::Func,
        ));
        spec.passes.push(Pass::with_scope(
            "convert-gpu-to-rocdl",
            PassScope::GpuModule,
        ));

        let pipeline = build_gpu_pipeline(&spec, &HashMap::new());

        assert!(pipeline.contains("func.func("));
        assert!(pipeline.contains("gpu-map-parallel-loops"));
        assert!(pipeline.contains("gpu.module(convert-gpu-to-rocdl)"));
        assert!(pipeline.contains("convert-linalg-to-parallel-loops"));
    }

    #[test]
    fn test_build_cpu_pipeline_basic() {
        let mut spec = CompilationCpu::new();
        spec.passes.push(Pass::with_scope(
            "convert-func-to-llvm",
            PassScope::Module,
        ));
        spec.passes.push(Pass::with_scope(
            "canonicalize",
            PassScope::Func,
        ));

        let pipeline = build_cpu_pipeline(&spec, &HashMap::new());

        assert!(pipeline.contains("func.func("));
        assert!(pipeline.contains("canonicalize"));
        assert!(pipeline.contains("convert-func-to-llvm"));
        assert!(!pipeline.contains("gpu.module"));
    }
}
