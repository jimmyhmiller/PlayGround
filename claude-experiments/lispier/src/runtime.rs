use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use thiserror::Error;

use crate::ast::{Compilation, Target};

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("no GPU backend detected")]
    NoBackendDetected,

    #[error("backend '{0}' not available on this system")]
    BackendNotAvailable(String),

    #[error("could not detect GPU chip")]
    ChipDetectionFailed,

    #[error("mlir-opt not found in PATH")]
    MlirOptNotFound,

    #[error("mlir-runner not found in PATH")]
    MlirRunnerNotFound,

    #[error("runtime library not found: {0}")]
    RuntimeLibNotFound(String),

    #[error("mlir-opt failed: {0}")]
    MlirOptFailed(String),

    #[error("mlir-runner failed: {0}")]
    MlirRunnerFailed(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("no matching target in compilation spec for backend '{0}'")]
    NoMatchingTarget(String),
}

/// Detected GPU backend
#[derive(Debug, Clone, PartialEq)]
pub enum Backend {
    Rocm,
    Cuda,
    Cpu,
}

impl Backend {
    pub fn name(&self) -> &str {
        match self {
            Backend::Rocm => "rocm",
            Backend::Cuda => "cuda",
            Backend::Cpu => "cpu",
        }
    }
}

/// Runtime environment information
#[derive(Debug)]
pub struct RuntimeEnv {
    pub backend: Backend,
    pub chip: Option<String>,
    pub mlir_opt: PathBuf,
    pub mlir_runner: PathBuf,
    pub runtime_libs: Vec<PathBuf>,
}

impl RuntimeEnv {
    /// Detect the runtime environment
    pub fn detect() -> Result<Self, RuntimeError> {
        // Try to detect backend
        let (backend, chip) = detect_backend_and_chip()?;

        // Find MLIR tools
        let mlir_opt = find_mlir_opt()?;
        let mlir_runner = find_mlir_runner()?;

        // Find runtime libraries
        let runtime_libs = find_runtime_libs(&backend)?;

        Ok(Self {
            backend,
            chip,
            mlir_opt,
            mlir_runner,
            runtime_libs,
        })
    }

    /// Get runtime attributes to inject into passes
    pub fn runtime_attrs(&self) -> HashMap<String, String> {
        let mut attrs = HashMap::new();
        if let Some(ref chip) = self.chip {
            attrs.insert("chip".to_string(), chip.clone());
        }
        attrs
    }

    /// Run the compilation pipeline for a given MLIR input
    pub fn run_pipeline(
        &self,
        mlir_input: &str,
        target: &Target,
    ) -> Result<String, RuntimeError> {
        let runtime_attrs = self.runtime_attrs();

        // Build pass arguments
        let pass_args: Vec<String> = target
            .passes
            .iter()
            .map(|p| p.to_pass_arg(&runtime_attrs))
            .collect();

        // Create temp file for input
        let temp_dir = std::env::temp_dir();
        let input_path = temp_dir.join("lispier_input.mlir");
        let output_path = temp_dir.join("lispier_output.mlir");

        std::fs::write(&input_path, mlir_input)?;

        // Run mlir-opt with passes
        let mut cmd = Command::new(&self.mlir_opt);
        cmd.arg(&input_path);
        cmd.arg("-o").arg(&output_path);

        for pass_arg in &pass_args {
            // Parse the pass arg - it might have quotes that need handling
            if pass_arg.contains('=') {
                // Split on first = for passes with options
                let trimmed = pass_arg.trim_start_matches("--");
                if let Some((name, opts)) = trimmed.split_once('=') {
                    let opts = opts.trim_matches('\'');
                    cmd.arg(format!("--{}={}", name, opts));
                } else {
                    cmd.arg(pass_arg);
                }
            } else {
                cmd.arg(pass_arg);
            }
        }

        let output = cmd.output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(RuntimeError::MlirOptFailed(stderr.to_string()));
        }

        std::fs::read_to_string(&output_path).map_err(|e| e.into())
    }

    /// Execute the compiled MLIR
    pub fn execute(&self, mlir_input: &str) -> Result<String, RuntimeError> {
        let temp_dir = std::env::temp_dir();
        let input_path = temp_dir.join("lispier_run.mlir");

        std::fs::write(&input_path, mlir_input)?;

        let mut cmd = Command::new(&self.mlir_runner);
        cmd.arg(&input_path);
        cmd.arg("--entry-point-result=void");

        // Add shared libs
        let libs_arg = self
            .runtime_libs
            .iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect::<Vec<_>>()
            .join(",");
        cmd.arg(format!("--shared-libs={}", libs_arg));

        let output = cmd.output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(RuntimeError::MlirRunnerFailed(stderr.to_string()));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(stdout.to_string())
    }

    /// Run the full pipeline: compile and execute
    pub fn compile_and_run(
        &self,
        mlir_input: &str,
        compilation: &Compilation,
    ) -> Result<String, RuntimeError> {
        // Find matching target
        let target = compilation
            .get_target(self.backend.name())
            .ok_or_else(|| RuntimeError::NoMatchingTarget(self.backend.name().to_string()))?;

        // Run compilation pipeline
        let compiled = self.run_pipeline(mlir_input, target)?;

        // Execute
        self.execute(&compiled)
    }
}

/// Detect the backend and GPU chip
fn detect_backend_and_chip() -> Result<(Backend, Option<String>), RuntimeError> {
    // Try ROCm first
    if let Ok(output) = Command::new("rocminfo").output() {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            // Parse chip from rocminfo output - look for "Name:" line with gfx
            for line in stdout.lines() {
                let line = line.trim();
                if line.starts_with("Name:") {
                    let name = line.trim_start_matches("Name:").trim();
                    if name.starts_with("gfx") {
                        return Ok((Backend::Rocm, Some(name.to_string())));
                    }
                }
            }
            // ROCm is available but couldn't parse chip
            return Ok((Backend::Rocm, None));
        }
    }

    // Try CUDA
    if let Ok(output) = Command::new("nvidia-smi")
        .arg("--query-gpu=compute_cap")
        .arg("--format=csv,noheader")
        .output()
    {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let compute_cap = stdout.trim();
            if !compute_cap.is_empty() {
                // Convert compute capability (e.g., "8.0") to SM version (e.g., "sm_80")
                let sm = compute_cap.replace('.', "");
                return Ok((Backend::Cuda, Some(format!("sm_{}", sm))));
            }
            return Ok((Backend::Cuda, None));
        }
    }

    // Fall back to CPU
    Ok((Backend::Cpu, None))
}

/// Find mlir-opt in PATH or common locations
fn find_mlir_opt() -> Result<PathBuf, RuntimeError> {
    // Try which command first
    if let Ok(output) = Command::new("which").arg("mlir-opt").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout);
            return Ok(PathBuf::from(path.trim()));
        }
    }

    // Try common versioned names
    for suffix in &["", "-20", "-19", "-18"] {
        let name = format!("mlir-opt{}", suffix);
        if let Ok(output) = Command::new("which").arg(&name).output() {
            if output.status.success() {
                let path = String::from_utf8_lossy(&output.stdout);
                return Ok(PathBuf::from(path.trim()));
            }
        }
    }

    // Try common locations
    let common_paths = [
        "/usr/local/bin/mlir-opt",
        "/usr/bin/mlir-opt",
        "/usr/bin/mlir-opt-20",
        "/usr/bin/mlir-opt-19",
    ];

    for path in &common_paths {
        let p = PathBuf::from(path);
        if p.exists() {
            return Ok(p);
        }
    }

    Err(RuntimeError::MlirOptNotFound)
}

/// Find mlir-runner in PATH or common locations
fn find_mlir_runner() -> Result<PathBuf, RuntimeError> {
    // Try which command first
    if let Ok(output) = Command::new("which").arg("mlir-runner").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout);
            return Ok(PathBuf::from(path.trim()));
        }
    }

    // Try common versioned names
    for suffix in &["", "-20", "-19", "-18"] {
        let name = format!("mlir-runner{}", suffix);
        if let Ok(output) = Command::new("which").arg(&name).output() {
            if output.status.success() {
                let path = String::from_utf8_lossy(&output.stdout);
                return Ok(PathBuf::from(path.trim()));
            }
        }
    }

    // Try common locations
    let common_paths = [
        "/usr/local/bin/mlir-runner",
        "/usr/bin/mlir-runner",
        "/usr/bin/mlir-runner-20",
        "/usr/bin/mlir-runner-19",
    ];

    for path in &common_paths {
        let p = PathBuf::from(path);
        if p.exists() {
            return Ok(p);
        }
    }

    Err(RuntimeError::MlirRunnerNotFound)
}

/// Find runtime libraries for the given backend
fn find_runtime_libs(backend: &Backend) -> Result<Vec<PathBuf>, RuntimeError> {
    let mut libs = Vec::new();

    // Common library search paths
    let search_paths = [
        "/usr/local/lib",
        "/usr/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/opt/rocm/lib",
    ];

    // Libraries we need
    let mut needed_libs = vec![
        "libmlir_runner_utils.so",
        "libmlir_c_runner_utils.so",
    ];

    // Add backend-specific runtime
    match backend {
        Backend::Rocm => needed_libs.push("libmlir_rocm_runtime.so"),
        Backend::Cuda => needed_libs.push("libmlir_cuda_runtime.so"),
        Backend::Cpu => {}
    }

    for lib_name in needed_libs {
        let mut found = false;
        for search_path in &search_paths {
            let path = PathBuf::from(search_path).join(lib_name);
            if path.exists() {
                libs.push(path);
                found = true;
                break;
            }
        }
        if !found {
            return Err(RuntimeError::RuntimeLibNotFound(lib_name.to_string()));
        }
    }

    Ok(libs)
}

/// Extract compilation spec from parsed nodes
pub fn extract_compilation(nodes: &[crate::ast::Node]) -> Option<Compilation> {
    for node in nodes {
        if let crate::ast::Node::Compilation(c) = node {
            return Some(c.clone());
        }
    }
    None
}

/// Extract extern declarations from parsed nodes
pub fn extract_externs(nodes: &[crate::ast::Node]) -> Vec<crate::ast::Extern> {
    let mut externs = Vec::new();
    for node in nodes {
        if let crate::ast::Node::Extern(e) = node {
            externs.push(e.clone());
        }
    }
    externs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Pass;

    #[test]
    fn test_pass_to_arg_simple() {
        let pass = Pass::new("gpu-kernel-outlining");
        let arg = pass.to_pass_arg(&HashMap::new());
        assert_eq!(arg, "--gpu-kernel-outlining");
    }

    #[test]
    fn test_pass_to_arg_with_runtime_attrs() {
        let pass = Pass::new("rocdl-attach-target");
        let mut runtime_attrs = HashMap::new();
        runtime_attrs.insert("chip".to_string(), "gfx1151".to_string());
        let arg = pass.to_pass_arg(&runtime_attrs);
        assert_eq!(arg, "--rocdl-attach-target='chip=gfx1151'");
    }

    #[test]
    fn test_pass_to_arg_with_static_attrs() {
        let mut attrs = HashMap::new();
        attrs.insert("use-bare-ptr-memref-call-conv".to_string(), "true".to_string());
        let pass = Pass::with_attributes("convert-gpu-to-rocdl", attrs);
        let arg = pass.to_pass_arg(&HashMap::new());
        assert!(arg.contains("convert-gpu-to-rocdl"));
        assert!(arg.contains("use-bare-ptr-memref-call-conv=true"));
    }
}
