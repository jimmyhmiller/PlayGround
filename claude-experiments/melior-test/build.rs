use std::env;

fn main() {
    // DISABLED: C++ TensorOps dialect build
    // Enable this when MLIR is properly installed and configured
    
    if env::var("BUILD_TENSOR_OPS_CPP").is_ok() {
        build_cpp_dialect();
    } else {
        println!("cargo:warning=C++ TensorOps dialect build disabled. Set BUILD_TENSOR_OPS_CPP=1 to enable.");
        println!("cargo:warning=Using superficial tensor_ops operations instead of proper MLIR dialect.");
    }
}

#[allow(dead_code)]
fn build_cpp_dialect() {
    use std::path::PathBuf;
    use std::process::Command;
    
    println!("cargo:rerun-if-changed=cpp/");
    println!("cargo:rerun-if-changed=capi/");
    println!("cargo:rerun-if-changed=CMakeLists.txt");

    // Get the output directory for build artifacts
    let out_dir = env::var("OUT_DIR").unwrap();
    let cmake_build_dir = PathBuf::from(&out_dir).join("cmake_build");

    // Find MLIR installation
    let mlir_dir = find_mlir_installation();

    // Configure and build with CMake
    let mut cmake_config = Command::new("cmake");
    cmake_config
        .args(["-S", "."])
        .args(["-B", cmake_build_dir.to_str().unwrap()])
        .args(["-DCMAKE_BUILD_TYPE=Release"]);

    if let Some(mlir_path) = mlir_dir {
        cmake_config.args(["-DMLIR_DIR", &mlir_path]);
    }

    let cmake_output = cmake_config.output()
        .expect("Failed to configure CMake project");

    if !cmake_output.status.success() {
        panic!("CMake configuration failed:\n{}", 
               String::from_utf8_lossy(&cmake_output.stderr));
    }

    // Build the project
    let cmake_build = Command::new("cmake")
        .args(["--build", cmake_build_dir.to_str().unwrap()])
        .args(["--config", "Release"])
        .output()
        .expect("Failed to build CMake project");

    if !cmake_build.status.success() {
        panic!("CMake build failed:\n{}", 
               String::from_utf8_lossy(&cmake_build.stderr));
    }

    // Tell Cargo where to find the built library
    println!("cargo:rustc-link-search=native={}", cmake_build_dir.display());
    println!("cargo:rustc-link-lib=dylib=tensor_ops");

    // Link required MLIR libraries
    println!("cargo:rustc-link-lib=dylib=MLIR");
    println!("cargo:rustc-link-lib=dylib=MLIRIR");
    println!("cargo:rustc-link-lib=dylib=MLIRSupport");
    println!("cargo:rustc-link-lib=dylib=MLIRCAPIIR");

    // Link LLVM libraries if needed
    println!("cargo:rustc-link-lib=dylib=LLVMSupport");
    println!("cargo:rustc-link-lib=dylib=LLVMCore");
}

fn find_mlir_installation() -> Option<String> {
    // Try common MLIR installation paths
    let common_paths = [
        "/usr/local/lib/cmake/mlir",
        "/opt/mlir/lib/cmake/mlir",
        "/usr/lib/cmake/mlir",
    ];

    for path in &common_paths {
        if std::path::Path::new(path).exists() {
            return Some(path.to_string());
        }
    }

    // Try environment variable
    if let Ok(mlir_dir) = env::var("MLIR_DIR") {
        return Some(mlir_dir);
    }

    // Try pkg-config
    if let Ok(output) = Command::new("pkg-config")
        .args(["--variable=prefix", "mlir"])
        .output() 
    {
        if output.status.success() {
            let prefix = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let cmake_dir = format!("{}/lib/cmake/mlir", prefix);
            if std::path::Path::new(&cmake_dir).exists() {
                return Some(cmake_dir);
            }
        }
    }

    println!("cargo:warning=MLIR installation not found. Please set MLIR_DIR environment variable.");
    None
}