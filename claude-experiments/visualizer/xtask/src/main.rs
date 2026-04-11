use std::env;
use std::path::PathBuf;
use std::process::Command;

mod server;

fn project_root() -> PathBuf {
    PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .parent()
        .unwrap()
        .to_path_buf()
}

fn run(cmd: &mut Command) {
    let status = cmd.status().unwrap_or_else(|e| panic!("failed to run {cmd:?}: {e}"));
    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }
}

fn build(release: bool) {
    let root = project_root();

    let mut cmd = Command::new("cargo");
    cmd.args(["build", "--package", "visualizer", "--target", "wasm32-unknown-unknown"]);
    if release {
        cmd.arg("--release");
    }
    cmd.current_dir(&root);
    run(&mut cmd);

    let profile = if release { "release" } else { "debug" };
    let wasm = root
        .join("target/wasm32-unknown-unknown")
        .join(profile)
        .join("visualizer.wasm");
    let out_dir = root.join("web");

    let mut cmd = Command::new("wasm-bindgen");
    cmd.args(["--target", "web", "--out-dir"]);
    cmd.arg(&out_dir);
    cmd.arg(&wasm);
    run(&mut cmd);

    eprintln!("Built to {}", out_dir.display());
}

fn serve() {
    let root = project_root();
    let web_dir = root.join("web");
    let scenes_dir = root.join("scenes");
    let port: u16 = 8080;

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed to build tokio runtime");
    rt.block_on(server::run(web_dir, scenes_dir, port));
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let cmd = args.first().map(|s| s.as_str()).unwrap_or("build");

    match cmd {
        "build" => {
            let release = args.iter().any(|a| a == "--release");
            build(release);
        }
        "serve" => {
            build(false);
            serve();
        }
        _ => {
            eprintln!("Usage: cargo xtask [build [--release] | serve]");
            std::process::exit(1);
        }
    }
}
