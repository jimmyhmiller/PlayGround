use std::{
    env, fs,
    path::{Path, PathBuf},
};

fn get_output_path() -> PathBuf {
    let manifest_dir_string = env::var("CARGO_MANIFEST_DIR").unwrap();
    let build_type = env::var("PROFILE").unwrap();
    let path = Path::new(&manifest_dir_string)
        .parent()
        .unwrap()
        .join("target")
        .join("wasm32-wasi")
        .join(build_type);
    path
}

fn main() {
    let target_dir = get_output_path();
    let src = Path::join(&env::current_dir().unwrap(), "resources/onebigfile.xml");
    let dest = Path::join(Path::new(&target_dir), Path::new("onebigfile.xml"));
    fs::copy(src, dest).unwrap();
}
