mod translate;

pub use translate::{translate_wasm, translate_wasm_module};

#[cfg(test)]
mod as_tests;
#[cfg(test)]
mod tests;
