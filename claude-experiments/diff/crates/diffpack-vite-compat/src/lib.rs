//! Opt-in Vite configuration and source-transform compatibility.

pub mod env_file;
pub mod import_meta_env;
pub mod import_meta_glob;
pub mod source_policy;
pub mod vite_config;
pub mod vite_define;
pub mod vite_manifest;

// Transitional internal aliases keep the moved implementation's paths stable;
// they point strictly downward into core, never back to the root package.
#[cfg(test)]
pub(crate) mod transform {
    pub use diffpack_core::transform::*;
}
