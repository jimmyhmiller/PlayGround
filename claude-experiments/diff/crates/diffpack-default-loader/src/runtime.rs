//! Default Node host bindings for emitted ESM chunks.

/// Bindings required by bundled CommonJS modules inside a Node ESM chunk.
pub fn node_esm_chunk_prelude(is_main: bool) -> String {
    let mut prelude = "import { fileURLToPath as __diffpackFileURLToPath } from \"node:url\";\nimport { dirname as __diffpackDirname } from \"node:path\";\nconst __filename = __diffpackFileURLToPath(import.meta.url);\nconst __dirname = __diffpackDirname(__filename);\n".to_string();
    if is_main {
        prelude.insert_str(
            0,
            "import { createRequire as __diffpackCreateRequire } from \"node:module\";\n",
        );
    }
    prelude
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_entry_installs_create_require() {
        assert!(node_esm_chunk_prelude(true).contains("createRequire"));
        assert!(!node_esm_chunk_prelude(false).contains("createRequire"));
        assert!(node_esm_chunk_prelude(false).contains("fileURLToPath"));
    }
}
