//! Deterministic content identity and public naming for loaded assets.

use std::path::{Path, PathBuf};

/// A filesystem asset owed by a loaded module to the output phase.
#[derive(Debug, Clone)]
pub struct AssetEmission {
    pub source: PathBuf,
    pub public_name: String,
    /// Raw Tailwind entry source requiring candidate-aware compilation at emit.
    pub tailwind_source: Option<String>,
    /// Responsive raster widths emitted beside the original asset.
    pub image_variants: Option<Vec<u32>>,
}

/// JavaScript source synthesized by an asset/query loader before core compilation.
pub struct SyntheticModule {
    pub source: String,
    pub identity: u64,
    pub assets: Vec<AssetEmission>,
}

pub fn raw_module(path: &Path) -> Result<SyntheticModule, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    Ok(SyntheticModule {
        source: format!("export default {};\n", quote(&text)),
        identity: content_hash(text.as_bytes()),
        assets: Vec::new(),
    })
}

pub fn inline_module(path: &Path) -> Result<SyntheticModule, String> {
    let bytes = std::fs::read(path)
        .map_err(|error| format!("cannot read asset {}: {error}", path.display()))?;
    let data_uri = svg_data_url(path, &bytes).unwrap_or_else(|| {
        format!(
            "data:{};base64,{}",
            asset_mime_type(path),
            base64_encode(&bytes)
        )
    });
    Ok(SyntheticModule {
        source: format!("export default {};\n", quote(&data_uri)),
        identity: content_hash(data_uri.as_bytes()),
        assets: Vec::new(),
    })
}

pub fn public_url_module(
    path: &Path,
    base: &str,
    root: Option<&Path>,
) -> Result<SyntheticModule, String> {
    let root = root.ok_or_else(|| {
        format!(
            "{}: a `?public-url` module needs the project root to derive its URL, and this build has none",
            path.display()
        )
    })?;
    let public_dir = root.join("public");
    let relative = path.strip_prefix(&public_dir).map_err(|_| {
        format!(
            "{}: a `?public-url` module must live under {}",
            path.display(),
            public_dir.display()
        )
    })?;
    let url_path = relative
        .components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/");
    let url = format!("{base}{url_path}");
    Ok(SyntheticModule {
        source: format!("export default {};\n", quote(&url)),
        identity: content_hash(url.as_bytes()),
        assets: Vec::new(),
    })
}

pub fn worker_module(path: &Path, inline: bool) -> Result<SyntheticModule, String> {
    if inline {
        return Err(format!(
            "loader `?worker&inline` (blob-inlined workers) is not yet implemented (requested for {}); use `?worker` for a separately emitted worker chunk",
            path.display()
        ));
    }
    let key = worker_key(path);
    let placeholder = format!("__diffpack_worker__{key}__");
    let source = format!(
        "export default function WorkerWrapper(options) {{\n  return new Worker({}, {{ type: \"module\", ...options }});\n}}\n",
        quote(&placeholder),
    );
    Ok(SyntheticModule {
        identity: content_hash(source.as_bytes()),
        source,
        assets: Vec::new(),
    })
}

/// Stable key shared by worker module synthesis and graph registration.
pub fn worker_key(path: &Path) -> String {
    diffpack_core::compiler::worker_key(path, "?worker")
}

pub fn wasm_init_module(
    path: &Path,
    base: &str,
    inline_limit: usize,
) -> Result<SyntheticModule, String> {
    if path.extension().and_then(|value| value.to_str()) != Some("wasm") {
        return Err(format!(
            "loader `?init` applies only to `.wasm` files (requested for {})",
            path.display()
        ));
    }
    let bytes =
        std::fs::read(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let (url, assets) = if inline_limit > 0 && bytes.len() <= inline_limit {
        (
            format!("data:application/wasm;base64,{}", base64_encode(&bytes)),
            Vec::new(),
        )
    } else {
        let public_name = asset_public_name(path, content_hash(&bytes));
        (
            format!("{base}assets/{public_name}"),
            vec![AssetEmission {
                source: path.to_path_buf(),
                public_name,
                tailwind_source: None,
                image_variants: None,
            }],
        )
    };
    let helper = include_str!("wasm_helper.js");
    let source = format!(
        "{helper}\nconst __diffpackWasmUrl = {};\nexport default (imports = {{}}) => __diffpackWasmInit(imports, __diffpackWasmUrl);\n",
        quote(&url),
    );
    Ok(SyntheticModule {
        identity: content_hash(source.as_bytes()),
        source,
        assets,
    })
}

fn quote(value: &str) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "\"\"".to_string())
}

/// How a default raster-image import materializes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageImportShape {
    #[default]
    Url,
    NextObject {
        responsive_variants: bool,
    },
}

pub fn content_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

pub fn asset_public_name(path: &Path, hash: u64) -> String {
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("asset");
    match path.extension().and_then(|value| value.to_str()) {
        Some(extension) => format!("{stem}-{hash:016x}.{extension}"),
        None => format!("{stem}-{hash:016x}"),
    }
}

pub fn asset_variant_public_name(public_name: &str, width: u32) -> String {
    match public_name.rsplit_once('.') {
        Some((stem, extension)) => format!("{stem}-{width}.{extension}"),
        None => format!("{public_name}-{width}"),
    }
}

pub fn svg_data_url(path: &Path, bytes: &[u8]) -> Option<String> {
    if path.extension().and_then(|value| value.to_str()) != Some("svg") {
        return None;
    }
    let text = std::str::from_utf8(bytes).ok()?;
    if text.contains('\'') && text.contains('"') {
        return None;
    }
    let collapsed = text
        .replace('"', "'")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .replace("> <", "><");
    let mut output = String::from("data:image/svg+xml,");
    for character in collapsed.chars() {
        match character {
            '%' => output.push_str("%25"),
            '#' => output.push_str("%23"),
            '<' => output.push_str("%3c"),
            '>' => output.push_str("%3e"),
            ' ' => output.push_str("%20"),
            '{' => output.push_str("%7b"),
            '}' => output.push_str("%7d"),
            '|' => output.push_str("%7c"),
            '^' => output.push_str("%5e"),
            '`' => output.push_str("%60"),
            '"' => output.push_str("%22"),
            '[' => output.push_str("%5b"),
            ']' => output.push_str("%5d"),
            '\\' => output.push_str("%5c"),
            '?' => output.push_str("%3f"),
            other => output.push(other),
        }
    }
    Some(output)
}

pub fn asset_mime_type(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|value| value.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        Some("avif") => "image/avif",
        Some("bmp") => "image/bmp",
        Some("ico") => "image/x-icon",
        Some("ttf") => "font/ttf",
        Some("otf") => "font/otf",
        Some("woff") => "font/woff",
        Some("woff2") => "font/woff2",
        Some("wasm") => "application/wasm",
        _ => "application/octet-stream",
    }
}

pub fn base64_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut output = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let bits = ((chunk[0] as u32) << 16)
            | ((chunk.get(1).copied().unwrap_or(0) as u32) << 8)
            | chunk.get(2).copied().unwrap_or(0) as u32;
        output.push(TABLE[((bits >> 18) & 63) as usize] as char);
        output.push(TABLE[((bits >> 12) & 63) as usize] as char);
        output.push(if chunk.len() > 1 {
            TABLE[((bits >> 6) & 63) as usize] as char
        } else {
            '='
        });
        output.push(if chunk.len() > 2 {
            TABLE[(bits & 63) as usize] as char
        } else {
            '='
        });
    }
    output
}

pub fn generate_blur_data_url(
    image: &image::DynamicImage,
    extension: &str,
) -> Result<String, String> {
    const WIDTH: u32 = 8;
    let (source_width, source_height) = (image.width().max(1), image.height().max(1));
    let height = ((source_height as u64 * WIDTH as u64) / source_width as u64).max(1) as u32;
    let small = image.resize_exact(WIDTH, height, image::imageops::FilterType::Triangle);
    let (format, mime) = if matches!(extension, "jpeg" | "jpg") {
        (image::ImageFormat::Jpeg, "image/jpeg")
    } else {
        (image::ImageFormat::Png, "image/png")
    };
    let mut buffer = std::io::Cursor::new(Vec::new());
    small
        .write_to(&mut buffer, format)
        .map_err(|error| format!("cannot encode blur placeholder: {error}"))?;
    Ok(format!(
        "data:{mime};base64,{}",
        base64_encode(&buffer.into_inner())
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn naming_preserves_the_extension_and_is_content_addressed() {
        let name = asset_public_name(Path::new("images/logo.svg"), content_hash(b"logo"));
        assert!(name.starts_with("logo-"), "{name}");
        assert!(name.ends_with(".svg"), "{name}");
    }

    #[test]
    fn variant_name_keeps_the_original_extension() {
        assert_eq!(
            asset_variant_public_name("photo-abcd.png", 640),
            "photo-abcd-640.png"
        );
    }

    #[test]
    fn svg_inlining_matches_the_compact_utf8_shape() {
        assert_eq!(
            svg_data_url(
                Path::new("icon.svg"),
                br#"<svg viewBox="0 0 1 1"> <path/></svg>"#
            ),
            Some(
                "data:image/svg+xml,%3csvg%20viewBox='0%200%201%201'%3e%3cpath/%3e%3c/svg%3e"
                    .into()
            )
        );
    }

    #[test]
    fn query_modules_are_synthesized_without_compiler_policy() {
        let dir = tempfile::tempdir().unwrap();
        let raw = dir.path().join("note.txt");
        std::fs::write(&raw, "hello\n").unwrap();
        assert!(raw_module(&raw).unwrap().source.contains("hello\\n"));

        let public = dir.path().join("public").join("icons").join("logo.svg");
        std::fs::create_dir_all(public.parent().unwrap()).unwrap();
        std::fs::write(&public, "<svg/>").unwrap();
        let module = public_url_module(&public, "/base/", Some(dir.path())).unwrap();
        assert!(module.source.contains("/base/icons/logo.svg"));

        let inline = inline_module(&public).unwrap();
        assert!(inline.source.contains("data:image/svg+xml"));
        assert!(inline.assets.is_empty());

        let worker = worker_module(Path::new("/app/worker.js"), false).unwrap();
        assert!(worker.source.contains("new Worker"));
        assert!(worker.source.contains("__diffpack_worker__"));

        let wasm = dir.path().join("small.wasm");
        std::fs::write(&wasm, b"\0asm").unwrap();
        let wasm = wasm_init_module(&wasm, "/", 100).unwrap();
        assert!(wasm.source.contains("data:application/wasm;base64"));
        assert!(wasm.assets.is_empty());
    }
}
