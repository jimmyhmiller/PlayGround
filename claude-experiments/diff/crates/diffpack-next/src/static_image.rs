//! Next static-image import representation.

use std::path::Path;

use diffpack_core::transform::TransformResult;
use diffpack_default_loader::asset::{
    AssetEmission, asset_variant_public_name, content_hash, generate_blur_data_url,
};
use diffpack_default_loader::module::SpecialModule;

pub fn module(
    source_path: &Path,
    bytes: &[u8],
    public_name: &str,
    base: &str,
    responsive_variants: bool,
    compile: impl FnOnce(&str) -> TransformResult,
) -> Result<Option<SpecialModule>, String> {
    let extension = source_path
        .extension()
        .and_then(|value| value.to_str())
        .map(str::to_ascii_lowercase);
    if !matches!(extension.as_deref(), Some("png" | "jpg" | "jpeg")) {
        return Ok(None);
    }
    let Ok(decoded) = image::load_from_memory(bytes) else {
        return Ok(None);
    };
    let width = decoded.width();
    let height = decoded.height();
    if width == 0 || height == 0 {
        return Ok(None);
    }
    let output_extension = if extension.as_deref() == Some("jpg") {
        "jpeg"
    } else {
        extension.as_deref().unwrap_or("png")
    };
    let blur = generate_blur_data_url(&decoded, output_extension)?;
    let source_url = format!("{base}assets/{public_name}");
    let widths = responsive_variants.then(|| crate::next_adapter::variant_widths(width));
    let variants = match &widths {
        Some(widths) => {
            let entries = widths
                .iter()
                .map(|&width| {
                    let url = format!(
                        "{base}assets/{}",
                        asset_variant_public_name(public_name, width)
                    );
                    format!("{}: {}", quote(&width.to_string()), quote(&url))
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!(", variants: {{ {entries} }}")
        }
        None => String::new(),
    };
    let source = format!(
        "export default {{ src: {}, width: {width}, height: {height}, blurDataURL: {}{variants} }};\n",
        quote(&source_url),
        quote(&blur),
    );
    let transformed = compile(&source);
    Ok(Some(SpecialModule {
        hash: content_hash(transformed.code.as_bytes()),
        code: transformed.code,
        flat_module: transformed.flat_module,
        assets: vec![AssetEmission {
            source: source_path.to_path_buf(),
            public_name: public_name.to_string(),
            tailwind_source: None,
            image_variants: widths,
        }],
        css: None,
        css_source_files: Vec::new(),
        css_external_imports: Vec::new(),
        dependency_specifiers: Vec::new(),
        dependency_demands: Vec::new(),
    }))
}

fn quote(value: &str) -> String {
    serde_json::to_string(value).expect("a string always serializes")
}

#[cfg(test)]
mod tests {
    use super::*;
    use diffpack_core::transform::Target;

    #[test]
    fn creates_nexts_static_image_object_and_variant_plan() {
        let image = image::DynamicImage::new_rgb8(300, 200);
        let mut bytes = Vec::new();
        image
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Png,
            )
            .unwrap();
        let loaded = module(
            Path::new("hero.png"),
            &bytes,
            "hero.hash.png",
            "/",
            true,
            |source| {
                diffpack_core::compiler::transform_module(
                    Path::new("diffpack-image-import.js"),
                    source,
                    Target::Server,
                )
            },
        )
        .unwrap()
        .unwrap();
        assert!(loaded.code.contains("blurDataURL"));
        assert!(loaded.code.contains("width: 300"));
        assert!(
            loaded.assets[0]
                .image_variants
                .as_ref()
                .unwrap()
                .contains(&300)
        );
    }
}
