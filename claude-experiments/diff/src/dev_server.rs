//! CLI dispatch for integration-owned development servers.

use std::path::PathBuf;

pub use diffpack_web::dev_build::DevOptions;
pub use diffpack_web::dev_proxy;
pub use diffpack_web::preview::preview;

pub fn run(options: DevOptions) -> Result<(), String> {
    let project_root = options.project_root.canonicalize().map_err(|error| {
        format!(
            "cannot open project root {}: {error}",
            options.project_root.display()
        )
    })?;

    if diffpack_next::next_adapter::is_app_router(&project_root) {
        return diffpack_next::dev_server::next::run_next(&options, &project_root);
    }

    let has_start = project_root
        .join("node_modules/@tanstack/react-start")
        .exists();
    let index_html = project_root.join("index.html");
    if !has_start && index_html.is_file() {
        let vite = diffpack_vite_compat::vite_config::config_file(&project_root).is_some();
        let config = if vite {
            diffpack_vite_compat::web_config::derive(&project_root)?.web
        } else {
            diffpack_web::config::derive_web_config(&project_root)?
        };
        return diffpack_web::spa_dev::spa::run_spa(
            &options,
            &project_root,
            &index_html,
            config,
            if vite { " (vite mode)" } else { "" },
        );
    }

    diffpack_tanstack::dev_server::run_tanstack(DevOptions {
        project_root: PathBuf::from(project_root),
        ..options
    })
}
