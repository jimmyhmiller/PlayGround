use anyhow::Result;
use crate::parsers::{LanguageInstaller, LanguageRegistry, GrammarSource};

pub fn cmd_lang_install(language: String) -> Result<()> {
    let mut installer = LanguageInstaller::new()?;
    installer.install(&language)?;
    Ok(())
}

pub fn cmd_lang_uninstall(language: String) -> Result<()> {
    let installer = LanguageInstaller::new()?;
    installer.uninstall(&language)?;
    Ok(())
}

pub fn cmd_lang_list_installed() -> Result<()> {
    let installer = LanguageInstaller::new()?;
    let installed = installer.list_installed()?;

    if installed.is_empty() {
        println!("No language grammars installed");
        println!("\nInstall a language with: code-notes lang install <language>");
        println!("See available languages with: code-notes lang list-available");
        return Ok(());
    }

    println!("Installed language grammars:\n");
    for metadata in installed {
        println!("  {} (v{})", metadata.name, metadata.version);
        println!("    Extensions: {}", metadata.extensions.join(", "));
        println!("    Source: {}", metadata.source);
        println!("    Installed: {}", metadata.installed_at);
        println!();
    }

    Ok(())
}

pub fn cmd_lang_list_available() -> Result<()> {
    let installer = LanguageInstaller::new()?;
    let available = installer.list_available();
    let installed = installer.list_installed()?;

    // Create a set of installed language names for quick lookup
    let installed_names: std::collections::HashSet<String> =
        installed.iter().map(|m| m.name.clone()).collect();

    println!("Available language grammars:\n");

    // Group by category for better display
    let systems_langs = vec!["rust", "c", "cpp", "go", "zig"];
    let scripting_langs = vec!["python", "ruby", "lua", "bash"];
    let web_langs = vec!["javascript", "typescript", "html", "css"];
    let jvm_langs = vec!["java", "kotlin", "scala", "clojure"];
    let functional_langs = vec!["haskell", "ocaml", "elixir", "erlang", "racket"];
    let other_langs = vec!["php", "swift", "json", "yaml", "toml", "markdown"];

    let print_category = |title: &str, langs: &[&str]| {
        println!("{}:", title);
        for grammar in &available {
            if langs.contains(&grammar.name.as_str()) {
                let status = if installed_names.contains(&grammar.name) {
                    "✓"
                } else {
                    " "
                };
                println!("  {} {:<15} - {}", status, grammar.name, grammar.extensions.join(", "));
            }
        }
        println!();
    };

    print_category("Systems Languages", &systems_langs);
    print_category("Scripting Languages", &scripting_langs);
    print_category("Web Languages", &web_langs);
    print_category("JVM Languages", &jvm_langs);
    print_category("Functional Languages", &functional_langs);
    print_category("Other Languages", &other_langs);

    println!("Legend: ✓ = installed");
    println!("\nInstall a language with: code-notes lang install <language>");

    Ok(())
}

pub fn cmd_lang_info(language: String) -> Result<()> {
    let registry = LanguageRegistry::new()?;

    // Check if it's installed
    if let Ok(metadata) = registry.get_metadata(&language) {
        println!("Language: {}", metadata.name);
        println!("Version: {}", metadata.version);
        println!("Extensions: {}", metadata.extensions.join(", "));
        println!("Source: {}", metadata.source);
        println!("Installed at: {}", metadata.installed_at);
        println!("\nStatus: ✓ Installed");
        return Ok(());
    }

    // Not installed, check if it's available
    if let Some(source) = GrammarSource::find_by_name(&language) {
        println!("Language: {}", source.name);
        println!("Repository: https://github.com/{}", source.repo);
        println!("Extensions: {}", source.extensions.join(", "));
        println!("\nStatus: Available (not installed)");
        println!("\nInstall with: code-notes lang install {}", language);
        return Ok(());
    }

    println!("Language '{}' not found", language);
    println!("\nSee available languages with: code-notes lang list-available");

    Ok(())
}
