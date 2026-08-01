//! Async-evaluation propagation for emitted JavaScript module graphs.

use std::collections::HashMap;

pub type DenseModuleId = usize;

pub struct AsyncDependency<'a> {
    pub specifier: &'a str,
    pub target: DenseModuleId,
    pub pruned: bool,
}

pub trait AsyncGraph {
    type Dependencies<'a>: Iterator<Item = AsyncDependency<'a>>
    where
        Self: 'a;

    fn module_count(&self) -> usize;
    fn id(&self, module: DenseModuleId) -> &str;
    fn emitted(&self, module: DenseModuleId) -> bool;
    fn uses_top_level_await(&self, module: DenseModuleId) -> bool;
    fn code(&self, module: DenseModuleId) -> &str;
    fn dependencies(&self, module: DenseModuleId) -> Self::Dependencies<'_>;
}

#[derive(Default)]
pub struct AsyncModules {
    flags: Vec<bool>,
    pub any: bool,
}

impl AsyncModules {
    pub fn is_async(&self, module: DenseModuleId) -> bool {
        self.any && self.flags.get(module).copied().unwrap_or(false)
    }
}

pub fn propagate(
    graph: &impl AsyncGraph,
    reachable: &[DenseModuleId],
) -> Result<AsyncModules, String> {
    propagate_with_policy(graph, reachable, true)
}

/// Computes the same async-evaluation closure for graph analysis performed
/// before emit. Edges that cannot legally await an async dependency are skipped
/// rather than diagnosed; the caller can use the result to rewrite those edges
/// before the strict emit-time propagation pass.
pub fn detect(graph: &impl AsyncGraph, reachable: &[DenseModuleId]) -> AsyncModules {
    propagate_with_policy(graph, reachable, false)
        .expect("detection mode never rejects a non-awaitable dependency")
}

fn propagate_with_policy(
    graph: &impl AsyncGraph,
    reachable: &[DenseModuleId],
    reject_non_awaitable: bool,
) -> Result<AsyncModules, String> {
    let mut flags = vec![false; graph.module_count()];
    let mut queue = Vec::new();
    for &module in reachable {
        if graph.emitted(module) && graph.uses_top_level_await(module) {
            flags[module] = true;
            queue.push(module);
        }
    }
    if queue.is_empty() {
        return Ok(AsyncModules { flags, any: false });
    }
    let mut importers: HashMap<DenseModuleId, Vec<(DenseModuleId, String)>> = HashMap::new();
    for &module in reachable {
        if !graph.emitted(module) {
            continue;
        }
        for dependency in graph.dependencies(module) {
            if !graph.emitted(dependency.target) || dependency.pruned {
                continue;
            }
            importers
                .entry(dependency.target)
                .or_default()
                .push((module, dependency.specifier.to_string()));
        }
    }
    while let Some(module) = queue.pop() {
        for (importer, specifier) in importers.get(&module).into_iter().flatten() {
            match AwaitableImport::classify(graph.code(*importer), specifier) {
                AwaitableImport::None => continue,
                AwaitableImport::Statement | AwaitableImport::ReExportAll => {}
                AwaitableImport::LazyNamespace => {
                    if !reject_non_awaitable {
                        continue;
                    }
                    return Err(format!(
                        "{} does `export * as ... from {:?}`, and {} uses top-level await: the namespace re-export is a lazy getter, which cannot await the module's initialisation. Import it with a normal `import * as ns from {:?}; export {{ ns }}` instead",
                        graph.id(*importer),
                        specifier,
                        graph.id(module),
                        specifier
                    ));
                }
                AwaitableImport::BareRequire => {
                    if !reject_non_awaitable {
                        continue;
                    }
                    return Err(format!(
                        "{} reaches {} through a CommonJS `require({:?})`, and that module uses top-level await: a synchronous `require` cannot wait for it (Node throws ERR_REQUIRE_ASYNC_MODULE here too). Reach it with a static `import` or a dynamic `import()` instead",
                        graph.id(*importer),
                        graph.id(module),
                        specifier
                    ));
                }
            }
            if !flags[*importer] {
                flags[*importer] = true;
                queue.push(*importer);
            }
        }
    }
    Ok(AsyncModules { flags, any: true })
}

/// Rewrites the marked static import sites in an async module so evaluation
/// waits for async dependencies. Dynamic imports and unmarked nested requires
/// are intentionally untouched.
pub fn rewrite_imports(code: &str, specifier: &str) -> String {
    let quoted = serde_json::to_string(specifier).unwrap_or_else(|_| "\"\"".to_string());
    let marker = format!("/*__diffpack_import:{quoted}__*/");
    let esm_call = format!("require.esm({quoted})");
    let plain_call = format!("require({quoted})");
    let mut out = String::with_capacity(code.len() + 32);
    let mut rest = code;
    while let Some(position) = rest.find(&marker) {
        let (head, tail) = rest.split_at(position + marker.len());
        out.push_str(head);
        let end = tail.find('\n').unwrap_or(tail.len());
        let (statement, after) = tail.split_at(end);
        if let Some(index) = statement.find(&esm_call) {
            out.push_str(&statement[..index]);
            out.push_str(&format!("await require.esmAsync({quoted})"));
            out.push_str(&statement[index + esm_call.len()..]);
        } else if let Some(index) = statement.find(&plain_call) {
            out.push_str(&statement[..index]);
            out.push_str(&format!("await require.async({quoted})"));
            out.push_str(&statement[index + plain_call.len()..]);
        } else {
            out.push_str(statement);
        }
        rest = after;
    }
    out.push_str(rest);
    out.replace(
        &format!("__reExport(exports,require.esm({quoted}));"),
        &format!("__reExport(exports,await require.esmAsync({quoted}));"),
    )
}

enum AwaitableImport {
    Statement,
    ReExportAll,
    LazyNamespace,
    BareRequire,
    None,
}

impl AwaitableImport {
    fn classify(code: &str, specifier: &str) -> Self {
        let quoted = serde_json::to_string(specifier).unwrap_or_else(|_| "\"\"".to_string());
        if code.contains(&format!("/*__diffpack_import:{quoted}__*/")) {
            Self::Statement
        } else if code.contains(&format!("__reExport(exports,require.esm({quoted}));")) {
            Self::ReExportAll
        } else if code.contains(&format!("()=>require.esm({quoted})")) {
            Self::LazyNamespace
        } else if code.contains(&format!("require({quoted})")) {
            Self::BareRequire
        } else {
            Self::None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewrite_only_awaits_marked_static_sites_and_reexports() {
        let code = concat!(
            "/*__diffpack_import:\"./x\"__*/ns=require.esm(\"./x\");\n",
            "const nested=()=>require(\"./x\");\n",
            "__reExport(exports,require.esm(\"./x\"));\n",
        );
        let rewritten = rewrite_imports(code, "./x");
        assert!(rewritten.contains("ns=await require.esmAsync(\"./x\")"));
        assert!(rewritten.contains("()=>require(\"./x\")"));
        assert!(rewritten.contains("__reExport(exports,await require.esmAsync(\"./x\"))"));
    }
}
