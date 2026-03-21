mod heap;
mod heap_cube;
mod heap_landscape;
mod heap_treemap;
mod source;
mod viewer;

use heap::HeapDumpSource;
use heap_cube::HeapCubeSource;
use heap_landscape::HeapLandscapeSource;
use heap_treemap::HeapTreemapSource;
use source::{PointCloudSource, TrigramSource};
use std::path::PathBuf;
use std::sync::Arc;
use viewer::App;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ViewMode {
    Binary,
    HeapLandscape,
    HeapGraph,
    HeapCube,
    HeapTreemap,
}

impl ViewMode {
    pub(crate) fn label(self) -> &'static str {
        match self {
            ViewMode::Binary => "binary trigram",
            ViewMode::HeapLandscape => "ownership landscape",
            ViewMode::HeapGraph => "heap graph",
            ViewMode::HeapCube => "heap cube",
            ViewMode::HeapTreemap => "heap treemap",
        }
    }
}

fn is_hprof(path: &std::path::Path) -> bool {
    path.extension().is_some_and(|ext| ext == "hprof")
}

pub(crate) fn next_view_mode(path: &std::path::Path, current: ViewMode) -> ViewMode {
    if !is_hprof(path) {
        return ViewMode::Binary;
    }

    match current {
        ViewMode::Binary => ViewMode::HeapLandscape,
        ViewMode::HeapLandscape => ViewMode::HeapGraph,
        ViewMode::HeapGraph => ViewMode::HeapCube,
        ViewMode::HeapCube => ViewMode::HeapTreemap,
        ViewMode::HeapTreemap => ViewMode::Binary,
    }
}

pub(crate) fn source_for_view(
    path: &std::path::Path,
    view_mode: ViewMode,
) -> Arc<dyn PointCloudSource> {
    match view_mode {
        ViewMode::Binary => Arc::new(TrigramSource::default()),
        ViewMode::HeapLandscape if is_hprof(path) => Arc::new(HeapLandscapeSource::default()),
        ViewMode::HeapGraph if is_hprof(path) => Arc::new(HeapDumpSource::default()),
        ViewMode::HeapCube if is_hprof(path) => Arc::new(HeapCubeSource::default()),
        ViewMode::HeapTreemap if is_hprof(path) => Arc::new(HeapTreemapSource::default()),
        _ => Arc::new(TrigramSource::default()),
    }
}

fn main() {
    let file_path = std::env::args().nth(1).map(PathBuf::from);
    let initial_view = file_path
        .as_deref()
        .filter(|path| is_hprof(path))
        .map(|_| ViewMode::HeapLandscape)
        .unwrap_or(ViewMode::Binary);
    let source = file_path
        .as_deref()
        .map(|path| source_for_view(path, initial_view))
        .unwrap_or_else(|| Arc::new(TrigramSource::default()));

    let factory: Arc<
        dyn Fn(&std::path::Path, ViewMode) -> Arc<dyn PointCloudSource> + Send + Sync,
    > = Arc::new(source_for_view);

    let app = App::new(source, factory, initial_view, file_path);
    app.run();
}
