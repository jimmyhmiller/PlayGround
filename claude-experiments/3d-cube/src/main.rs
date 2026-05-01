mod source;
mod viewer;

use source::{PointCloudSource, TrigramSource};
use std::path::PathBuf;
use std::sync::Arc;
use viewer::App;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ViewMode {
    Binary,
}

impl ViewMode {
    pub(crate) fn label(self) -> &'static str {
        match self {
            ViewMode::Binary => "binary trigram",
        }
    }
}

pub(crate) fn source_for_view(
    _path: &std::path::Path,
    _view_mode: ViewMode,
) -> Arc<dyn PointCloudSource> {
    Arc::new(TrigramSource::default())
}

fn main() {
    let file_path = std::env::args().nth(1).map(PathBuf::from);
    let initial_view = ViewMode::Binary;
    let source: Arc<dyn PointCloudSource> = Arc::new(TrigramSource::default());

    let factory: Arc<
        dyn Fn(&std::path::Path, ViewMode) -> Arc<dyn PointCloudSource> + Send + Sync,
    > = Arc::new(source_for_view);

    let app = App::new(source, factory, initial_view, file_path);
    app.run();
}
