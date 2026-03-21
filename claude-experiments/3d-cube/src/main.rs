mod heap;
mod source;
mod viewer;

use source::TrigramSource;
use heap::HeapDumpSource;
use std::path::PathBuf;
use std::sync::Arc;
use viewer::App;

fn main() {
    let file_path = std::env::args().nth(1).map(PathBuf::from);

    // Auto-detect source type from file extension
    let source: Arc<dyn source::PointCloudSource> = if file_path
        .as_ref()
        .and_then(|p| p.extension())
        .is_some_and(|ext| ext == "hprof")
    {
        Arc::new(HeapDumpSource::default())
    } else {
        Arc::new(TrigramSource::default())
    };

    let app = App::new(source, file_path);
    app.run();
}
