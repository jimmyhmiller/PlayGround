//! Brendan Gregg's folded-stacks text format. One line per unique stack, where the
//! frames are joined with `;` and a trailing integer weight is space-separated:
//!
//! ```text
//! main;a;b 50
//! main;a;c 30
//! main;d 10
//! ```
//!
//! Sample-only (no timestamps, no threads). The builder synthesizes a flame graph
//! laid out left-to-right by descending weight.

use flame_core::{
    LoadError, LoadResult, ProfileBuilder, StringId, TraceSource, TrackKind,
};

pub struct FoldedSource;

pub const SOURCE: &dyn TraceSource = &FoldedSource;

impl TraceSource for FoldedSource {
    fn name(&self) -> &'static str {
        "Folded stacks"
    }

    fn detect(&self, input: &[u8], filename: Option<&str>) -> bool {
        if let Some(fname) = filename {
            let lower = fname.to_ascii_lowercase();
            if lower.ends_with(".folded") || lower.ends_with(".collapsed") {
                return true;
            }
        }
        // Content sniff: at least one line of `frames_with_semicolons SPACE integer`.
        // Cheap and conservative — only confirms format, doesn't reject ambiguous text.
        let prefix = std::str::from_utf8(&input[..input.len().min(4096)]).ok();
        if let Some(text) = prefix {
            for line in text.lines().take(8) {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                if let Some((stack, weight)) = line.rsplit_once(' ') {
                    if !stack.is_empty()
                        && stack.contains(';')
                        && weight.bytes().all(|b| b.is_ascii_digit())
                    {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn load(&self, input: &[u8], builder: &mut ProfileBuilder) -> LoadResult<()> {
        let text = std::str::from_utf8(input)
            .map_err(|e| LoadError::Parse(format!("not utf-8: {e}")))?;

        let process = builder.add_process(0, "samples");
        let thread = builder.add_thread(Some(process), 0, "main");
        let track_name = builder.intern_string("Folded samples");
        let track = builder.add_track(TrackKind::Thread(thread), "Folded samples", None);
        let _ = track_name; // (interned for future use; track itself stores name)
        let category = builder.intern_category("samples");

        let empty_file = StringId::EMPTY;
        let mut had_any = false;

        for (line_no, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((stack_str, weight_str)) = line.rsplit_once(' ') else {
                return Err(LoadError::Parse(format!(
                    "line {}: missing weight: {raw:?}",
                    line_no + 1
                )));
            };
            let weight: u32 = weight_str.trim().parse().map_err(|_| {
                LoadError::Parse(format!(
                    "line {}: weight not an integer: {weight_str:?}",
                    line_no + 1
                ))
            })?;
            if weight == 0 {
                continue;
            }
            let mut parent = None;
            for frame_name in stack_str.split(';') {
                let fid = builder.intern_frame(frame_name, "", 0, 0);
                let _ = empty_file;
                let stack_id = builder.intern_stack(fid, parent);
                parent = Some(stack_id);
            }
            if let Some(stack) = parent {
                builder.add_sample(thread, 0, stack, weight);
                had_any = true;
            }
        }

        if !had_any {
            return Err(LoadError::Parse("no samples parsed".into()));
        }

        builder.synthesize_slices_from_samples(track, category);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_basic() {
        let input = b"main;a;b 50\nmain;a;c 30\nmain;d 10\n";
        assert!(FoldedSource.detect(input, Some("test.folded")));
        let mut b = ProfileBuilder::new();
        FoldedSource.load(input, &mut b).unwrap();
        let p = b.finish();
        // Expected slices: main(90), a(80), b(50), c(30), d(10) = 5 total.
        assert_eq!(p.slices.len(), 5);
        assert_eq!(p.duration_ns(), 90); // sum of all weights
    }

    #[test]
    fn detect_rejects_obvious_non_folded() {
        let json = br#"{"traceEvents":[]}"#;
        assert!(!FoldedSource.detect(json, Some("foo.json")));
    }
}
