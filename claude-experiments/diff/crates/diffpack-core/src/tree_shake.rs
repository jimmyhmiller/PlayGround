//! Statement-level liveness for lowered module code.

use std::collections::{HashMap, HashSet};

use crate::source_map::{ColumnEdit, LineOrigin, LineTrack, utf16_len};

/// Export demand supplied by the graph/link stage without exposing graph records.
pub trait ExportDemand {
    fn includes(&self, name: &str) -> bool;
}

#[derive(Clone, Default)]
pub struct Demand {
    pub all: bool,
    pub names: HashSet<String>,
}

impl Demand {
    pub fn merge(&mut self, other: Self) {
        self.all |= other.all;
        self.names.extend(other.names);
    }
}

impl ExportDemand for Demand {
    fn includes(&self, name: &str) -> bool {
        self.all || self.names.contains(name)
    }
}

/// Shakes marker-delimited declarations and exports from one lowered module.
/// When requested, the returned line track maps surviving lines to the lowered
/// input and accounts for marker prefixes removed from kept statements.
pub fn shake(
    code: &str,
    demand: &impl ExportDemand,
    pruned_imports: &HashSet<String>,
    track_lines: bool,
) -> (String, Option<LineTrack>) {
    enum Segment<'a> {
        Keep(&'a str),
        Import {
            line: &'a str,
        },
        Export {
            statement: &'a str,
        },
        Declaration {
            names: Vec<&'a str>,
            lines: Vec<&'a str>,
        },
    }

    let mut segments = Vec::new();
    let mut open_declaration: Option<(Vec<&str>, Vec<&str>)> = None;
    let mut origins: Vec<Vec<(usize, u32)>> = Vec::new();
    let mut open_origins = Vec::new();
    for (line_index, line) in code.lines().enumerate() {
        if let Some((_, lines)) = open_declaration.as_mut() {
            if line == "/*__diffpack_decl_end__*/" {
                let (names, lines) = open_declaration.take().expect("declaration is open");
                segments.push(Segment::Declaration { names, lines });
                origins.push(std::mem::take(&mut open_origins));
            } else {
                lines.push(line);
                open_origins.push((line_index, 0));
            }
            continue;
        }
        if let Some(names) = line
            .strip_prefix("/*__diffpack_decl:")
            .and_then(|line| line.strip_suffix("__*/"))
        {
            open_declaration = Some((names.split(',').collect(), Vec::new()));
            continue;
        }
        if let Some(marked) = line.strip_prefix("/*__diffpack_import:")
            && let Some((specifier, import_code)) = marked.split_once("__*/")
            && let Ok(specifier) = serde_json::from_str::<String>(specifier)
        {
            if !pruned_imports.contains(&specifier) {
                let stripped = utf16_len(&line[..line.len() - import_code.len()]);
                segments.push(Segment::Import { line: import_code });
                origins.push(vec![(line_index, stripped)]);
            }
            continue;
        }
        if let Some(marked) = line.strip_prefix("/*__diffpack_export:")
            && let Some((name, statement)) = marked.split_once("__*/")
        {
            if demand.includes(name) {
                let stripped = utf16_len(&line[..line.len() - statement.len()]);
                segments.push(Segment::Export { statement });
                origins.push(vec![(line_index, stripped)]);
            }
            continue;
        }
        segments.push(Segment::Keep(line));
        origins.push(vec![(line_index, 0)]);
    }
    if let Some((_, lines)) = open_declaration.take() {
        for (line, origin) in lines.into_iter().zip(open_origins) {
            segments.push(Segment::Keep(line));
            origins.push(vec![origin]);
        }
    }

    let mut owner_of = HashMap::new();
    for (index, segment) in segments.iter().enumerate() {
        if let Segment::Declaration { names, .. } = segment {
            for name in names {
                owner_of.insert(*name, index);
            }
        }
    }

    let mut live = vec![false; segments.len()];
    let mut queue = Vec::new();
    let mark = |index: usize, live: &mut Vec<bool>, queue: &mut Vec<usize>| {
        if !live[index] {
            live[index] = true;
            queue.push(index);
        }
    };
    for (index, segment) in segments.iter().enumerate() {
        match segment {
            Segment::Keep(_) | Segment::Import { .. } | Segment::Export { .. } => {
                mark(index, &mut live, &mut queue);
            }
            Segment::Declaration { names, .. }
                if names.iter().any(|name| demand.includes(name)) =>
            {
                mark(index, &mut live, &mut queue);
            }
            Segment::Declaration { .. } => {}
        }
    }
    while let Some(index) = queue.pop() {
        let scan = |text: &str, live: &mut Vec<bool>, queue: &mut Vec<usize>| {
            for word in identifier_runs(text) {
                if let Some(&owner) = owner_of.get(word)
                    && !live[owner]
                {
                    live[owner] = true;
                    queue.push(owner);
                }
            }
        };
        match &segments[index] {
            Segment::Keep(line) | Segment::Import { line } => scan(line, &mut live, &mut queue),
            Segment::Export { statement } => scan(statement, &mut live, &mut queue),
            Segment::Declaration { lines, .. } => {
                for line in lines {
                    scan(line, &mut live, &mut queue);
                }
            }
        }
    }

    let mut output = String::with_capacity(code.len());
    let mut track = track_lines.then(LineTrack::default);
    for (index, segment) in segments.iter().enumerate() {
        if !live[index] {
            continue;
        }
        if let Some(track) = track.as_mut() {
            for &(line, stripped) in &origins[index] {
                let mut origin = LineOrigin {
                    source_line: Some(line as u32),
                    edits: Vec::new(),
                };
                if stripped > 0 {
                    origin.edits.push(ColumnEdit {
                        column: 0,
                        removed: stripped,
                        inserted: 0,
                    });
                }
                track.push(origin);
            }
        }
        match segment {
            Segment::Keep(line) | Segment::Import { line } => {
                output.push_str(line);
                output.push('\n');
            }
            Segment::Export { statement } => {
                output.push_str(statement);
                output.push('\n');
            }
            Segment::Declaration { lines, .. } => {
                for line in lines {
                    output.push_str(line);
                    output.push('\n');
                }
            }
        }
    }
    (output, track)
}

fn identifier_runs(text: &str) -> impl Iterator<Item = &str> {
    text.split(|character: char| {
        !(character.is_ascii_alphanumeric() || character == '_' || character == '$')
    })
    .filter(|run| !run.is_empty() && !run.starts_with(|character: char| character.is_ascii_digit()))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Names<'a>(&'a [&'a str]);

    impl ExportDemand for Names<'_> {
        fn includes(&self, name: &str) -> bool {
            self.0.contains(&name)
        }
    }

    #[test]
    fn demanded_declarations_keep_their_transitive_helpers() {
        let code = concat!(
            "/*__diffpack_decl:helper__*/\n",
            "const helper=()=>1;\n",
            "/*__diffpack_decl_end__*/\n",
            "/*__diffpack_decl:value__*/\n",
            "const value=helper();\n",
            "/*__diffpack_decl_end__*/\n",
            "/*__diffpack_decl:dead__*/\n",
            "const dead=2;\n",
            "/*__diffpack_decl_end__*/\n",
            "/*__diffpack_export:value__*/__export(exports,\"value\",()=>value);\n",
        );
        let (shaken, _) = shake(code, &Names(&["value"]), &HashSet::new(), false);
        assert!(shaken.contains("helper"));
        assert!(shaken.contains("value"));
        assert!(!shaken.contains("dead"));
    }

    #[test]
    fn pruned_imports_are_removed_and_kept_markers_are_stripped() {
        let code = concat!(
            "/*__diffpack_import:\"./drop\"__*/require(\"./drop\");\n",
            "/*__diffpack_import:\"./keep\"__*/require(\"./keep\");\n",
        );
        let (shaken, track) = shake(
            code,
            &Names(&[]),
            &HashSet::from(["./drop".to_string()]),
            true,
        );
        assert_eq!(shaken, "require(\"./keep\");\n");
        let track = track.unwrap();
        assert!(track.line(0).is_some());
        assert!(track.line(1).is_none());
    }
}
