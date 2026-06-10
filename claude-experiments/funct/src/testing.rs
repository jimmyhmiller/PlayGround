//! Test runner for tests written in funct itself.
//!
//! Tests are marked with an annotation, right next to the code they test:
//!
//! ```text
//! export fn area(s) = ...
//!
//! #[test]
//! fn squares_have_side_squared_area() {
//!     assert_eq(area(square(3.0)), 9.0)
//! }
//! ```
//!
//! `funct test <file-or-dir>` runs them; assertions are the prelude's
//! `assert` / `assert_eq` / `assert_ne` / `fail`, which fault with the
//! failing function and line. An annotated fn is an ordinary fn otherwise —
//! importing a module never runs its tests.
//!
//! Each test runs in a FRESH engine (the file is re-evaluated), so tests are
//! fully isolated: top-level atoms and state can't leak between them. The
//! file's directory is the module root, so tests can use the file's own
//! imports — or test files can `import { thing } from "module_under_test"`.

use crate::vm::Funct;
use std::path::{Path, PathBuf};

pub struct TestOutcome {
    pub name: String,
    /// None = passed; Some(message) = failed
    pub error: Option<String>,
}

pub struct TestReport {
    pub file: PathBuf,
    /// error loading/evaluating the file itself (no tests ran)
    pub load_error: Option<String>,
    pub outcomes: Vec<TestOutcome>,
}

impl TestReport {
    pub fn passed(&self) -> usize {
        self.outcomes.iter().filter(|o| o.error.is_none()).count()
    }

    pub fn failed(&self) -> usize {
        self.outcomes.iter().filter(|o| o.error.is_some()).count()
    }

    pub fn ok(&self) -> bool {
        self.load_error.is_none() && self.failed() == 0
    }
}

/// A file → itself; a directory → every `*.ft` under it (recursive, sorted).
/// Files without `#[test]` functions are skipped silently when scanning a
/// directory.
pub fn discover_files(path: &Path) -> Vec<PathBuf> {
    if path.is_file() {
        return vec![path.to_path_buf()];
    }
    let mut found = Vec::new();
    collect_ft_files(path, &mut found);
    found.sort();
    found
}

fn collect_ft_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect_ft_files(&p, out);
        } else if p.extension().and_then(|e| e.to_str()) == Some("ft") {
            out.push(p);
        }
    }
}

fn fresh_engine_for(path: &Path) -> Funct {
    let mut vm = Funct::new();
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            vm.set_module_root(dir);
        }
    }
    vm
}

/// Run every `#[test]` function in one file, each in a fresh engine.
pub fn run_test_file(path: &Path) -> TestReport {
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            return TestReport {
                file: path.to_path_buf(),
                load_error: Some(format!("cannot read file: {}", e)),
                outcomes: vec![],
            }
        }
    };

    // discovery pass: evaluate once, list the #[test] fns
    let names = {
        let mut vm = fresh_engine_for(path);
        if let Err(e) = vm.eval(&src) {
            return TestReport {
                file: path.to_path_buf(),
                load_error: Some(e.to_string()),
                outcomes: vec![],
            };
        }
        vm.test_names()
    };

    let mut outcomes = Vec::new();
    for name in names {
        let mut vm = fresh_engine_for(path);
        let error = match vm.eval(&src) {
            Err(e) => Some(e.to_string()),
            Ok(_) => match vm.call(&name, vec![]) {
                Ok(_) => None,
                Err(e) => Some(e.to_string()),
            },
        };
        outcomes.push(TestOutcome { name, error });
    }
    TestReport { file: path.to_path_buf(), load_error: None, outcomes }
}

/// Run a file or directory of tests, printing cargo-test-style output.
/// Returns true when everything passed.
pub fn run_and_print(path: &Path) -> bool {
    let scanning_dir = path.is_dir();
    let files = discover_files(path);
    if files.is_empty() {
        eprintln!("no .ft files found under {}", path.display());
        return false;
    }
    let mut passed = 0;
    let mut failed = 0;
    let mut any = false;
    for file in files {
        // when scanning a directory, don't evaluate files that can't contain
        // tests — evaluating runs their top-level code (side effects)
        if scanning_dir {
            match std::fs::read_to_string(&file) {
                Ok(src) if src.contains("#[test]") => {}
                _ => continue,
            }
        }
        let report = run_test_file(&file);
        if let Some(err) = &report.load_error {
            // a file that doesn't load is always a failure, even in a scan
            println!("{}", report.file.display());
            println!("  FAILED to load: {}", err);
            failed += 1;
            any = true;
            continue;
        }
        if report.outcomes.is_empty() {
            if !scanning_dir {
                println!("{}", report.file.display());
                println!("  (no #[test] functions)");
            }
            continue;
        }
        any = true;
        println!("{}", report.file.display());
        for o in &report.outcomes {
            match &o.error {
                None => println!("  test {} ... ok", o.name),
                Some(e) => {
                    println!("  test {} ... FAILED", o.name);
                    for line in e.lines() {
                        println!("      {}", line);
                    }
                }
            }
        }
        passed += report.passed();
        failed += report.failed();
    }
    if !any && scanning_dir {
        println!("no #[test] functions found under {}", path.display());
    }
    println!();
    println!(
        "test result: {}. {} passed; {} failed",
        if failed == 0 { "ok" } else { "FAILED" },
        passed,
        failed
    );
    failed == 0
}
