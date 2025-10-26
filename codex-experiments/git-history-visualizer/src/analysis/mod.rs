use std::{
    collections::{HashMap, HashSet},
    fs::{self, File},
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
};

use anyhow::{Context, Result, anyhow, ensure};
use chrono::{TimeZone, Utc};
use git2::{
    BlameOptions, Commit, Mailmap, ObjectType, Oid, Repository, Signature, Sort, TreeWalkMode,
    TreeWalkResult,
};
use rayon::{ThreadPool, prelude::*};
use serde_json::json;

mod filetypes;
mod filter;

use filter::FileFilter;

#[derive(Debug, Clone)]
pub struct AnalyzeConfig {
    pub repo: PathBuf,
    pub cohort_format: String,
    pub interval_secs: u64,
    pub ignore_patterns: Vec<String>,
    pub only_patterns: Vec<String>,
    pub outdir: PathBuf,
    pub branch: String,
    pub all_filetypes: bool,
    pub ignore_whitespace: bool,
    pub quiet: bool,
    pub jobs: usize,
    pub opt: bool,
}

impl AnalyzeConfig {
    pub fn validate(self) -> Result<Self> {
        ensure!(
            self.repo.exists(),
            "Repository path {:?} does not exist",
            self.repo
        );
        ensure!(
            self.repo.join(".git").exists(),
            "Repository {:?} does not look like a git repository (missing .git)",
            self.repo
        );
        Ok(self)
    }
}

pub fn analyze(config: AnalyzeConfig) -> Result<()> {
    let config = config.validate()?;
    perform_analysis(config)
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum CurveKeyKind {
    Cohort,
    Author,
    Domain,
    Ext,
    Dir,
    Sha,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CurveKey {
    kind: CurveKeyKind,
    label: String,
}

impl CurveKey {
    fn new(kind: CurveKeyKind, label: String) -> Self {
        Self { kind, label }
    }

    fn cohort(label: String) -> Self {
        Self::new(CurveKeyKind::Cohort, label)
    }

    fn author(label: String) -> Self {
        Self::new(CurveKeyKind::Author, label)
    }

    fn domain(label: String) -> Self {
        Self::new(CurveKeyKind::Domain, label)
    }

    fn extension(label: String) -> Self {
        Self::new(CurveKeyKind::Ext, label)
    }

    fn directory(label: String) -> Self {
        Self::new(CurveKeyKind::Dir, label)
    }

    fn sha(label: String) -> Self {
        Self::new(CurveKeyKind::Sha, label)
    }

    fn kind(&self) -> CurveKeyKind {
        self.kind
    }

    fn label(&self) -> &str {
        &self.label
    }
}

#[allow(dead_code)]
#[derive(Debug)]
struct MasterSample {
    commit_id: Oid,
    committed_time: i64,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct FileEntry {
    path: String,
    blob_id: Oid,
}

fn perform_analysis(config: AnalyzeConfig) -> Result<()> {
    let repo = Repository::open(&config.repo).with_context(|| {
        format!(
            "Unable to open repository at {}",
            config.repo.to_string_lossy()
        )
    })?;

    if config.opt {
        run_commit_graph(&config.repo, config.quiet)
            .context("Running git commit-graph optimization")?;
    }

    let mailmap = repo.mailmap().ok();
    let (branch_name, branch_ref_name) =
        resolve_branch(&repo, &config.branch, config.quiet).context("Resolving branch")?;

    let mut revwalk = repo.revwalk().context("Creating revision walk")?;
    revwalk
        .push_ref(&branch_ref_name)
        .with_context(|| format!("Setting revwalk start at {branch_ref_name}"))?;
    revwalk
        .set_sorting(Sort::TOPOLOGICAL | Sort::TIME)
        .context("Configuring revwalk sorting")?;

    let mut commit2cohort_map: HashMap<Oid, String> = HashMap::new();
    let mut curve_keys: HashSet<CurveKey> = HashSet::new();

    for oid in revwalk {
        let oid = oid?;
        let commit = repo.find_commit(oid)?;
        let cohort = format_cohort(commit.time().seconds(), &config.cohort_format)?;
        curve_keys.insert(CurveKey::cohort(cohort.clone()));
        commit2cohort_map.insert(commit.id(), cohort);

        let (author, domain) = resolve_author(&commit, mailmap.as_ref());
        curve_keys.insert(CurveKey::author(author));
        curve_keys.insert(CurveKey::domain(domain));
    }

    let head_commit = repo
        .find_reference(&branch_ref_name)?
        .peel_to_commit()
        .context("Unable to resolve head commit for branch")?;

    let master_samples = collect_master_commits(&head_commit, config.interval_secs as i64)?;

    let filter = FileFilter::new(
        config.all_filetypes,
        &config.only_patterns,
        &config.ignore_patterns,
    )?;

    let mut commit_entries: Vec<Vec<FileEntry>> = Vec::new();
    for sample in &master_samples {
        let commit = repo
            .find_commit(sample.commit_id)
            .with_context(|| format!("Unable to load commit {}", sample.commit_id))?;
        let entries = collect_commit_entries(&commit, &filter, &mut curve_keys)?;
        commit_entries.push(entries);
    }

    let commit2cohort = Arc::new(commit2cohort_map);
    let use_mailmap = mailmap.is_some();

    if !config.quiet {
        let total_files: usize = commit_entries.iter().map(|entries| entries.len()).sum();
        eprintln!(
            "Collected {} commits on branch '{}' ({} sampled commits for analysis); tracking {} files",
            commit2cohort.len(),
            branch_name,
            master_samples.len(),
            total_files
        );
    }

    compute_curves(
        &repo,
        &config.repo,
        &commit_entries,
        &mut curve_keys,
        Arc::clone(&commit2cohort),
        &master_samples,
        use_mailmap,
        &config,
    )
}

fn resolve_branch(repo: &Repository, requested: &str, quiet: bool) -> Result<(String, String)> {
    let requested_ref = format!("refs/heads/{requested}");
    match repo.find_reference(&requested_ref) {
        Ok(reference) => {
            let name = reference
                .name()
                .ok_or_else(|| anyhow!("Reference {requested_ref} has no name"))?
                .to_string();
            Ok((requested.to_string(), name))
        }
        Err(_) => {
            let head = repo.head().context("Unable to resolve HEAD")?;
            let fallback_name = head
                .shorthand()
                .ok_or_else(|| anyhow!("HEAD is detached; please specify a branch"))?
                .to_string();
            if !quiet {
                eprintln!(
                    "Requested branch '{}' does not exist. Falling back to default branch '{}'",
                    requested, fallback_name
                );
            }
            let name = head
                .name()
                .ok_or_else(|| anyhow!("HEAD reference name missing"))?
                .to_string();
            Ok((fallback_name, name))
        }
    }
}

fn format_cohort(seconds: i64, format: &str) -> Result<String> {
    let dt = Utc
        .timestamp_opt(seconds, 0)
        .single()
        .ok_or_else(|| anyhow!("Invalid timestamp"))?;
    Ok(dt.format(format).to_string())
}

fn resolve_author(commit: &Commit<'_>, mailmap: Option<&Mailmap>) -> (String, String) {
    let signature = commit.author();
    let mapped = mailmap
        .and_then(|map| map.resolve_signature(&signature).ok())
        .unwrap_or_else(|| signature.to_owned());
    let name = mapped.name().unwrap_or("").to_string();
    let email = mapped.email().unwrap_or("").to_string();
    let domain = email.split('@').nth(1).unwrap_or("").to_string();
    (name, domain)
}

fn compute_curves(
    repo: &Repository,
    repo_path: &Path,
    commit_entries: &[Vec<FileEntry>],
    curve_keys: &mut HashSet<CurveKey>,
    commit2cohort: Arc<HashMap<Oid, String>>,
    master_samples: &[MasterSample],
    use_mailmap: bool,
    config: &AnalyzeConfig,
) -> Result<()> {
    let jobs = usize::max(1, config.jobs);
    let pool = if jobs > 1 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(jobs)
                .build()
                .context("Failed to create rayon thread pool")?,
        )
    } else {
        None
    };

    let mut last_file_y: HashMap<String, HashMap<CurveKey, i64>> = HashMap::new();
    let mut cur_y: HashMap<CurveKey, i64> = HashMap::new();
    let mut curves: HashMap<CurveKey, Vec<i64>> = HashMap::new();
    let mut ts: Vec<String> = Vec::new();
    let mut commit_history: HashMap<String, Vec<(i64, i64)>> = HashMap::new();
    let mut last_file_hash: HashMap<String, Oid> = HashMap::new();

    for (idx, sample) in master_samples.iter().enumerate() {
        let commit = repo
            .find_commit(sample.commit_id)
            .with_context(|| format!("Unable to reload commit {}", sample.commit_id))?;
        let entries = &commit_entries[idx];

        let mut cur_file_hash: HashMap<String, Oid> = HashMap::new();
        let mut check_entries: Vec<FileEntry> = Vec::new();

        for entry in entries {
            cur_file_hash.insert(entry.path.clone(), entry.blob_id);
            match last_file_hash.get(&entry.path) {
                Some(prev) if prev == &entry.blob_id => {}
                Some(_) => {
                    if let Some(prev_map) = last_file_y.get(&entry.path) {
                        for (key, count) in prev_map {
                            *cur_y.entry(key.clone()).or_insert(0) -= count;
                        }
                    }
                    check_entries.push(entry.clone());
                }
                None => {
                    check_entries.push(entry.clone());
                }
            };
        }

        let previous_paths: Vec<String> = last_file_hash.keys().cloned().collect();
        for deleted_path in previous_paths {
            if !cur_file_hash.contains_key(&deleted_path) {
                if let Some(prev_map) = last_file_y.remove(&deleted_path) {
                    for (key, count) in prev_map {
                        *cur_y.entry(key).or_insert(0) -= count;
                    }
                }
            }
        }

        last_file_hash = cur_file_hash;

        let blame_results = run_blame_for_entries(
            repo_path,
            commit.id(),
            &check_entries,
            Arc::clone(&commit2cohort),
            config.ignore_whitespace,
            use_mailmap,
            jobs,
            pool.as_ref(),
        )?;

        for (path, file_histogram) in blame_results {
            for key in file_histogram.keys() {
                curve_keys.insert(key.clone());
            }
            for (key, count) in &file_histogram {
                *cur_y.entry(key.clone()).or_insert(0) += *count;
            }
            last_file_y.insert(path, file_histogram);
        }

        let timestamp_secs = commit.time().seconds();
        ts.push(format_timestamp(timestamp_secs)?);

        for key in curve_keys.iter() {
            let value = *cur_y.get(key).unwrap_or(&0);
            curves.entry(key.clone()).or_default().push(value);
        }

        update_commit_history(&mut commit_history, &cur_y, timestamp_secs);
    }

    write_outputs(curve_keys, &curves, &ts, &commit_history, config)
}

fn collect_master_commits(head: &Commit<'_>, interval_secs: i64) -> Result<Vec<MasterSample>> {
    let mut samples = Vec::new();
    let mut current = head.clone();
    let mut last_timestamp: Option<i64> = None;
    loop {
        let ts = current.time().seconds();
        if last_timestamp.is_none() || ts < last_timestamp.unwrap() - interval_secs {
            samples.push(MasterSample {
                commit_id: current.id(),
                committed_time: ts,
            });
            last_timestamp = Some(ts);
        }
        if current.parent_count() == 0 {
            break;
        }
        current = current.parent(0)?;
    }
    samples.reverse();
    Ok(samples)
}

fn run_commit_graph(repo_path: &Path, _quiet: bool) -> Result<()> {
    let status = Command::new("git")
        .arg("commit-graph")
        .arg("write")
        .arg("--changed-paths")
        .current_dir(repo_path)
        .status()
        .context("Failed to invoke git commit-graph")?;

    if !status.success() {
        let code = status.code().unwrap_or(-1);
        return Err(anyhow!(
            "git commit-graph exited with status {}; see stderr for details",
            code
        ));
    }
    Ok(())
}

fn collect_commit_entries(
    commit: &Commit<'_>,
    filter: &FileFilter,
    curve_keys: &mut HashSet<CurveKey>,
) -> Result<Vec<FileEntry>> {
    let tree = commit.tree().context("Failed to load commit tree")?;
    let mut entries = Vec::new();

    tree.walk(TreeWalkMode::PreOrder, |root, entry| {
        let name = match entry.name() {
            Some(n) => n,
            None => return TreeWalkResult::Ok,
        };

        if entry.kind() != Some(ObjectType::Blob) {
            return TreeWalkResult::Ok;
        }

        let path = if root.is_empty() {
            name.to_string()
        } else {
            format!("{root}{name}")
        };

        if !filter.matches(&path, name) {
            return TreeWalkResult::Ok;
        }

        curve_keys.insert(CurveKey::extension(file_extension(&path)));
        curve_keys.insert(CurveKey::directory(top_directory(&path)));

        entries.push(FileEntry {
            path,
            blob_id: entry.id(),
        });

        TreeWalkResult::Ok
    })
    .context("Traversing commit tree")?;

    Ok(entries)
}

fn file_extension(path: &str) -> String {
    let path_obj = Path::new(path);
    match path_obj.file_name().and_then(|name| name.to_str()) {
        Some(file_name) if !file_name.starts_with('.') => {
            match path_obj.extension().and_then(|ext| ext.to_str()) {
                Some(ext) => format!(".{ext}"),
                None => String::new(),
            }
        }
        _ => String::new(),
    }
}

fn top_directory(path: &str) -> String {
    match path.rsplit_once('/') {
        Some((prefix, _)) => match prefix.split('/').next() {
            Some(dir) if !dir.is_empty() => format!("{dir}/"),
            _ => "/".to_string(),
        },
        None => "/".to_string(),
    }
}

fn run_blame_for_entries(
    repo_path: &Path,
    commit_id: Oid,
    entries: &[FileEntry],
    commit2cohort: Arc<HashMap<Oid, String>>,
    ignore_whitespace: bool,
    use_mailmap: bool,
    jobs: usize,
    pool: Option<&ThreadPool>,
) -> Result<HashMap<String, HashMap<CurveKey, i64>>> {
    if entries.is_empty() {
        return Ok(HashMap::new());
    }

    let jobs = usize::max(1, jobs);

    let collected: Vec<Option<(String, HashMap<CurveKey, i64>)>> = if jobs <= 1 || pool.is_none() {
        let repo = Repository::open(repo_path)
            .with_context(|| format!("Opening repository at {}", repo_path.display()))?;
        entries
            .iter()
            .map(|entry| {
                blame_entry(
                    &repo,
                    commit_id,
                    entry,
                    &commit2cohort,
                    ignore_whitespace,
                    use_mailmap,
                )
            })
            .collect::<Result<Vec<_>>>()?
    } else if let Some(pool) = pool {
        let repo_path_buf = repo_path.to_path_buf();
        pool.install(|| {
            entries
                .par_iter()
                .map_init(
                    || {
                        Repository::open(&repo_path_buf).unwrap_or_else(|err| {
                            panic!(
                                "Failed to open repository {}: {err}",
                                repo_path_buf.display()
                            )
                        })
                    },
                    |repo, entry| {
                        blame_entry(
                            repo,
                            commit_id,
                            entry,
                            &commit2cohort,
                            ignore_whitespace,
                            use_mailmap,
                        )
                    },
                )
                .collect::<Result<Vec<_>>>()
        })?
    } else {
        unreachable!();
    };

    let mut results = HashMap::new();
    for item in collected {
        if let Some((path, histogram)) = item {
            results.insert(path, histogram);
        }
    }
    Ok(results)
}

fn blame_entry(
    repo: &Repository,
    commit_id: Oid,
    entry: &FileEntry,
    commit2cohort: &HashMap<Oid, String>,
    ignore_whitespace: bool,
    use_mailmap: bool,
) -> Result<Option<(String, HashMap<CurveKey, i64>)>> {
    let mut options = BlameOptions::new();
    options.newest_commit(commit_id);
    if ignore_whitespace {
        options.ignore_whitespace(true);
    }
    if use_mailmap {
        options.use_mailmap(true);
    }

    let blame = match repo.blame_file(Path::new(&entry.path), Some(&mut options)) {
        Ok(blame) => blame,
        Err(_) => return Ok(None),
    };

    let ext_key = CurveKey::extension(file_extension(&entry.path));
    let dir_key = CurveKey::directory(top_directory(&entry.path));
    let mailmap = if use_mailmap {
        repo.mailmap().ok()
    } else {
        None
    };

    let mut histogram: HashMap<CurveKey, i64> = HashMap::new();

    for hunk in blame.iter() {
        let lines = hunk.lines_in_hunk() as i64;
        let final_oid = hunk.final_commit_id();

        let cohort_label = commit2cohort
            .get(&final_oid)
            .cloned()
            .unwrap_or_else(|| "MISSING".to_string());
        accumulate(&mut histogram, CurveKey::cohort(cohort_label), lines);
        accumulate(&mut histogram, ext_key.clone(), lines);
        accumulate(&mut histogram, dir_key.clone(), lines);

        let (author, domain) = canonical_author(hunk.final_signature(), mailmap.as_ref());
        accumulate(&mut histogram, CurveKey::author(author), lines);
        accumulate(&mut histogram, CurveKey::domain(domain), lines);

        if commit2cohort.contains_key(&final_oid) {
            let sha_key = CurveKey::sha(final_oid.to_string());
            accumulate(&mut histogram, sha_key, lines);
        }
    }

    Ok(Some((entry.path.clone(), histogram)))
}

fn canonical_author(signature: Signature<'_>, mailmap: Option<&Mailmap>) -> (String, String) {
    let resolved = mailmap
        .and_then(|map| map.resolve_signature(&signature).ok())
        .unwrap_or_else(|| signature.to_owned());
    let name = resolved.name().unwrap_or("").to_string();
    let email = resolved.email().unwrap_or("").to_string();
    let domain = email.split('@').nth(1).unwrap_or("").to_string();
    (name, domain)
}

fn accumulate(map: &mut HashMap<CurveKey, i64>, key: CurveKey, delta: i64) {
    if delta == 0 {
        return;
    }
    *map.entry(key).or_insert(0) += delta;
}

fn update_commit_history(
    commit_history: &mut HashMap<String, Vec<(i64, i64)>>,
    cur_y: &HashMap<CurveKey, i64>,
    timestamp_secs: i64,
) {
    for (key, value) in cur_y {
        if key.kind() == CurveKeyKind::Sha {
            commit_history
                .entry(key.label().to_string())
                .or_default()
                .push((timestamp_secs, *value));
        }
    }
}

fn write_outputs(
    curve_keys: &HashSet<CurveKey>,
    curves: &HashMap<CurveKey, Vec<i64>>,
    ts: &[String],
    commit_history: &HashMap<String, Vec<(i64, i64)>>,
    config: &AnalyzeConfig,
) -> Result<()> {
    fs::create_dir_all(&config.outdir)
        .with_context(|| format!("Creating output directory {}", config.outdir.display()))?;

    write_curve_file(
        &config.outdir,
        "cohorts.json",
        CurveKeyKind::Cohort,
        curve_keys,
        curves,
        ts,
        |label| format!("Code added in {label}"),
    )?;
    write_curve_file(
        &config.outdir,
        "exts.json",
        CurveKeyKind::Ext,
        curve_keys,
        curves,
        ts,
        |label| label.to_string(),
    )?;
    write_curve_file(
        &config.outdir,
        "authors.json",
        CurveKeyKind::Author,
        curve_keys,
        curves,
        ts,
        |label| label.to_string(),
    )?;
    write_curve_file(
        &config.outdir,
        "dirs.json",
        CurveKeyKind::Dir,
        curve_keys,
        curves,
        ts,
        |label| label.to_string(),
    )?;
    write_curve_file(
        &config.outdir,
        "domains.json",
        CurveKeyKind::Domain,
        curve_keys,
        curves,
        ts,
        |label| label.to_string(),
    )?;

    write_survival_data(&config.outdir, commit_history)?;
    Ok(())
}

fn write_curve_file<F>(
    outdir: &Path,
    filename: &str,
    kind: CurveKeyKind,
    curve_keys: &HashSet<CurveKey>,
    curves: &HashMap<CurveKey, Vec<i64>>,
    ts: &[String],
    label_fn: F,
) -> Result<()>
where
    F: Fn(&str) -> String,
{
    let mut keys: Vec<CurveKey> = curve_keys
        .iter()
        .filter(|key| key.kind() == kind)
        .cloned()
        .collect();
    keys.sort_by(|a, b| a.label().cmp(b.label()));

    let y: Vec<Vec<i64>> = keys
        .iter()
        .map(|key| {
            curves
                .get(key)
                .cloned()
                .unwrap_or_else(|| vec![0; ts.len()])
        })
        .collect();
    let labels: Vec<String> = keys.iter().map(|key| label_fn(key.label())).collect();

    let output = json!({
        "y": y,
        "ts": ts,
        "labels": labels,
    });

    let mut file = File::create(outdir.join(filename))
        .with_context(|| format!("Writing {}", outdir.join(filename).display()))?;
    serde_json::to_writer_pretty(&mut file, &output)
        .with_context(|| format!("Serialising {}", outdir.join(filename).display()))?;
    Ok(())
}

fn write_survival_data(
    outdir: &Path,
    commit_history: &HashMap<String, Vec<(i64, i64)>>,
) -> Result<()> {
    let mut file = File::create(outdir.join("survival.json"))
        .with_context(|| format!("Writing {}", outdir.join("survival.json").display()))?;

    let mut map = serde_json::Map::new();
    for (sha, history) in commit_history {
        let points: Vec<_> = history.iter().map(|(t, c)| json!([t, c])).collect();
        map.insert(sha.clone(), json!(points));
    }

    serde_json::to_writer_pretty(&mut file, &serde_json::Value::Object(map))
        .context("Serialising survival data")
}

fn format_timestamp(seconds: i64) -> Result<String> {
    let dt = Utc
        .timestamp_opt(seconds, 0)
        .single()
        .ok_or_else(|| anyhow!("Invalid timestamp"))?;
    Ok(dt.format("%Y-%m-%dT%H:%M:%S").to_string())
}
