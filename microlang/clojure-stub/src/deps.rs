//! Maven dependency resolution for deps.edn `:mvn/version` coordinates —
//! the local repository (`~/.m2/repository`, override with `$MICROCLJ_M2`)
//! first, Maven Central otherwise (downloading INTO the local repo, like
//! tools.deps). Transitive dependencies come from a deliberately naive scan
//! of each artifact's pom: compile-scope, non-optional `<dependency>`
//! entries, with `<dependencyManagement>`/`<profiles>`/`<build>` sections
//! ignored and property-interpolated versions (`${…}`) skipped with a
//! warning. Version conflicts resolve first-request-wins (breadth-first) —
//! not tools.deps' full algorithm, but honest about it.
//!
//! org.clojure/clojure (and spec) are always excluded: this runtime IS the
//! Clojure implementation.

use std::collections::HashSet;
use std::io::Read;
use std::path::PathBuf;

/// Resolve `group/artifact@version` plus its transitive compile deps to a
/// list of JAR paths (in the local repo, downloading as needed).
pub fn resolve_mvn(group: &str, artifact: &str, version: &str) -> Result<Vec<PathBuf>, String> {
    let mut jars = Vec::new();
    let mut seen: HashSet<(String, String)> = HashSet::new();
    let mut queue = vec![(group.to_string(), artifact.to_string(), version.to_string())];
    while let Some((g, a, v)) = queue.pop() {
        if !seen.insert((g.clone(), a.clone())) {
            continue; // first version requested wins
        }
        let (jar, pom) = ensure_artifact(&g, &a, &v)?;
        jars.push(jar);
        let pom_text = std::fs::read_to_string(&pom)
            .map_err(|e| format!("mvn: reading {}: {e}", pom.display()))?;
        for (dg, da, dv) in pom_compile_deps(&pom_text) {
            if excluded(&dg, &da) || seen.contains(&(dg.clone(), da.clone())) {
                continue;
            }
            match dv {
                Some(dv) if !dv.contains("${") => queue.push((dg, da, dv)),
                other => eprintln!(
                    "microclj: warning: skipping transitive dep {dg}/{da} \
                     (unresolvable version {other:?} in {g}/{a}'s pom)"
                ),
            }
        }
    }
    Ok(jars)
}

/// We ARE Clojure; never pull the reference implementation (or its spec libs).
fn excluded(group: &str, artifact: &str) -> bool {
    group == "org.clojure"
        && matches!(artifact, "clojure" | "spec.alpha" | "core.specs.alpha")
}

fn m2_repository() -> PathBuf {
    if let Ok(p) = std::env::var("MICROCLJ_M2") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".m2/repository")
}

/// The default remote repositories, in search order — Maven Central then
/// Clojars, exactly tools.deps' defaults.
const REPOS: &[&str] = &["https://repo1.maven.org/maven2", "https://repo.clojars.org"];

/// The artifact's jar + pom in the local repo, downloading both (from the
/// first repository that has them) if absent.
fn ensure_artifact(g: &str, a: &str, v: &str) -> Result<(PathBuf, PathBuf), String> {
    let rel = format!("{}/{a}/{v}", g.replace('.', "/"));
    let dir = m2_repository().join(&rel);
    let jar = dir.join(format!("{a}-{v}.jar"));
    let pom = dir.join(format!("{a}-{v}.pom"));
    if !jar.is_file() || !pom.is_file() {
        let mut errs = Vec::new();
        let mut ok = false;
        for repo in REPOS {
            let jar_url = format!("{repo}/{rel}/{a}-{v}.jar");
            let pom_url = format!("{repo}/{rel}/{a}-{v}.pom");
            match download(&jar_url, &jar).and_then(|_| download(&pom_url, &pom)) {
                Ok(()) => {
                    ok = true;
                    break;
                }
                Err(e) => errs.push(e),
            }
        }
        if !ok {
            return Err(format!("mvn: cannot resolve {g}/{a} {v}: {}", errs.join("; ")));
        }
    }
    Ok((jar, pom))
}

fn download(url: &str, dest: &PathBuf) -> Result<(), String> {
    if dest.is_file() {
        return Ok(());
    }
    eprintln!("microclj: downloading {url}");
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mvn: mkdir: {e}"))?;
    }
    let resp = ureq::get(url)
        .call()
        .map_err(|e| format!("mvn: fetching {url}: {e}"))?;
    let mut bytes = Vec::new();
    resp.into_body()
        .into_reader()
        .read_to_end(&mut bytes)
        .map_err(|e| format!("mvn: reading {url}: {e}"))?;
    std::fs::write(dest, bytes).map_err(|e| format!("mvn: writing {}: {e}", dest.display()))?;
    Ok(())
}

/// Compile-scope, non-optional `<dependency>` coordinates from a pom, with
/// the sections that don't declare REAL dependencies stripped first.
fn pom_compile_deps(pom: &str) -> Vec<(String, String, Option<String>)> {
    let mut s = pom.to_string();
    for section in ["dependencyManagement", "profiles", "build", "plugins"] {
        while let Some(start) = s.find(&format!("<{section}>")) {
            match s[start..].find(&format!("</{section}>")) {
                Some(off) => {
                    let end = start + off + section.len() + 3;
                    s.replace_range(start..end, "");
                }
                None => break,
            }
        }
    }
    let mut out = Vec::new();
    let mut rest = s.as_str();
    while let Some(start) = rest.find("<dependency>") {
        let Some(off) = rest[start..].find("</dependency>") else { break };
        let body = &rest[start + "<dependency>".len()..start + off];
        let tag = |name: &str| -> Option<String> {
            let open = format!("<{name}>");
            let close = format!("</{name}>");
            let b = body.find(&open)? + open.len();
            let e = body[b..].find(&close)? + b;
            Some(body[b..e].trim().to_string())
        };
        let scope = tag("scope").unwrap_or_else(|| "compile".into());
        let optional = tag("optional").as_deref() == Some("true");
        if (scope == "compile" || scope == "runtime") && !optional {
            if let (Some(g), Some(a)) = (tag("groupId"), tag("artifactId")) {
                out.push((g, a, tag("version")));
            }
        }
        rest = &rest[start + off + "</dependency>".len()..];
    }
    out
}

/// Read a namespace's source out of a JAR on the load path (`rel` is the
/// slash-form path without extension, e.g. `nrepl/bencode`).
pub fn jar_source(jar: &std::path::Path, rel: &str) -> Option<String> {
    let f = std::fs::File::open(jar).ok()?;
    let mut zip = zip::ZipArchive::new(f).ok()?;
    for ext in ["clj", "cljc", "cljs"] {
        if let Ok(mut entry) = zip.by_name(&format!("{rel}.{ext}")) {
            let mut s = String::new();
            if entry.read_to_string(&mut s).is_ok() {
                return Some(s);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    #[test]
    fn pom_scan_takes_compile_skips_test_and_management() {
        let pom = r#"<project>
          <dependencyManagement><dependencies>
            <dependency><groupId>x</groupId><artifactId>managed</artifactId><version>9</version></dependency>
          </dependencies></dependencyManagement>
          <dependencies>
            <dependency><groupId>g1</groupId><artifactId>a1</artifactId><version>1.0</version></dependency>
            <dependency><groupId>org.clojure</groupId><artifactId>clojure</artifactId><version>1.12.0</version><scope>provided</scope></dependency>
            <dependency><groupId>g2</groupId><artifactId>a2</artifactId><version>2.0</version><scope>test</scope></dependency>
            <dependency><groupId>g3</groupId><artifactId>a3</artifactId><version>3.0</version><optional>true</optional></dependency>
          </dependencies>
        </project>"#;
        let deps = super::pom_compile_deps(pom);
        assert_eq!(deps, vec![("g1".into(), "a1".into(), Some("1.0".into()))]);
    }
}
