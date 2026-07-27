// Materialize the pinned real-application corpus.
//
//   node integration/e2e/fetch.mjs            # all apps
//   node integration/e2e/fetch.mjs next-mdx   # one app (id substring)
//   node integration/e2e/fetch.mjs --no-install
//
// Sources are cloned once (blobless sparse clone) into .cache/ and checked out
// at the SHA pinned in corpus.json. Each app's subdirectory is copied verbatim
// into apps/<id>/ — the application source is NEVER edited. The single change
// made to any app is a dependency version pin in package.json (the examples
// declare `next: "latest"` and `react: "^18"`, which are neither reproducible
// nor mutually consistent); the pinned versions live in corpus.json.
//
// A corpus entry can also be FIRST-PARTY (`"firstParty": "fixtures/<dir>"`):
// an application written by diffpack and checked into this repository, copied
// from `fixtures/` instead of cloned. It is still built twice and compared the
// same way — the app's own toolchain remains the oracle, so it is a differential
// test and not a self-assertion — but it is not third-party evidence, and both
// its provenance file and `FINDINGS.md` say so. First-party fixtures exist only
// for behaviour no pinned upstream example exercises.
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync, rmSync, cpSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const cacheDir = join(here, ".cache");
const appsDir = join(here, "apps");
const corpus = JSON.parse(readFileSync(join(here, "corpus.json"), "utf8"));

const args = process.argv.slice(2);
const noInstall = args.includes("--no-install");
const force = args.includes("--force");
const filters = args.filter((a) => !a.startsWith("--"));

const run = (cmd, cmdArgs, opts = {}) => {
  const r = spawnSync(cmd, cmdArgs, { stdio: "inherit", ...opts });
  if (r.status !== 0) throw new Error(`${cmd} ${cmdArgs.join(" ")} failed (${r.status ?? r.signal})`);
  return r;
};
const capture = (cmd, cmdArgs, opts = {}) =>
  spawnSync(cmd, cmdArgs, { encoding: "utf8", ...opts }).stdout?.trim() ?? "";

mkdirSync(cacheDir, { recursive: true });
mkdirSync(appsDir, { recursive: true });

// --- sources --------------------------------------------------------------
const wantedSources = new Set(
  corpus.apps
    .filter((a) => !filters.length || filters.some((f) => a.id.includes(f)))
    .map((a) => a.source)
);

for (const [name, src] of Object.entries(corpus.sources)) {
  if (!wantedSources.has(name)) continue;
  const dir = join(cacheDir, name);
  const sparse = src.sparse ?? ["examples"];
  if (!existsSync(join(dir, ".git"))) {
    console.log(`== clone ${name} (${src.repo})`);
    run("git", ["clone", "--filter=blob:none", "--no-checkout", src.repo, dir]);
    run("git", ["-C", dir, "sparse-checkout", "set", "--cone", ...sparse]);
  }
  let sha = src.sha;
  if (!sha || sha === "PINNED_AT_FETCH") {
    run("git", ["-C", dir, "fetch", "--depth", "1", "origin", "HEAD"]);
    sha = capture("git", ["-C", dir, "rev-parse", "FETCH_HEAD"]);
    src.sha = sha;
    writeFileSync(join(here, "corpus.json"), `${JSON.stringify(corpus, null, 2)}\n`);
    console.log(`   pinned ${name} -> ${sha}`);
  }
  const current = capture("git", ["-C", dir, "rev-parse", "HEAD"]);
  // A `--no-checkout` clone leaves an empty worktree even when HEAD already
  // points at the pinned SHA, so the worktree state is checked too.
  const checkedOut = sparse.every((path) => existsSync(join(dir, path)));
  if (current !== sha || !checkedOut) {
    console.log(`== checkout ${name}@${sha.slice(0, 12)}`);
    const fetched = spawnSync("git", ["-C", dir, "fetch", "--depth", "1", "origin", sha], { stdio: "inherit" });
    if (fetched.status !== 0) run("git", ["-C", dir, "fetch", "origin"]);
    run("git", ["-C", dir, "checkout", "--force", sha]);
  }
}

// --- apps -----------------------------------------------------------------
const pins = corpus.pins ?? {};
const failures = [];

for (const app of corpus.apps) {
  if (filters.length && !filters.some((f) => app.id.includes(f))) continue;
  // Exactly one origin per entry: a pinned upstream checkout, or a first-party
  // fixture in this repository. Neither (or both) is a corpus authoring mistake
  // and is refused rather than guessed at — an app whose origin is unclear
  // cannot be honestly described in its provenance file.
  if (Boolean(app.source) === Boolean(app.firstParty)) {
    failures.push(
      `${app.id}: a corpus entry needs exactly one of "source" (+ "subdir") or "firstParty" (a path under integration/e2e/)`
    );
    continue;
  }
  const from = app.firstParty ? join(here, app.firstParty) : join(cacheDir, app.source, app.subdir);
  const to = join(appsDir, app.id);
  if (!existsSync(from)) {
    failures.push(
      app.firstParty
        ? `${app.id}: first-party fixture missing (${app.firstParty})`
        : `${app.id}: upstream path missing (${app.subdir})`
    );
    continue;
  }
  // A first-party fixture is edited in THIS repository, so it is re-copied on
  // every fetch: a stale copy under apps/ would silently test yesterday's
  // fixture while the checked-in one says otherwise.
  const needsCopy = force || Boolean(app.firstParty) || !existsSync(join(to, "package.json"));
  const pkgPath = join(to, "package.json");
  const previousPkg = existsSync(pkgPath) ? readFileSync(pkgPath, "utf8") : null;
  if (needsCopy) {
    console.log(`== materialize ${app.id}${app.firstParty ? " (first-party fixture)" : ""}`);
    // Everything except node_modules is replaced. Deleting the whole directory
    // and copying the modules back would be hundreds of megabytes of I/O per
    // app; removing the source entries leaves exactly the same tree.
    if (existsSync(to)) {
      for (const entry of readdirSync(to)) {
        if (entry === "node_modules") continue;
        rmSync(join(to, entry), { recursive: true, force: true });
      }
    }
    cpSync(from, to, { recursive: true, filter: (src) => !src.endsWith("/node_modules") });

    // Reproducible dependency pins (the ONLY edit made to any vendored app).
    const pkg = JSON.parse(readFileSync(pkgPath, "utf8"));
    let pinned = 0;
    for (const field of ["dependencies", "devDependencies"]) {
      for (const [dep, version] of Object.entries(pkg[field] ?? {})) {
        if (pins[dep] && version !== pins[dep]) {
          pkg[field][dep] = pins[dep];
          pinned++;
        }
      }
    }
    if (pinned) writeFileSync(pkgPath, `${JSON.stringify(pkg, null, 2)}\n`);
    writeFileSync(
      join(to, "DIFFPACK_E2E_PROVENANCE.json"),
      `${JSON.stringify(
        app.firstParty
          ? {
              id: app.id,
              origin: "first-party",
              authoredBy: "diffpack (this repository)",
              fixture: `integration/e2e/${app.firstParty}`,
              why: app.firstPartyReason ?? app.why,
              oracle:
                "still built and driven twice — the app's own toolchain (next build) is the reference, diffpack is the subject — so this is a differential test, not a self-assertion",
              caveat:
                "NOT third-party evidence: written by diffpack, so it cannot show that an app nobody here wrote works",
              modifications: pinned
                ? `${pinned} dependency version(s) pinned to corpus.json "pins"`
                : "none",
            }
          : {
              id: app.id,
              origin: "third-party",
              upstream: corpus.sources[app.source].repo,
              sha: corpus.sources[app.source].sha,
              subdir: app.subdir,
              license: corpus.sources[app.source].license,
              copyright: corpus.sources[app.source].copyright,
              modifications: pinned
                ? `${pinned} dependency version(s) pinned to corpus.json "pins"; application source unmodified`
                : "none",
            },
        null,
        2
      )}\n`
    );
  }

  // Re-copying a first-party fixture does not imply reinstalling it; only a
  // changed dependency list (or a missing tree) does.
  const pkgChanged = previousPkg !== readFileSync(pkgPath, "utf8");
  if (!noInstall && ((needsCopy && pkgChanged) || !existsSync(join(to, "node_modules")))) {
    console.log(`== install ${app.id}`);
    const r = spawnSync("npm", ["install", "--no-audit", "--no-fund", "--legacy-peer-deps"], {
      cwd: to,
      stdio: ["ignore", "pipe", "pipe"],
      encoding: "utf8",
    });
    if (r.status !== 0) {
      const log = join(to, "npm-install-failure.log");
      writeFileSync(log, `${r.stdout ?? ""}\n${r.stderr ?? ""}`);
      failures.push(`${app.id}: npm install failed (see ${log})`);
      continue;
    }
    // NOTHING is installed beyond what the app itself declares. The harness used to
    // add `react-server-dom-webpack` to every app-router app because diffpack could
    // not otherwise resolve its generated entries — a diffpack requirement charged to
    // the app's dependency list, and itself a recorded finding. diffpack now resolves
    // the flight runtime from the copy `next` vendors, exactly as Next does, so the
    // apps are installed as published. If that regresses, the app-router builds fail
    // with `cannot resolve "react-server-dom-webpack/client"` — do not re-add the
    // install to paper over it.
  }
}

if (failures.length) {
  console.error(`\n${failures.length} app(s) could not be materialized:`);
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
console.log("\ncorpus ready");
