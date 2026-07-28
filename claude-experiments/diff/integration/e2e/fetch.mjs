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
// A corpus entry can also be a MONOREPO app (`"monorepo": { "appDir", … }`): a
// workspace whose application cannot be lifted out of its repository, because it
// depends on sibling workspace packages that resolve only from the repository
// root. Such an entry materializes the WHOLE checkout and builds in place — see
// `lib/apps.mjs`'s corpus-layout section. Its dependencies are installed by the
// package manager the repository declares (`"packageManager": "corepack yarn"`),
// at the directory that manager hoists to (`"installAt": "root"`).
//
// An entry may also declare `"env"`: values its build REQUIRES. They are written
// to the app's `.env` and recorded in its provenance file, because a build that
// only works because of something ambient in the developer's shell is neither
// reproducible nor auditable. They are dummies, and the corpus tests refuse
// anything that looks like a real secret.
//
// `"heavy": true` marks an entry too large for the default corpus (cal.com is a
// 349 MB checkout and a ~20-minute install); it is materialized only with --heavy.
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
import {
  appDirOf,
  envFileOf,
  installDirOf,
  licenseMatches,
  materializedRootOf,
  renderEnvFile,
  selectApps,
} from "./lib/apps.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const cacheDir = join(here, ".cache");
const appsDir = join(here, "apps");
const corpus = JSON.parse(readFileSync(join(here, "corpus.json"), "utf8"));

const args = process.argv.slice(2);
const noInstall = args.includes("--no-install");
const force = args.includes("--force");
const heavy = args.includes("--heavy");
const filters = args.filter((a) => !a.startsWith("--"));
const { selected: selectedApps, excludedHeavy } = selectApps(corpus.apps, { filters, heavy });
if (excludedHeavy.length) {
  console.log(
    `note: skipping ${excludedHeavy.length} heavy app(s) — ${excludedHeavy
      .map((a) => a.id)
      .join(", ")}. Pass --heavy to materialize them (gigabytes, tens of minutes).`
  );
}
if (!selectedApps.length) {
  console.error(
    filters.length
      ? `no apps matched [${filters.join(", ")}]${excludedHeavy.length ? " (the matches are heavy: pass --heavy)" : ""}`
      : "corpus.json declares no apps"
  );
  process.exit(2);
}

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
const wantedSources = new Set(selectedApps.map((a) => a.source));

for (const [name, src] of Object.entries(corpus.sources)) {
  if (!wantedSources.has(name)) continue;
  const dir = join(cacheDir, name);
  // `"sparse": []` means the WHOLE repository: a monorepo app is built inside its
  // own checkout, so there is no subdirectory to narrow to.
  const sparse = src.sparse ?? ["examples"];
  const full = sparse.length === 0;
  if (!existsSync(join(dir, ".git"))) {
    console.log(`== clone ${name} (${src.repo})`);
    run("git", ["clone", "--filter=blob:none", "--no-checkout", src.repo, dir]);
    if (!full) run("git", ["-C", dir, "sparse-checkout", "set", "--cone", ...sparse]);
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
  const checkedOut = full
    ? existsSync(join(dir, "package.json"))
    : sparse.every((path) => existsSync(join(dir, path)));
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

// Declared build inputs. Written on every fetch (not only when the tree is
// re-materialized) so the file on disk always says what corpus.json says, and
// returned so the provenance file can record exactly what the build was given.
const writeDeclaredEnv = (app) => {
  if (!app.env) return null;
  const path = envFileOf(app);
  writeFileSync(path, renderEnvFile(app.env, { id: app.id }));
  return { file: path.slice(here.length + 1), values: app.env };
};

const monorepoSources = new Map();

for (const app of selectedApps) {
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

  // --- monorepo entries: materialized in place, never copied ---------------
  if (app.monorepo) {
    const { appDir: sub, packageManager, installAt } = app.monorepo;
    if (!sub || !packageManager || !installAt) {
      failures.push(`${app.id}: "monorepo" needs "appDir", "packageManager" and "installAt"`);
      continue;
    }
    if (app.firstParty || app.subdir) {
      failures.push(`${app.id}: a monorepo entry is built in its checkout — it declares neither "subdir" nor "firstParty"`);
      continue;
    }
    // Two monorepo apps cannot share one checkout: they would build into the
    // same tree and overwrite each other's output.
    if (monorepoSources.has(app.source)) {
      failures.push(`${app.id}: source "${app.source}" is already built in place by ${monorepoSources.get(app.source)}`);
      continue;
    }
    monorepoSources.set(app.source, app.id);

    const root = materializedRootOf(app);
    const appDir = appDirOf(app);
    const installDir = installDirOf(app);
    if (!existsSync(join(appDir, "package.json"))) {
      failures.push(`${app.id}: "${sub}" is not a package in ${root} (no package.json)`);
      continue;
    }

    // The license is a fact about the PINNED TREE, not about the project's
    // reputation — upstreams relicense. If the checkout stops matching what the
    // corpus claims, the fetch fails rather than writing a provenance file that
    // misdescribes what was cloned.
    const src = corpus.sources[app.source];
    const licensePath = join(root, src.licenseFile ?? "LICENSE");
    if (!existsSync(licensePath)) {
      failures.push(`${app.id}: no ${src.licenseFile ?? "LICENSE"} in ${root} to verify "${src.license}" against`);
      continue;
    }
    if (!licenseMatches(src.license, readFileSync(licensePath, "utf8"))) {
      failures.push(
        `${app.id}: ${src.licenseFile ?? "LICENSE"} at ${src.sha.slice(0, 12)} is not ${src.license} — ` +
          `re-read it and update corpus.json's "license"/"licenseNote"`
      );
      continue;
    }

    const env = writeDeclaredEnv(app);
    writeFileSync(
      join(root, "DIFFPACK_E2E_PROVENANCE.json"),
      `${JSON.stringify(
        {
          id: app.id,
          origin: "third-party",
          shape: "monorepo (built in place)",
          upstream: src.repo,
          sha: src.sha,
          appDir: sub,
          license: src.license,
          licenseNote: src.licenseNote ?? null,
          copyright: src.copyright,
          vendored: false,
          vendoringNote:
            "nothing from this repository is copied into diffpack: it is CLONED into the gitignored " +
            "integration/e2e/.cache/ and built there. diffpack redistributes none of it, which is why an " +
            "app under a copyleft license can be used as an oracle here at all — but the license of the " +
            "pinned tree is recorded because that is what governs anything anyone does with this checkout.",
          packageManager,
          installedAt: installDir.slice(here.length + 1),
          env: env?.values ?? null,
          envFile: env?.file ?? null,
          envNote: env
            ? "dummy values, declared in corpus.json and written by fetch.mjs; the build reads no ambient secret"
            : null,
          referenceBuild: app.relaxChecks
            ? {
                typeCheck: "disabled",
                lint: "disabled",
                why: app.relaxChecksNote ?? "declared in corpus.json",
                scope:
                  "next.config only, for the REFERENCE build; removed before diffpack sees the app, and nothing about the app's runtime behaviour is changed",
              }
            : { typeCheck: "as published", lint: "as published" },
          modifications:
            "none — a lockfile-managed workspace is installed exactly as published, so corpus.json's " +
            '"pins" are not applied to it',
        },
        null,
        2
      )}\n`
    );

    if (!noInstall && !existsSync(join(installDir, "node_modules"))) {
      const [cmd, ...pmArgs] = packageManager.split(/\s+/).filter(Boolean);
      console.log(`== install ${app.id} (${packageManager} install in ${installDir}) — this takes a while`);
      const r = spawnSync(cmd, [...pmArgs, "install"], {
        cwd: installDir,
        stdio: ["ignore", "pipe", "pipe"],
        encoding: "utf8",
      });
      if (r.status !== 0) {
        const log = join(root, "install-failure.log");
        writeFileSync(log, `${r.stdout ?? ""}\n${r.stderr ?? ""}`);
        failures.push(`${app.id}: ${packageManager} install failed (see ${log})`);
        continue;
      }
    }
    console.log(`== ready ${app.id} (monorepo, built in ${appDir})`);
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
    const declaredEnv = writeDeclaredEnv(app);
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
              env: declaredEnv?.values ?? null,
              envFile: declaredEnv?.file ?? null,
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
              env: declaredEnv?.values ?? null,
              envFile: declaredEnv?.file ?? null,
              modifications: pinned
                ? `${pinned} dependency version(s) pinned to corpus.json "pins"; application source unmodified`
                : "none",
            },
        null,
        2
      )}\n`
    );
  }

  if (!needsCopy) writeDeclaredEnv(app);

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
