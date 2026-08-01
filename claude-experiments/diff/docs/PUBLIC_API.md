# Public API policy

Diffpack's supported external path is intentionally small:

- `diffpack-core`: provider contracts, environment records, diagnostics, graph
  updates, cancellation, and emission options.
- `diffpack-default-loader`: `BuildEngine` and `BuildEngineBuilder`.
- `diffpack-web`: resolved Web configuration and browser integration entry
  points.
- Framework crates: their public profile/build entry points.
- `diffpack`: CLI and profile selection only; it is not an SDK dependency.

Migration-era public modules remain available for built-in integrations but are
not yet semver-stable. New external examples must use crate-root facades. Public
surface snapshots will become a breaking-change gate before a stable release.
