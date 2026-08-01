# Removing obsolete compiler dumps from Git history

The compact oracle keeps targeted `.dump` files in the current tree. Older commits
contain broad generated snapshots, especially under `load`, `expand`, and their
expanded `corpus/` directories. Deleting those files normally does not remove their
blobs from clones; that requires rewriting every affected Git ref.

Do this only after the compact-oracle change is committed and coordinated with every
person and automation system using the repository. All commit IDs change, open work
must be rebased or recloned, and both configured remotes must be force-updated.

Use a disposable mirror clone, not a working checkout:

```sh
git clone --mirror git@github.com:jimmyhmiller/PlayGround.git Playground-oracle-rewrite.git
cd Playground-oracle-rewrite.git

# Preserve the compact snapshots from the pre-rewrite tip. The history filter below
# intentionally removes every generated dump, including the small current set.
snapshot_tip=$(git rev-parse refs/heads/master)
snapshot_dir=$(mktemp -d)
git archive "$snapshot_tip" coil/tests/compiler/oracle \
  | tar -x -C "$snapshot_dir"

git filter-repo --force \
  --path-glob 'coil/tests/compiler/oracle/reference/*.dump' \
  --path-glob 'coil/tests/compiler/oracle/*/reference/*.dump' \
  --path-glob 'coil/tests/compiler/oracle/*/full-reference/*.dump' \
  --path-glob 'coil/tests/compiler/oracle/*/corpus/*' \
  --invert-paths
```

Then make a normal checkout of the rewritten repository, restore only the compact
snapshots from `$snapshot_dir/coil/tests/compiler/oracle`, run
`python3 scripts/dev.py build full`, and commit them as a new tip commit. Compare
`git count-objects -vH` and a fresh clone size before publishing.

Finally, update every branch and tag deliberately. For the primary remote this is
normally `git push --force --all origin` followed by `git push --force --tags origin`;
repeat for the `computer` remote only after confirming it should carry the identical
rewritten graph. Keep a temporary backup ref or mirror until fresh-clone verification
passes, then have collaborators reclone rather than merge the unrelated old history.

The rewrite targets only generated compiler dumps and expanded corpus copies. It does
not remove diagnostic text, runtime stdout/stderr/exit references, fixtures, manifests,
or unrelated large objects elsewhere in the PlayGround monorepo.
