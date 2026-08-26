# mill

A tiny record-processing pipeline.

- Build an `Engine` from `settings.yml`; startup hooks named there
  (e.g. `reload_limits`) run against the limits at construction.
- Feed records through `Engine.process`, or merge two sources first
  with `combine` from `mill/merge.py` and use `Engine.process_pair`.
- Probes named under `probes:` in the settings gate each record.
