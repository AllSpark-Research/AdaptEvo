# Public Release Boundaries

## Included

- Audit agent, tool interfaces, multimodal rendering and structured parsing.
- GT-confidence and process-reward implementation and prompt templates.
- Relax training integration and framework source snapshots.
- Unit tests and synthetic examples.

## Excluded

- Real training/test data, human review records, post/user identifiers and media.
- Business rule caches, tool caches, learned experience and cached service results.
- Model weights, checkpoint shards, rollout traces, reward logs and result tables.
- Scheduler jobs, internal deployment recipes and infrastructure addresses.
- API keys, cookies, access tokens, private keys and credential stores.
- Original Git history, private export manifests and audit reports.

The release is a new source snapshot. Export sanitization uses an allowlist,
credential scanning and targeted code review; scanners do not prove that every
possible sensitive value has been detected. Review new contributions before
publication, especially fixtures, prompt examples and exception logs.

Service-specific credentials and media URL construction were replaced by
explicitly configured adapters. A hardcoded production rule supplement was
removed. Some optional legacy framework helpers retain placeholder paths and
require deployment-specific configuration. This is not a drop-in production
deployment and does not contain the data needed to reproduce private metrics.

Project-specific licensing remains a maintainer decision. Upstream licenses
remain attached to their respective components. No original repository history
or secrets are needed to use this snapshot.

## Verification of the initial snapshot

- Python source compilation completed successfully.
- Both included shell entry points passed `bash -n`.
- Isolated CPU suite: 88 tests passed, 6 subtests passed, 1 test skipped.
  The skipped test targets an excluded internal scheduler launcher.
- Gitleaks 8.30.1 reported no detected secrets in the release tree.
- Staged-file checks excluded datasets, media, logs, weights and key files.

No live model-service inference or distributed GPU training was performed as
part of this export. Those require separate deployment validation.
