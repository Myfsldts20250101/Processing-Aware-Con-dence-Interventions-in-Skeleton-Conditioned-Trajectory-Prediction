# Diagnostic release: local packaging version 2026-10-16

Paper: Processing-Aware Confidence Interventions in Skeleton-Conditioned Trajectory Prediction.

## Assemble the release

Extract code-v1.0.zip, checkpoints.zip, processed-jrdb_PENDING_PERMISSION.zip,
diagnostic-results.zip and figures.zip into the same parent directory. Each
archive uses the common root mva_confidence_diagnostic/. When cloning the
GitHub source instead, extract only the asset archives into the clone's parent.
The GitHub ZIP contains that root directory; upload its contents, not the ZIP,
to the repository. Editable figure sources occur in both code and figures
archives and are byte-identical.

The source code package alone needs the separately distributed assets for
inference and the compressed tables for statistical recomputation. All paths
are documented in README.md, diagnostics/README.md and figures/README.md.

## Verification status

Eight B checkpoints match their recorded SHA-256 values. Archived U1F checks
passed after path adaptation. The small inference entry returned BLOCKED
because torch and PyYAML are absent in the checked local environment; model
loading and forward inference were not executed. No full matrix rerun or
fresh figure rendering is claimed. New runtime outputs default to local_runs/.

Historical U1F replay additionally requires original E2/U1 reference CSVs
not distributed here. The HST materials are posthoc statistics only. The
80,000 historical F sample rows are unavailable; ten-bin summaries are included.

## Distribution

This is a local packaging version, not a published GitHub release or registered
DOI. No repository commit or DOI has been invented. Before publication, record
the actual GitHub release tag and commit, register the new diagnostic archive,
and update cross-links. The previous fusion@20 DOI does not identify this release.

processed-jrdb_PENDING_PERMISSION.zip contains JRDB-derived arrays. Confirm
the applicable redistribution terms before uploading that archive publicly.
Code, arrays and checkpoints require separate license treatment. The supplied
LICENSE is a short Apache-2.0 notice; complete the license text before publishing
the code. Preserve third-party notices where applicable.

## Integrity

ARCHIVE_CONTENTS_*.sha256 lists paths and hashes inside each ZIP. The separate
PAYLOAD_SHA256SUMS.txt covers the distributed source files. The original source
package manifest is retained as provenance/SOURCE_PACKAGE_SHA256SUMS_20261016.txt;
it describes the pre-packaging local tree, including excluded runtime files,
and is not the release verification list. No environments, caches, local run
outputs or credentials are distributed.
