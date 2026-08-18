# PR #25 Codex review hotfix plan

This branch is reserved for the post-merge correctness fixes raised by the Codex review of PR #25.

Scope:

1. Preserve source-run identity when aggregating persisted outer predictions so two independent runs with the same seed can never be merged into one trade stream.
2. Treat `outer_predictions.parquet` as valid only when the adjacent walk-forward artifact explicitly declares it and the file matches the artifact's folds, per-fold seed, outer row range, and recorded sample count.
3. Make directional attribution respect actual dataset `row_index` spacing across market-data gaps instead of treating the compressed prediction array as contiguous candle time.

No model/feature/target tuning, no live-trading changes, and no sealed-test access.
