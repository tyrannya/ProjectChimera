# CURRENT

## P6-EXT — decision

**the preregistered gate, applied to frozen cells; the P6 outcome**

Preregistration `sha256:f0ce8bb4281389df5c877f20c88228350b2a20477ce36ad77da4acb7719c5804`, decided by `xgboost`.

| clock | horizon | positive folds | mean net return | beats momentum | verdict |
| --- | --- | --- | --- | --- | --- |
| `4h` | 1 day | 0 of 4 | -0.201455 | 2 of 4 | **not_viable** |
| `1d` | 6 days | 1 of 4 | -0.1920555 | 2 of 4 | **not_viable** |

Viable clocks: none.

the per-clock verdicts above, one for every clock the checkpoint registered. `outcome` says only whether any clock cleared the absolute gate; it is not a checkpoint-level score and there is deliberately no best-clock row.

If no clock passes, P6 is negative: changing the clock, on this asset, over these four periods, under this information set and these costs, did not produce robust cost-aware signal. P7 may still run, because consensus among unprofitable specialists is a separate question, and a negative P7 on top is a cleaner answer than not asking.

Evidence ceiling: Exploratory, adaptive, and designed with P6's results known. The same four real-world windows every checkpoint since v4 has read. No P6-EXT result is confirmatory.
