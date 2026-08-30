# P6 — decision

**the preregistered gate, applied to frozen cells; the P6 outcome**

Preregistration `sha256:2785b1a7b19ecceca58cd0e936d14a5cbbbe6eb10f7ddf2796800409a0eaaaf2`, decided by `xgboost`.

| clock | horizon | positive folds | mean net return | beats momentum | verdict |
| --- | --- | --- | --- | --- | --- |
| `1m` | 6 minutes | 2 of 4 | 0.030087 | 4 of 4 | **not_viable** |
| `5m` | 30 minutes | 2 of 4 | 0.0114415 | 4 of 4 | **not_viable** |
| `15m` | 90 minutes | 2 of 4 | 0.00926375 | 4 of 4 | **not_viable** |
| `30m` | 3 hours | 2 of 4 | -0.0091055 | 4 of 4 | **not_viable** |
| `1h` | 6 hours | 2 of 4 | -0.0267705 | 4 of 4 | **not_viable** |

Viable clocks: none.

the five per-clock verdicts above. `outcome` says only whether any clock cleared the absolute gate; it is not a checkpoint-level score and there is deliberately no best-clock row.

If no clock passes, P6 is negative: changing the clock, on this asset, over these four periods, under this information set and these costs, did not produce robust cost-aware signal. P7 may still run, because consensus among unprofitable specialists is a separate question, and a negative P7 on top is a cleaner answer than not asking.

Evidence ceiling: Exploratory, adaptive. The four temporal periods are the same real-world windows v4, P2a, P2b, P2c, P3, P4 and P5 have already read, mapped onto faster clocks. Faster clocks multiply rows, not independent temporal periods: 2.8 million minutes of one asset over four windows is four observations of a market regime, not 2.8 million. No P6 result is confirmatory, and a positive P6 needs confirmation these blocks cannot supply.
