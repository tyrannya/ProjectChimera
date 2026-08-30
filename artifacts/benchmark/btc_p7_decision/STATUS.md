# CURRENT

## P7 — decision

**the preregistered decision, applied to frozen mode evidence; the P7 outcome**

Preregistration `sha256:b79365ce0c6a8d1464b2420e589fd502cf3520daba775c2da5430497be12cb50`.

| mode | decision clock | improved folds | mean delta | verdict |
| --- | --- | --- | --- | --- |
| `SCALPING` | `1m` | 1 of 4 | -0.0265515 | **negative** |
| `DAY_TRADING` | `5m` | 1 of 4 | -0.034336 | **negative** |

Per-fold deltas against the fold-wise best constituent:

### `SCALPING` — 2 of 3, `15m` vetoes

| fold | consensus | best constituent | delta | trades |
| --- | --- | --- | --- | --- |
| 0 | -0.031842 | `5m` -0.012302 | **-0.01954** | 18 |
| 1 | 0.031669 | `1m` 0.080876 | **-0.049207** | 9 |
| 2 | -0.063037 | `1m` -0.016257 | **-0.04678** | 31 |
| 3 | 0.113459 | `1m` 0.104138 | **0.009321** | 11 |

### `DAY_TRADING` — 3 of 4, `1h` vetoes

| fold | consensus | best constituent | delta | trades |
| --- | --- | --- | --- | --- |
| 0 | 0.044898 | `5m` -0.007995 | **0.052893** | 6 |
| 1 | -0.001485 | `5m` 0.073974 | **-0.075459** | 1 |
| 2 | 0.005206 | `15m` 0.061887 | **-0.056681** | 6 |
| 3 | 0.0 | `5m` 0.058097 | **-0.058097** | 0 |

Supportive modes: none. Outcome: **neither supportive**.

The two modes are reported separately and neither is collapsed into the other. 'Scalping supportive, day trading not' is a result, not a reason to report scalping as P7's answer.

a mode whose consensus did not beat its own components is recorded as negative. The trading-mode scaffold may still describe it, and must not claim alpha for it.

Evidence ceiling: Exploratory, adaptive, and a rung lower than P6's. These are the same four real-world windows v4, P2a, P2b, P2c, P3, P4, P5 and P6 have read — the ninth reading — and P7 is designed with P6's results already known. No P7 result is confirmatory. A positive P7 would need confirmation these blocks cannot supply; a negative one needs no discounting.
