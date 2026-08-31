# CURRENT

## P13 — structural funding/basis carry feasibility: **NOT EVALUABLE**

**the governed acquisition attempt, and the evidence behind a NOT EVALUABLE determination**

Preregistration `sha256:1369c8828767c04e5b0609fc0125947c91f1cb5f15e977804ff1d1d70fd68767`
([`docs/p13_preregistration.md`](../../../docs/p13_preregistration.md),
[`nn/p13_preregistration.py`](../../../nn/p13_preregistration.py)), committed and
pushed at `939f38151cfa607e04c4d74846e081a8ab91ed49` before this attempt was made.

Generated at revision `2b1b400e4669557fe62e035497afc4d16e2f7431`, `dirty: false`.

### What happened

The frozen design requires **260 archive objects** — 65 months across four Binance
source families — spanning `2020-01-01T00:00:00+00:00` to
`2025-05-19T08:00:00+00:00` exclusive. All four families were probed. All four
were refused:

| source family | archive | result |
| --- | --- | --- |
| `spot_price` | `data/spot/monthly/klines/BTCUSDT/1h/` | 403 Forbidden |
| `perpetual_price` | `data/futures/um/monthly/klines/BTCUSDT/1h/` | 403 Forbidden |
| `mark_price` | `data/futures/um/monthly/markPriceKlines/BTCUSDT/1h/` | 403 Forbidden |
| `funding_settlement` | `data/futures/um/monthly/fundingRate/BTCUSDT/` | 403 Forbidden |

`data.binance.vision` is blocked by this execution environment's organisation
egress policy — the gateway answers `403` to `CONNECT`, which is a policy denial
rather than a transient network failure. `api.binance.com`, `fapi.binance.com`
and `www.binance.com` are refused identically.

The repository holds no substitute. Across the **entire git object database, all
branches and all history**, the only non-artifact parquet files ever committed
are four spot-derived research files; no funding rate, perpetual price, mark
price, index price or premium series has ever been committed. P4's
`*_derivatives_v1_*` artifacts contain model outputs — class probabilities,
actions, spot `future_return` — and no source data.

### The verdict

**`P13 ALWAYS-ON ANNUAL SPOT/PERP CARRY: NOT EVALUABLE`.**

Per the preregistration's stopping rule, NOT EVALUABLE is **terminal for this
design in this environment**, not an invitation to redesign it against whatever
data happens to be reachable. The design stays frozen and needs **no further design decision**: given egress to
`data.binance.vision`, `python -m tools.acquire_p13_sources --plan` names every
object required. That is a statement about the *design*, not about readiness —
substantial implementation remains before the screen can run. Absent today: the
downloader and its checksum verification, the loader and its truncating read, the
source manifests, the block runner, the G1–G6 gate, the S1–S4 stress runners, the
event ledger and the decision writer. What exists is the accounting core
(`nn/p13_carry.py`), its dimensional controls, and the networkless plan.

### What this is not

**This is not a negative economic result.** No P13 return, funding total, basis
figure, block result or gate decision was computed, and none is estimated from
anything that was reachable. The viability gate was never evaluated.

What was deliberately **not** done, each of which would have produced a number at
the cost of the checkpoint's meaning:

- no substitution of a different venue for Binance;
- no substitution of a REST endpoint for the preregistered historical archive;
- no synthetic, proxied or reconstructed perpetual or funding series;
- no relaxation of the frozen source set to fit reachable data;
- no P4-HOLD read, no Styx read.

### Files

| file | what it is |
| --- | --- |
| `acquisition_refusal.json` | the machine-readable refusal, with per-family probes, the plan digest and generation provenance |
| `acquisition_plan.json` | the networkless plan: all 260 objects, their URLs and their published `.CHECKSUM` companions |

The decision aggregate `artifacts/benchmark/btc_p13_decision/decision.json` does
**not** exist and must not: P13 remains `preregistered` rather than `answered` in
`nn.research_state`, because it has a committed design and no evidence. That is
exactly what the state means, and
`tests/test_p13_preregistration.py::test_no_p13_result_artifact_exists` holds it
there.
