# R0 — Owner adoption record

**Decision:** ADOPT AS CORRECTED
**Date:** 2026-09-14
**Phase:** R0 — Adoption and freeze, as defined in §37 of the corrected audit (= Part C of the corrigendum)
**Base `main` at adoption:** `46921ef1206748c6b7304432a26c8295b7830e27` (ordinary merge of PR #95; verified by `git rev-parse origin/main` at the start of the R0 session, and equal to the SHA both records audited)
**Governance branch:** `governance/r0-adopt-audit-roadmap-20260914`, created from that exact commit
**Governance PR / R0 commits:** see the last section; filled in by a follow-up ordinary commit once the PR exists

This record is a governance document. It creates **no scientific evidence, no prospective boundary, no alpha claim and no real-money authority**. It changes no runtime behaviour, no recorder behaviour, no scientific artifact, no preregistration and no frozen result.

---

## 1. What is adopted

The owner adopts the **complete corrected §37 master roadmap, R0–R18, as written in the final owner corrigendum's Part C** (which replaces §37 of the audit in full and is byte-identical to §37 of the corrected audit apart from Part C's two-line preamble), together with the corrigendum's **Part B dependency graph** as the definition of execution order. Every section — §37.0, §37.1, §37.2 and phases R0 through R18 including the real-money authority boundary — is adopted; none is declined or adopted in part.

The authoritative working copy is [`docs/master_roadmap_r0_r18.md`](../../master_roadmap_r0_r18.md). It reproduces Part B and Part C verbatim, with one disclosed typographic normalisation in the §37.0 phase map (a run of ASCII `=` drawn as `═`, because the repository's merge-conflict-marker check refuses lines that start with seven `=`). The two records in this directory keep the original bytes.

[`docs/proposed_futures_first_roadmap_v3.md`](../../proposed_futures_first_roadmap_v3.md) (Roadmap v3, adopted 2026-09-12) is **superseded by R0** for all active sequencing. It is preserved intact as historical provenance and is not rewritten.

Semantics the owner explicitly confirms are part of what is adopted:

- **R9 PASS before R6.** The prospective boundary activates only after the autonomous soak has passed on the exact runtime head that will run the campaign. SOAK PRECEDES PROSPECTIVE ACTIVATION.
- **Deciding cost semantics frozen before R10 and identical through R11.** R10 and R11 test the same object; confirmation adds data and never redefines success.
- **Mechanical, non-adaptive invalidation.** Block validity is decided by frozen code from engineering records only, before any readout, with a frozen target of VALID blocks, a frozen extension cap and a NOT EVALUABLE outcome.
- **Multi-timeframe capability is a required engineering property; hierarchical MTC as a scientific model is not mandatory for campaign 1.**
- **The primary futures directional lane has priority** over the optional carry lane.
- **Aegis remains the sole central risk authority.**
- **Real-money authority remains a separate four-condition boundary** (scientific = Gate 1, engineering = R9 PASS + R14 re-soak + Gate 2, operational = Gate 3, owner authorisation = Gate 4), never crossed by a backtest, demo, soak, canary or single positive checkpoint.

## 2. Execution order (load-bearing path)

Phase IDs are stable labels; execution order is the Part B graph, not the ID sequence.

```text
R0
  ├─ R1
  └─ R2

R2 -> R3
R3 -> R4
R4 -> R5
R3 + R4 -> R7
R1 + R7 -> R8
R8 -> R9

R3
+ R4
+ R5
+ R9 PASS
+ completed qualifying recorder period
+ independent boundary review
    ->
R6

R6 -> R10 -> R11
```

**The next permitted phases after the R0 merge are R1 (runtime integrity remediation) and R2 (gen4 engineering preflight on a separate host), both from R0, in parallel.** Nothing else starts. R0 itself starts neither.

## 3. Owner dispositions recorded by R0 (per §37 R0 EXACT WORK (a)–(f))

| item | recorded value |
| --- | --- |
| (a) adoption | ADOPT AS CORRECTED, all sections, naming the audit and corrigendum hashes in section 5 |
| (b) PR #76 disposition | **Not merged by R0.** PR #76 ("PR-06 — recorder archive reconciliation and 30-day coverage gate") stays open and draft at head `8a8f4a1f7d754cff190a3668873072a7efbb1542` on base `95322a991e762ca40a406bb69093d148bcd9d22a`; it remains governed by the **R5 branch rule** (PASS / FAIL branches of its second acceptance campaign). Its acceptance VPS and its state are untouched by R0. |
| (c) gen3 | stays **engineering-only**; `prospective_from` stays **`null`**; nothing recorded under gen3 becomes prospective evidence |
| (d) budget | at most **2 directional candidates** and at most **1 carry candidate**, counted separately, before a mandatory stop-and-decide review; per candidate one first campaign (R10), at most one automatic same-design re-run after a NOT EVALUABLE outcome, and one continuation (R11); ABORTED and DECLINED consume the slot; **zero deciding use of the burned historical blocks**; R1–R9 engineering effort cap: the corrigendum's default (≤ 10 person-weeks) applies, not overridden |
| (e) Lane B election | **DEFER TO R4.** The optional carry lane is secondary; it may be elected at R4 only if it does not materially delay or constrain the primary roadmap; if still undecided at R4 it is not elected, and a later carry campaign needs its own contract and boundary and never delays R6 |
| (f) resource rule | **the primary directional futures lane wins every resource conflict** |

**Primary lane, stated once:** multi-symbol, single-leg, directional LONG / SHORT Binance USD-M perpetual futures at 1× leverage. Spot is reference data, a hedge leg where a lane needs one, and support infrastructure.

**Defaults from corrigendum Part E that apply unless a later owner decision overrides them:** the extension cap E = ⌈N/2⌉ blocks; the continuation size N′ ≥ ⌈N/2⌉ and ≥ 3 blocks (R3's power report fixes the numbers inside the frozen rule); the alpha-spending interim-look rule at block ends.

## 4. Standing that R0 leaves unchanged

- `P4-HOLD` remains retired and unread.
- Styx remains sealed and unread, with its disclosed hindsight-era ceiling.
- P8 remains withdrawn as moot and unopened; it has no result.
- P14 remains declined before opening; it has no result.
- P13 remains closed on source validity with its economic screen never run.
- The four outer historical blocks remain BURNED for deciding purposes.
- Every closed-checkpoint scientific conclusion (v4, P2a, P2b, P2c, P3, P4, P5, P6, P6-EXT, P7) stands exactly as recorded; R0 is a governance sequencing change, not a reinterpretation of old evidence.
- No prospective data exists; no campaign is preregistered; no boundary is active.
- Aegis is the sole risk authority; the futures path is dry-run only; no live route is authorised.

## 5. Provenance of the immutable records

Both records were supplied by the owner and copied into this directory **byte for byte** (`cp`, then `cmp` against the source, then `sha256sum -c`). No line-ending conversion occurred (`.gitattributes` declares `* -text`), so the source-file hash and the committed-file hash are the same value.

| record | source SHA-256 (upload) | committed SHA-256 |
| --- | --- | --- |
| `ProjectChimera_full_audit_and_master_roadmap_2026-09-14_corrected.md` (257,427 bytes, 1,444 lines) | `3bddc12e577f5e6e385860ef8484fc0ac023976e4bbc515fd0b8e8cff501e62f` | `3bddc12e577f5e6e385860ef8484fc0ac023976e4bbc515fd0b8e8cff501e62f` |
| `ProjectChimera_final_owner_corrigendum_and_adoption_ready_roadmap_2026-09-14.md` (148,144 bytes, 1,168 lines) | `baaf2d0b7cfc4082aacc61a9ce7fa08d76036af044b50d2b2adcefd2b3d18797` | `baaf2d0b7cfc4082aacc61a9ce7fa08d76036af044b50d2b2adcefd2b3d18797` |

The digests are also listed in [`SHA256SUMS.txt`](SHA256SUMS.txt) (`sha256sum -c` format) and pinned by `tests/test_r0_governance_records.py`.

Keeping the bytes required two exact-path exemptions, because both records draw the §37.0 phase map with an intentional `================` separator row and the corrigendum carries three trailing spaces inside a quoted table (Part D, D9): `tests/test_config_and_cli.py::test_no_merge_conflict_markers_remain` skips exactly these two paths, and `.pre-commit-config.yaml` excludes exactly the corrigendum from `trailing-whitespace`. No directory-wide exemption exists; the derived master roadmap is normalised to repository style instead.

Consistency check performed before adoption: audit §37 (corrected) and corrigendum Part C are textually identical apart from Part C's preamble ("This section replaces §37 of the audit in full."); the Part D replacements are present in the corrected audit where spot-checked (§1 findings 7 and 8, §35, §36, §38, §39). No material disagreement between the two records was found.

## 6. What R0 does not do

- It does not start R1 or R2, or any later phase.
- It does not modify runtime, recorder, Aegis, executor, ledger or replay code.
- It does not touch PR #76, its branch, its acceptance VPS or its recorded acceptance days.
- It does not activate, move or create any prospective boundary; it creates no scientific result and reads no sealed or burned data.
- It does not create real-money authority; no leverage above 1× is authorised at any stage of the adopted roadmap.
- It does not rewrite the audit, the corrigendum, Roadmap v3 or any closed-checkpoint narrative; supersession is recorded by banners and pointers.

## 7. Governance PR and commits

| item | value |
| --- | --- |
| R0 content commit | recorded by the follow-up commit |
| governance PR | recorded by the follow-up commit |
| merge | an ordinary merge commit into `main`, only after exact-head CI is green and a final adversarial diff review confirms the frozen R0 scope |
