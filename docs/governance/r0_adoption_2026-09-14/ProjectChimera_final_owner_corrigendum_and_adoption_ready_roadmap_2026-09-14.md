# ProjectChimera — Final Owner Corrigendum and Adoption-Ready Master Roadmap

**Date:** 2026-09-14.
**Basis:** the completed 39-section adversarial audit dated 2026-09-13 (`main` = `46921ef1206748c6b7304432a26c8295b7830e27`, verified then; not re-scouted now).
**Standing:** a PROPOSAL from an independent read-only review. It creates no scientific evidence, no prospective boundary, no alpha claim and no real-money authority. Nothing in the repository, on GitHub, on any VPS or in any recorder/acceptance environment was touched.
**Scope:** consistency corrections and owner-level clarifications to the completed audit, plus the adoption-ready rewrite of §37. Every audit finding stands unchanged unless a correction below required a wording change; those places are listed exhaustively in Part D. Evidence labels as in the audit: [RF] repository fact, [EF] external fact, [INF] inference, [REC] recommendation.
**Not done here, by instruction:** no repository scouting, no new workflow over the code, no technology re-survey, no reopening of scientific history, no modification of anything.

---

## Part A — Corrigendum

| # | ISSUE | ORIGINAL PROBLEM | FINAL DECISION | AFFECTED SECTIONS | WHY |
|---|---|---|---|---|---|
| A1 | R6 / R7 / R8 / R9 ordering | §37.0 listed R6 (boundary activation) before R7 (snapshot), R8 (runtime) and R9 (soak); R6's INPUT did not name R9; R9's text allowed the soak to read a "prospective root". §38, §14 (coherence rule) and §39 said the opposite: v3 was defective precisely because it activated the irreversible boundary before the runtime existed and had soaked. | R6 activates only after R3 + R4 + R5 + **R9 PASS** + ≥ 30 qualifying production-recorder days + an independent boundary review that has seen the frozen soak report. Phase IDs are kept as stable labels; execution order is defined by the dependency graph (Part B), not by ID. The 30-day qualification may accrue in parallel with R7–R9; activation may not. **SOAK PRECEDES PROSPECTIVE ACTIVATION.** | §37.0, §37 R6, R7, R8, R9, R10; §26; §35; §36; §38; §39 | the boundary is irreversible; every phase that can still be repaired must finish before it |
| A2 | §35 items 5–6 vs the R4 → R7 edge | §35 said "in parallel and without waiting for R2/R3: R7 and R8, then R9", but R7's acceptance needs the contract's declared streams/clocks (R4) and the frozen feature function (R3). | Dependency edges are **completion/acceptance** dependencies. Contract-agnostic implementation work in R7/R8 may start early (marked ENG-START-EARLY) but R7 cannot be ACCEPTED before R3 and R4, and R8 cannot be accepted before R7. | §35; §37.0 parallelism paragraph; R7, R8 | a snapshot cannot be validated against a contract that does not exist |
| A3 | §37.0 parallelism paragraph | "R7/R8 ∥ R4/R5" contradicted the R4 → R7 edge; "R10 needs R6 + R9" was correct but did not say R6 needs R9. | Paragraph replaced by the explicit dependency list of Part B. | §37.0 | same as A1/A2 |
| A4 | Soak economics as an activation input (new) | With the soak now before the boundary, the frozen candidate runs in shadow on pre-boundary data during R9. If the owner could decline activation after reading its shadow PnL, activation would be optional stopping on a pre-boundary outcome. | The R9 PASS/KILL gates are engineering-only. The candidate's shadow economics during the soak are logged (Aegis needs them) but sealed: hashed at soak end, not reported, not an input to the R6 decision, opened only after R10 closes as a DIAGNOSTIC shadow-vs-live comparison. Declining activation after a PASSED soak is permitted only as an explicit owner decision recorded with a reason; the candidate is then DECLINED: its engineering soak report stays ENGINEERING evidence, its sealed shadow economics are never opened (not even for descriptive context), the decline consumes the lane's candidate slot, and re-election requires a new R3, a fresh soak and a fresh seal. | §26 "Separation"; §37 R9, R6; §37.1 | removes the only new adaptive path the reordering creates |
| A5 | Runtime head between R9 and R10 (new) | Nothing bound the campaign's runtime head to the soaked head. | **Head-freeze rule:** R10 runs the exact head that passed R9. Changes are allowed only as ENGINEERING-FIX PRs: reviewed, touching no contract-hashed module (feature function, model, policy, cost model, evaluator, validity evaluator), with replay parity re-proven on ≥ 3 prior days, recorded as an engineering event in the campaign log. Anything else invalidates the block in which it lands (A7) and requires re-soak before the next block. | §37 R9, R10, §37.2 | a soak certifies a head, not a branch |
| A6 | Cost semantics changed at R11 | R11 read "with full cost fidelity (measured slippage distributions, funding by settlement, reject/partial models from a testnet adapter if available)"; §23's fidelity table added impact, reject/partial rates and mid-referenced slippage at the R11 stage; §27's cost bullet said "frozen from preflight" without saying that R11 uses the same model. Confirmatory evidence cannot use a success definition altered after R10 was observed. | All DECISION-RELEVANT economic semantics (fee tier and BNB assumption, spread rule, executable-price rule, slippage envelope, funding treatment, turnover accounting, reject/partial-fill treatment, latency assumption, impact rule and the notional cap that keeps impact out of scope) are frozen in R3 as hashed code. R10 and R11 test the same object. R11 may use newly observed execution/testnet/live-operational information only (A) via a deterministic update rule preregistered before R10, (B) as a non-deciding stress/diagnostic analysis, or (C) by opening a new preregistration/campaign. Confirmation adds data; it never redefines success. | §37 R3, R10, R11, R14; §23 fidelity table and a new paragraph; §27 cost bullet and R11 outcome bullet; §33 row 15 | the confirmatory period must be a replication, not a re-scoring |
| A7 | Engineering invalidation discretionary | "the block is excluded and the campaign extends by one block, never substituted" left who decides, on what, and for how long unspecified; it could become selective block substitution. | Frozen in the R3 contract before activation: (i) the mechanical invalidity criteria; (ii) the evaluator: frozen code `nn/prospective/validity.py` reading engineering records only (coverage manifests, recorder health, runner liveness/halt/recovery records, parity results, identity/head records) and **no prices, scores, positions or PnL**; (iii) blind ordering: the validity verdict for a block is computed, committed and hashed **before** that block's scientific readout runs; (iv) the target number N of VALID scored blocks (from the power report); (v) the deterministic calendar-extension rule: accrual continues in contiguous calendar blocks until N valid blocks exist; (vi) the maximum total extension E (proposal: E = ⌈N/2⌉ blocks); (vii) exhaustion → **CAMPAIGN = NOT EVALUABLE / ENGINEERING FAILURE**, neither PASS nor scientific FAIL; the candidate is not retired; the successor rule is pre-committed in the contract: at most one automatic same-design successor (new hash, new boundary, re-soak first, no pooling), taken once the engineering cause is fixed and never decided on outcomes; the predecessor's VALID-block readouts stay SEALED-DIAGNOSTIC until the successor closes or the candidate is retired; a second exhaustion is terminal for the design and consumes its slot. An owner stop, or a change to a contract-hashed module during accrual, is ABORTED — never NOT EVALUABLE: it consumes the slot, permits no same-design rerun and leaves all readouts sealed. INVALID blocks stay in chronology, labelled, reported, never a negative result, never replaced. | §37 R3, R10, R12, §37.1 (new status labels), §37.2; §27 invalidation bullet; §13 fitness paragraph; §28 rule 1 | prevents adaptive substitution while keeping engineering failures honest |
| A8 | MTF capability vs "thin" snapshot | §16, §25, §33 row 7, §37.0 ("MarketContext-thin") and R7 described a "thin" snapshot, which could be read as a single-timeframe runtime. The scientific conclusion (mandatory hierarchical MTC is premature) was correct; the engineering restriction was not intended. | **MULTI-TIMEFRAME CAPABILITY = REQUIRED ENGINEERING PROPERTY. HIERARCHICAL MTC AS A SCIENTIFIC MODEL = NOT REQUIRED FOR CAMPAIGN 1.** `MarketSnapshot` is generic over contract-declared causal clocks; every input/timeframe carries `source_time`, `available_time`, `as_of`, `age`, `staleness`, `complete`, `missing`; multiple closed-bar clocks are derived causally from the base 1 m stream. Campaign 1's deciding feature set = the preregistered decision clock + at most two preregistered slow closed-bar variables; every other available clock is capability only and creates no degree of freedom during R10. No "HTF always wins", no voting, no coherence rules, no mandatory narrative state. A richer hierarchy returns only under a new preregistered campaign. | §1 finding 7; §16 decision; §25 row "Causal feature state"; §33 row 7; §37.0; §37 R7, R3; §38 MTC row | keeps the runtime future-proof without granting the first campaign hidden degrees of freedom |
| A9 | Futures-first priority and the carry lane | §1 finding 8, §14 items 3–4, §27 Lane B and §33 row 21 recommended the carry lane as "parallel" and noted its economic evidence "may arrive sooner", which could be read as priority. | The PRIMARY lane is multi-symbol, single-leg, directional LONG/SHORT on Binance USD-M perpetuals. Spot = reference data, hedge leg where necessary, support infrastructure. The carry lane is a **SECONDARY, OPTIONAL** parallel experiment: it must not delay R1–R9, must not consume the directional lane's hypothesis budget (it has its own single slot), is not a prerequisite for anything, is not orchestration (own process, own state root, own Aegis instance, own ledger, no capital netting before R13), does not replace futures-first, and gets no priority for arriving sooner. **If resources conflict, the primary directional futures lane wins.** Election is an owner decision at R0 with a deadline at R4 (A16). | §1 finding 8; §14 items 3–4; §27 Lane B; §33 row 21; §37 R0, R3, R4, R9, R10; §38 strategy-family row | removes ambiguity without dropping the only cheap economic experiment |
| A10 | Candidate budget inconsistency | R0 said "at most TWO candidates across the programme"; §28 rule 3 and §38 said two directional + one carry. | Budget = at most two directional candidates and at most one carry candidate, counted separately; per candidate one R10, at most one automatic same-design re-run after a NOT EVALUABLE outcome, and one R11; ABORTED and DECLINED consume the slot; a mandatory stop-and-decide after two directional negatives. | §37 R0, R12; §28 (unchanged) | one number, stated once |
| A11 | R15 gate wording | §37.0 read "R15 Owner authorisation (Gate 1+2+3+4 all PASS + signed contract)" although Gate 4 *is* the signed capital-governance contract (§30). | R15 = Gate 4 executed (the signed live-authorisation contract) after Gates 1–3 PASS; the four real-money conditions map to Gates 1, 2, 3 and 4 respectively. | §37.0, R15 | gate numbering must match §30 |
| A12 | Per-phase field completeness | R9 lacked NEXT MAY ASSUME and several fields; R6 INPUT lacked R9; R11–R18 were written as summaries without the full field set. | Every phase R0–R18 now carries all eighteen fields. | §37 (all phases) | adoption-ready means no field left to inference |
| A13 | Preservation of strong findings | Risk that a cleaner roadmap softens confirmed remediation. | Verified: each item in the owner's list maps to a named work item (R1-a … R1-o, R8-a … R8-g, R2, R3, R10 no-peek). None was downgraded to a recommendation; the R1 list was expanded, not shortened. | §37 R1, R8 | the findings are the reason the roadmap exists |
| A14 | §36 first bullet incomplete | "Do not activate any prospective boundary before R1, R3 and R9 are complete" omitted R4, R5, the qualifying period and the review. | Bullet replaced with the full R6 input list. | §36 | must match R6 |
| A15 | §38 / §39 sequencing wording | §38's sequence row already placed R6 after R9 but did not say R6 *requires* R9 PASS; §39's boundary bullet listed "no soak" among blockers without naming R9 PASS as a gate. | Both restated explicitly. | §38, §39 | no residual ambiguity |
| A16 | gen4 streams for the carry lane (new) | R2/R4's Tier A listed perpetual streams only; the carry lane needs spot kline_1m and spot bookTicker for its symbols. Nothing tied the contract's content to the lane election. | If Lane B is elected at R0 (deadline R4), R4's contract adds the spot streams for BTCUSDT (and ETHUSDT if named) as HEALTH_GATED capture; if not elected by R4, a later carry campaign needs its own contract and boundary and never delays R6. | §37 R0, R2, R4; §27 Lane B | a stream absent from the contract cannot serve the campaign |
| A17 | Validity verdict vs interim looks (new) | The alpha-spending interim look at block ends and the validity evaluation of the same block were not ordered. | Validity first, committed and hashed; the block's readout (interim or final) is computed only for blocks already marked VALID, by the frozen evaluator; an INVALID block's readout is never computed. | §37 R10; §27 no-peek and invalidation bullets | the readout must never inform validity |
| A18 | §24 build-vs-adopt row still revised the fill model at R11 (verifier-found) | The "Execution simulation" row read "REVISE for R11 with measured slippage distributions and a size-dependent impact term", the pre-correction R11 cost change, and Part D had listed §24 as unchanged. | Row rewritten: the deciding fill model is frozen in R3 and identical in R10/R11; measured slippage distributions and impact enter only as non-deciding stress analyses or via routes (A)/(C). | §24 (D36) | mandate 2 |
| A19 | Consequence of a non-ENGINEERING-FIX change stated three ways (verifier-found) | §37.2/R10 said INVALID block + re-soak for any change; §36 said new campaign; R11 said route (C). | Single-valued: a change to a non-contract-hashed module → INVALID block + re-soak (on an engineering root or, read-only and unscored, on the prospective root); a change to a contract-hashed module → the campaign ends ABORTED (the object changed) and any successor is route (C). | §37.2; R10 KILL; §36 (D29) | mandate 2 |
| A20 | Owner stop mapped onto NOT EVALUABLE (verifier-found) | R10 KILL recorded an owner stop as NOT EVALUABLE, a label 37.1 reserves for extension exhaustion, which would have opened a discretionary abort-and-rerun path. | New immutable status ABORTED (owner stop, or a hashed-module change during accrual): consumes the candidate's slot, no same-design rerun, all readouts stay sealed, never NOT EVALUABLE. | §37.1; R10; R12; §28 (D19) | mandate 3 |
| A21 | NOT EVALUABLE successor was outcome-informed and unbounded (verifier-found) | R12 disclosed the partial VALID-block readouts as DIAGNOSTIC before the successor decision and allowed same-design reruns without limit; the original per-candidate campaign cap had been dropped. | Successor rule pre-committed in R3: at most one automatic same-design successor, never decided on outcomes; partial readouts SEALED-DIAGNOSTIC until the successor closes or the candidate is retired, then ADAPTIVE context only; a second exhaustion is terminal; per-candidate cap restored (one R10, ≤ 1 re-run, one R11). | R0 (d); R3 item 4; R10; R12; §28 (D19); A7, A10 | mandate 3 |
| A22 | §36 bullet contradicted route (A); the route-(A) example loosened the model (verifier-found) | The new §36 bullet forbade any change between R10 and R11, while R3/R11 permit a preregistered deterministic update; the example replaced a p90 envelope with a median. | Bullet allows route (A) explicitly; a route-(A) rule may never make the deciding economics less conservative than R10's; example changed to max(frozen p90, testnet p90). | §36 (D29); R3 item 4 | mandate 2 |
| A23 | Lane B coupled to the primary soak and to an undefined "ready" (verifier-found) | Lane B soaked inside the primary R9 with shared PASS/KILL gates, so a Lane B drill failure could fail R9 and delay R6; R6 activated Lane B "if elected and ready"; the shadow-economics seal covered only Lane A; soak-time daily reports were not declared PnL-free. | Lane B soaks in its own process under its own PASS/KILL record and never enters the primary R9 verdict; readiness at R6 is mechanical (own preregistration merged ≥ 7 days earlier; own R9 record passed) and may not reference sealed economics; a Lane B elected after R3 gets its own preregistration PR; the seal and DECLINED consequence apply to both lanes; daily reports carry engineering metrics only and economic fields live in a hashed sub-root read only by the runtime and Aegis. | R3 item 5; R6; R9; Part B side lane | mandate 5 |
| A24 | Qualification clock defined three ways (verifier-found) | R4 started the 30-day clock at deployment; §37.0 said the qualification needs R5; R6 (5) required reconciliation on every qualifying day, which only R5 provides. | One rule: the clock starts at R4's production deployment; a day counts only once R5's accepted reconciliation has been run over it (offline, retroactive where needed); the record closes only after R5 acceptance; any failure restarts the clock; R5 owns the qualification-record PR; R5 → qualification drawn in Part B. | §37.0; R4; R5; R6 (5)–(6); Part B | mandate 1 |
| A25 | "R9 PASS on the exact head to be deployed" was unsatisfiable; key-creation timing disagreed (verifier-found) | The deployable head contains the R14 adapter, which post-dates R9; R14 allowed an API key after Gate 1 while §30 Gate 4/R15 forbid any key before the signed contract; R14 also allowed adapter code "earlier" without bound. | R14 re-runs the R9 gates on the exact deployable head (adapter in testnet mode) and that report is what real-money condition (2) names; only testnet credentials or a read-only key exist before R15; the trade-capable key is created in R15; the adapter is built in R14 after Gate 1. | R14; real-money boundary (2); R15; §30 Gates 2 and 4 (D41, D42) | mandates 1, 6, 7 |
| A26 | R6 inputs without a producing phase or PR (verifier-found) | The qualification record, the source-identity verification and the boundary review had no owning phase, and R6's only PR "contains nothing else". | R5 produces the qualification record (with source identity) as ENGINEERING evidence in its own PR; the boundary review is recorded in its own governance PR before the activation PR. | R5; R6 PR BOUNDARIES; Part B | mandate 1 |
| A27 | R13 Aegis wording drifted from §7/§29 (verifier-found) | R13 spoke of per-process engines plus a portfolio "check", dropping "one ledger"; §7 and §29 say one engine over the lanes that share an account. | Original wording restored: one Aegis engine over the lanes that share an account, one account ledger, one evidence trail; separate dry-run lanes on separate accounts keep separate engines. | R13 | mandate 7 (no second authority) |
| A28 | Edge R1 → R7 appeared only in R7's DEPENDENCIES (verifier-found) | Neither 37.0 nor Part B carried it. | Removed from R7; R1 feeds R8, as the owner's graph states; R7's acceptance runs the snapshot builder standalone over the recorder root. | R7 | mandate 1 |
| A29 | Findings without a named work item (verifier-found) | AEG-6 (continuous position guards / maintenance rate single source, resolve-by R8), the live-vs-replay funding-booking minute (§11, §34 risk 17), the ≡ 0 feed-age source (TIME-1), AEG-1's reproducing test cases, disk-full fail-silence (§34 risk 13), the §12 per-minute provenance flag, and the coherence rule had no concrete item or test. | Added R8-h (guards inside Aegis from `MarginState`; maintenance rate from `SymbolConstraints` only); R1-g funding-settlement-minute rule; R1-f rewritten with the operational-clock age source and R1-e's injected operational clock; R1-b's three reproducing tests; disk-full behaviour and crash-harness fault in R1-i; the provenance flag in R4; a two-sided coherence control in R3 tests, R8 acceptance and R10 tests. | R1-b, R1-e, R1-f, R1-g, R1-i; R3; R4; R8-h; R10 | mandate 6 |
| A30 | DECLINED had no consumer, no budget effect and a leaky BURNED label (verifier-found) | BURNED material may be reused "for descriptive context", so a declined candidate's sealed economics could be read; no phase consumed the DECLINED record. | DECLINED: soak report stays ENGINEERING; sealed shadow economics never opened; consumes the lane's slot; re-election only via a new R3, fresh soak, fresh seal; R6 → DECLINED → R12 drawn; R12 accepts the record. | §37.1; R6 KILL; R12; Part B | mandate 3 |
| A31 | Graph errors (verifier-found) | An unconditional R12 → R14 arrow let NEGATIVE/NOT EVALUABLE outcomes reach R14; re-entry edges (new candidate, automatic successor) were undrawn; R15 was marked irreversible although revocable; the R3 merge was unmarked; R9's recorder source had no edge; R5's new-contract-version path re-entered R4 silently; §38's sequence row read as if R3 waited for R1. | R14 hangs off R11 Gate 1 only; R12 is the governance sink with marked re-entry edges; R15 = revocable; R3 merge marked; production recorder → R9 edge; R5 → R4 re-freeze edge with clock restart; §38 row rewritten. | Part B; §37.0; R5; R15 row; §38 (D30) | mandate 1 |
| A32 | R3's data fields contradicted its own λ selection (verifier-found) | DATA USED said "null calibration and cost measurement only" while item 1 selects λ and fits coefficients on pre-boundary engineering data. | Fields state the one-time predeclared λ-selection CV and coefficient fit (hashed, EXPLORATORY-labelled, disclosed, never a go/no-go input). | R3 | honesty of the disclosure block |
| A33 | Validity evaluator: "who" unspecified (verifier-found) | The evaluator, records and ordering were fixed but not the actor. | Run by the CI/verifier job, never by hand; output, input manifest and hash committed by that job in the block's evidence PR before the readout job runs; reviewed by a person certifying no readout seen; a manual verdict is itself an invalidity criterion. | §37.2; R10 | mandate 3 |
| A34 | §35 item 2 and §8/§9/§11 cells carried pre-correction text (verifier-found) | §35's R1 enumeration omitted AEG-1, `risk.json`, clock hardening and evidence labels; EXE-4/ACC-8 assigned the cost-model unification to the "gen4 freeze"; REC-2 said a crash-ended campaign is "counted". | §35 item 2 lists R1-a … R1-o; EXE-4/ACC-8 resolve in R3; REC-2 resolves by the frozen validity evaluator. | §35 (D37); §8 (D38); §9 (D39); §11 (D40) | consistency |
| A35 | R11's inputs not produced by R3 (verifier-found) | R11 took N′ and a non-inferiority rule "from the R3 contract" that R3's EXACT WORK did not list. | Both added to R3 items 3 and 4. | R3 | A12 (no field left to inference) |

Rows A18–A35 were raised by five independent read-only document verifiers (same serving model; distinct lenses: mandate coverage, residual contradictions, field completeness and non-softening, over-correction, graph integrity) run against the first draft of this corrigendum; every substantive finding was accepted and applied. A second verifier pass over the revised text was launched and failed on the account's usage limit before producing any result; the revised passages were re-read by the lead only, and that is disclosed here rather than claimed as independent verification. No verifier read the repository.

---

## Part B — Final Phase Dependency Graph

Edges are **acceptance dependencies**: a phase is ACCEPTED only when every predecessor is accepted. Implementation work may begin earlier only where a phase says ENG-START-EARLY. Phase IDs are stable labels; **execution order is this graph, not the ID sequence**. Double rules mark irreversible transitions.

```
CURRENT MAIN  46921ef1206748c6b7304432a26c8295b7830e27
      │
      ▼
R0  Adoption & freeze (owner decision; Lane B election opens; budget fixed)
      │
      ├────────────────────────────────────────────┐
      ▼                                            ▼
R1  Runtime integrity remediation           R2  gen4 engineering preflight
    (R1-a … R1-o; 72 h SOAK acceptance)          (separate host, ≥ 30 days, DIAGNOSTIC)
      │                                            │
      │                                            ▼
      │                                     R3  Scientific design, power module,
      │                                         campaign contract (PREREGISTERED;
      │                                         deciding cost semantics, validity,
      │                                         extension and successor rules frozen)
      │                                     ══ R3 merge: IRREVERSIBLE (a design change
      │                                        is a new preregistration) ══
      │                                            │
      │                                            ▼
      │                                     R4  gen4 contract freeze
      │                                         (verification classes; provenance flag;
      │                                          Lane B spot streams iff elected)
      │                                            │
      │                     ┌──────────────────────┼──────────────────────┐
      │                     ▼                      ▼                      ▼
      │              R5  Reconciliation     R7  Multi-clock causal   Production recorder
      │                  generalisation         MarketSnapshot +         deployed under the
      │                  (PR #76: PASS /        frozen feature fn        pre-activation hash:
      │                   FAIL branches)        (ENG-START-EARLY)        30-day clock STARTS
      │                     │                      │                      │
      │                     │  R5 → qualification: │                      │
      │                     │  a day counts only   │                      │
      │                     │  once R5 reconciles  │                      │
      │                     │  it; R5 freezes the  │                      │
      │                     │  qualification record│                      │
      │                     ├──────────────────────┼─────────────────────►│
      └─────────────────────┼──────────────────────┤                      │
                            │                      ▼                      │
                            │               R8  Multi-symbol single-leg   │
                            │                   futures runtime, one      │
                            │                   ledger, Aegis set scope   │
                            │                   (R8-a … R8-h;             │
                            │                    ENG-START-EARLY)         │
                            │                      │                      │
                            │                      │◄─── recorder root ───┤
                            │                      ▼   (pre-activation)   │
                            │               R9  Autonomous demo / soak    │
                            │                   ≥ 14 days, drills,        │
                            │                   PARITY every day;         │
                            │                   Lane B in its own record  │
                            │                   ──► R9 PASS (frozen       │
                            │                       ENGINEERING report;   │
                            │                       head record; shadow   │
                            │                       economics sealed)     │
                            │                      │                      │
                            └──────────┬───────────┴──────────────────────┘
                                       ▼
                     Independent boundary review — own governance PR, after
                     R9 PASS; reads the soak report, the contract, the power
                     report and the R5 qualification record; ≥ 7 days after
                     the R3 merge; may not reference sealed economics
                                       │
                          ┌────────────┴────────────┐
                          ▼                         ▼
══════  R6  PROSPECTIVE BOUNDARY ACTIVATION  ══   DECLINED (owner; reason recorded;
══      IRREVERSIBLE — requires R9 PASS      ══   slot consumed; seal never opened)
                          ▼                         │
                     R10  First prospective campaign│
                          (head-frozen runtime; no-peek; CI-run block
                           validity → N VALID blocks or extension ≤ E)
                          ── first scored day is irreversible for the candidate ──
                                       │
         ┌──────────────┬──────────────┼──────────────────┐
         ▼              ▼              ▼                  ▼
   NEGATIVE        ABORTED        NOT EVALUABLE /    PASS information +
   (immutable;     (owner stop    ENGINEERING        economic screen
    family         or hashed-     FAILURE            NOT FALSIFIED
    retired)       module change; (E exhausted;             │
         │          slot consumed) readouts sealed)          ▼
         │              │              │           R11  Confirmation — same
         │              │              │                frozen object, same
         │              │              │                deciding economics
         │              │              │                ──► CONFIRMATORY PASS
         │              │              │                    = Gate 1 (eligibility
         │              │              │                      for consideration only)
         │              │              │                      │              │
         ▼              ▼              ▼                      ▼              │
   R12  Strategy expansion governance (sink for every outcome and     │
        for DECLINED; budget ≤ 2 directional + ≤ 1 carry, per         │
        candidate one R10 + ≤ 1 automatic re-run + one R11)           │
         │                                                            │
         ├─ re-entry (new candidate): R12 → R3 → R7 → R9 re-soak ──┐   │
         │            → new R6 (all eight inputs) → R10          │   │
         ├─ re-entry (NOT EVALUABLE, automatic same-design        │   │
         │            successor, at most once): R9 re-soak ───────┤   │
         │            → new R6 → R10                              │   │
         │                                                        │   │
         └──► R13  Portfolio / orchestration                      │   │
                   (ONLY with ≥ 2 CONFIRMATORY families;          │   │
                    the second arrives via Lane B or a re-entry)  │   │
                                                                  │   │
      R14  Live-readiness qualification ◄─── Gate 1 candidates only ──┘
           (Gates 2, 3; Gate-4 contract text; R9 gates re-run on
            the exact deployable head with the adapter in testnet mode)
              │
══════════════════════  REAL-MONEY AUTHORITY BOUNDARY  ══════════════════════
══  Crossing needs ALL FOUR, separately recorded: (1) SCIENTIFIC = Gate 1,      ══
══  (2) ENGINEERING = R9 PASS + R14 deployable-head re-soak + Gate 2,          ══
══  (3) OPERATIONAL = Gate 3, (4) OWNER AUTHORISATION = Gate 4 signed.         ══
══  IRREVERSIBLE once an order exists.                                         ══
              │
              ▼
      R15  Owner authorisation (Gate 4 executed; revocable until an order exists)
              ▼
      R16  Canary (Gate 5) — the first real order is irreversible
              ▼
      R17  Controlled real-money operation
              ▼
      R18  Scale-up / scale-down governance (Gate 6)

Side lane (SECONDARY, OPTIONAL; never on the critical path):
      Lane B carry  ──  R1 → own PREREGISTERED contract (in R3, or its own PR
                        merged ≥ 7 days before R6 if elected later; own hash,
                        own slot) → spot streams in R4 iff elected (deadline R4)
                        → own soak record inside R9 (never enters the primary
                        verdict) → opens on the same R6 boundary iff its PR is
                        merged and its soak record passed, else waits for its
                        own later contract and boundary; own process, state
                        root, Aegis instance and ledger. It never delays
                        R1–R9 or R6. On any resource conflict the primary
                        lane wins.

Back-edge not on the critical path: R5 (a stream downgraded in a new
contract version) → R4 re-freeze: the qualification clock restarts under
the new hash; R7/R9 acceptance is re-established if any declared stream
or clock changed.
```

**Irreversible transitions, in order:** the R3 preregistration merge (design cannot change without a new preregistration); the R6 activation commit (`prospective_from` set; a fresh storage root; no mixed hashes); R10's first scored day (the candidate is consumed); the real-money boundary (four conditions); R16's first real order. R15's authorisation is revocable until an order exists and is therefore not on this list.

---
## Part C — COMPLETE REVISED §37: MASTER ROADMAP (PROPOSAL, adoption-ready)

This section replaces §37 of the audit in full.

**Status:** PROPOSAL produced by an independent read-only audit and corrected by this corrigendum. It creates no scientific evidence, no prospective boundary, no alpha claim and no real-money authority. Adoption is an owner decision recorded in a governance PR (R0).

**Starting state [RF, as of the audit]:** `main` = `46921ef1206748c6b7304432a26c8295b7830e27` (merge of PR #95). PR #76 open/draft at `8a8f4a1`, 129 commits behind `main`, with a known conflict in `tests/test_recorder_no_network.py`. gen3 recorder contract with `prospective_from = null`. No prospective data. No campaign protocol frozen. P4-HOLD retired/unread; Styx sealed/unread; P8 withdrawn; four outer historical blocks BURNED. Real-money authority: NONE.

**Primary lane, stated once:** Binance USD-M perpetual futures, multi-symbol, single-leg, directional LONG/SHORT, at 1× leverage. Spot is reference data, a hedge leg where a lane needs one, and support infrastructure. The carry lane (Lane B) is SECONDARY and OPTIONAL: it never delays the primary lane, has its own hypothesis slot, is not a prerequisite for anything, is not orchestration, and yields on any resource conflict.

### 37.0 Phase map

```
ID   phase                                                     class                         irreversible?
R0   Adoption & freeze (owner)                                 governance                    no
R1   Runtime integrity remediation                             engineering                   no
R2   gen4 engineering preflight (separate host)                engineering + DIAGNOSTIC      no
R3   Scientific design, power module, campaign contract        PREREGISTERED design          yes (merge)
R4   gen4 contract freeze                                      data governance               no (until R6)
R5   Reconciliation generalisation (PR #76 → gen4)             engineering  [PASS/FAIL branches]  no
R7   Multi-clock causal MarketSnapshot + frozen feature fn     engineering (MTF-capable)     no
R8   Multi-symbol single-leg futures runtime + Aegis set scope engineering                   no
R9   Autonomous demo / soak                                    ENGINEERING evidence only     no
R6   Prospective boundary activation  (AFTER R9 PASS)          irreversible governance act   YES
R10  First prospective campaign (accrual, no-peek)             PROSPECTIVE evidence          YES (first scored day)
R11  Economic confirmation and continuation                    CONFIRMATORY evidence         no
R12  Strategy expansion governance                             science governance            no
R13  Portfolio / orchestration (conditional)                   engineering                   no
R14  Live-readiness qualification (Gates 2–3, Gate-4 text)     engineering + operational     no
================  REAL-MONEY AUTHORITY BOUNDARY (four conditions)  ================  YES
R15  Owner authorisation (Gate 4 executed)                     owner governance              no (revocable until an order exists)
R16  Canary (Gate 5)                                           operational                   YES (first order)
R17  Controlled real-money operation                           operational                   —
R18  Scale-up / scale-down governance (Gate 6)                 owner + operational           —
```

Listed in execution order. IDs are stable labels inherited from the audit; **R6 executes after R9** (Part B). Dependencies, exhaustively: R1 ∥ R2 (both need R0); R3 needs R2 (≥ 30 measured days); R4 needs R3 (and the Lane B election, deadline R4); R5 needs R4; R7 needs R3 (frozen feature function) and R4 (declared streams/clocks) for acceptance, ENG-START-EARLY for the generic snapshot type; R8 needs R1 and R7 for acceptance, ENG-START-EARLY for the runtime skeleton; R9 needs R8 and a running gen4 recorder (the production recorder's pre-activation root from R4, or the R2 preflight recorder if R2 records that it keeps running); the ≥ 30-day production-recorder qualification starts at R4's production deployment and accrues in parallel with R5 and R7–R9, but a day counts only once R5's accepted reconciliation has been run over it (offline recomputation, retroactive where needed), so the qualification record — produced by R5 — closes only after R5 acceptance; R6 needs R3 (merged ≥ 7 days earlier), R4, R5, **R9 PASS**, the completed qualification and the independent boundary review; R10 needs R6; R11 needs R10 PASS on both floors; R12 follows any R10/R11 outcome; R13 needs ≥ 2 CONFIRMATORY families; R14 needs R11 Gate 1 for the candidate that will trade and re-runs the R9 gates on the exact deployable head; R15–R18 are sequential behind the real-money boundary. Re-entry edges: R12 → R3 (a new candidate) → R7 feature-function PR → R9 re-soak on the new head → a new R6 with all eight inputs → R10; R12 (NOT EVALUABLE; the automatic same-design successor, at most once) → R9 re-soak → new R6 → R10; R6 → DECLINED → R12; R5 (a stream downgraded in a new contract version) → R4 re-freeze, restarting the qualification clock under the new hash and re-establishing R7/R9 acceptance if any declared stream or clock changed.

### 37.1 Governance classes and status labels used throughout

| class | meaning | may be reused for |
|---|---|---|
| ENGINEERING | tests, invariants, soak logs, replay parity, coverage | any later stage; never as alpha evidence |
| DIAGNOSTIC | descriptive statistics on engineering-root or pre-activation data (rates, spreads, correlations, coverage, cost envelopes) | design inputs before a boundary; never as a result |
| EXPLORATORY | any model/feature/rule evaluation on pre-boundary or engineering-root data | hypothesis generation; must be labelled; never promotable |
| PREREGISTERED | a hashed design committed and pushed before the data it governs exists or is read | governs exactly one campaign |
| PROSPECTIVE | evidence accrued at or after an activated `prospective_from` under a frozen contract and preregistration | promotion decisions |
| CONFIRMATORY | a second, independent prospective period under the same frozen candidate and the same deciding economics that does not contradict the first | eligibility for real-money consideration (Gate 1) |
| ADAPTIVE | any evaluation whose design was chosen after seeing outcomes from the same data | disclosure only |
| BURNED | the four outer historical blocks; P4-HOLD; anything read ≥ 2 times for deciding purposes | descriptive context, fixtures; never deciding |
| SEALED-DIAGNOSTIC | a hashed artifact that exists but may not be read before a named event: the soak's shadow economics until R10 closes; a NOT EVALUABLE campaign's VALID-block readouts until its successor closes or the candidate is retired; a DECLINED or ABORTED candidate's sealed material, which is never opened | opened only at the named event, then DIAGNOSTIC (or ADAPTIVE context where the label says so); never for descriptive context before it |

Block and campaign status labels (R10/R11): a scored block is **VALID** or **INVALID** (mechanical, blind to outcomes; INVALID blocks stay in chronology and reporting forever and are never a negative result); a campaign ends **PASS**, **NEGATIVE** (scientific), **NOT EVALUABLE / ENGINEERING FAILURE** (only: the maximum extension exhausted before N valid blocks), or **ABORTED** (an owner stop, or a change to a contract-hashed module during accrual: not PASS, not NEGATIVE, not NOT EVALUABLE; the reason recorded; the candidate's slot consumed; no same-design rerun; all readouts stay sealed). A candidate whose owner declines activation after a passed soak is **DECLINED** (slot consumed; sealed shadow economics never opened; re-election only via a new R3, a fresh soak and a fresh seal).

Every artifact directory carries an `evidence_class` field in its manifest; `tools.verify_research_state` refuses a front-door document that cites a non-PROSPECTIVE artifact as prospective evidence, a non-CONFIRMATORY artifact as confirmatory evidence, or an INVALID block as a result (new check; R1-o).

### 37.2 Git / PR policy (all phases)

Ordinary commits; ordinary merge commits; no amend/squash/rebase/force-push on `main` or on any branch carrying a preregistration, a boundary activation, an evidence freeze, a validity verdict or a negative result; immutable negative results; PR-specific exact-head CI required for every merge into `main`; every scientific/governance PR names its exact base SHA and the SHA it produces; large research PRs stay draft until independently audited; evidence directories are frozen with SHA256SUMS in the same PR that creates them; a failed acceptance, an INVALID block or a NOT EVALUABLE campaign is never replaced by a later one (a new attempt is a new directory with a new prospective declaration); preregistration PRs sit ≥ 7 calendar days between merge and first scored instant with one independent review recorded in the PR. **Head-freeze rule (R9 → R10 → R11):** the runtime head that passed R9 is the head that runs the campaign; the only permitted changes are ENGINEERING-FIX PRs (reviewed; no change to any contract-hashed module — feature function, model artifact loader, policy, cost model, endpoint evaluator, validity evaluator; replay parity re-proven on ≥ 3 prior days; recorded as an engineering event in the campaign log). Any other change to a non-contract-hashed module lands as an INVALID block and requires a re-soak (on an engineering recorder root or, read-only and unscored with its shadow economics sealed, on the prospective root) before the next block is scored; any change to a contract-hashed module ends the campaign as ABORTED — the object changed — and any successor is a new preregistration (route C). **Validity-before-readout rule:** a block's validity verdict is computed by the CI/verifier job, never by hand, from engineering records only; the job commits the verdict, its input manifest and its hash in the block's evidence PR before the readout job runs; that PR is reviewed by a person who certifies having read no readout of the block; a manually produced verdict is itself an invalidity criterion.

### R0 — Adoption and freeze

**PURPOSE:** record the owner's decision on this roadmap and the dispositions it implies.
**WHY:** nothing below is legitimate without an explicit adoption; the roadmap's budget and lane structure must be fixed before work starts.
**INPUT PREREQUISITES:** the completed audit and this corrigendum.
**EXACT WORK:** (a) a governance PR adopting or declining this roadmap by section, naming the audit SHA and this corrigendum; (b) disposition of PR #76 under R5's branch rule (it is not merged in R0); (c) a record that gen3 stays engineering-only and that its `prospective_from` stays `null`; (d) the finite budget: at most **two directional candidates** and at most **one carry candidate** across the programme before a mandatory stop-and-decide review, each candidate holding one first campaign (R10), at most one automatic same-design re-run after a NOT EVALUABLE outcome, and one continuation (R11), with ABORTED and DECLINED consuming the slot; zero deciding use of burned blocks; an engineering effort cap for R1–R9 set by the owner (the audit suggests ≤ 10 person-weeks); (e) the Lane B election opened (elect / decline / defer to R4); (f) the standing rule that the primary directional lane wins any resource conflict.
**TECHNICAL DESIGN:** documents only; regenerate front-door state blocks with the existing tooling.
**SCIENTIFIC DESIGN:** none.
**DATA USED:** none.
**DATA NOT USED:** everything.
**TESTS / VERIFICATION:** `tools.verify_research_state`; CI on the exact head.
**ACCEPTANCE:** decision recorded with date and SHA; budget and lane rule visible in `docs/current_development_plan.md`.
**KILL / STOP CONDITIONS:** owner declines → Roadmap v3 stands unchanged and this proposal is archived as a document.
**EVIDENCE CREATED:** none.
**EVIDENCE NOT CREATED:** any.
**PROSPECTIVE STANDING:** unchanged.
**REAL-MONEY STATUS:** none; no authority created.
**DEPENDENCIES:** none.
**NEXT MAY ASSUME:** an adopted plan, a fixed budget, the lane rule, and a recorded PR #76 disposition.
**PR BOUNDARIES:** one docs/governance PR.

### R1 — Runtime integrity remediation (engineering prerequisites)

**PURPOSE:** make the existing runtime honest before it grows.
**WHY:** the demo cannot start under the campaign profile, cannot run as a service, cannot clear its own disputes, cannot detect a dead feed, skips most minutes as stale, can halt a healthy campaign at a day roll, and cannot prove parity across a restart [RF §§7, 10, 11]. Growing a multi-symbol runtime on top would carry every defect forward.
**INPUT PREREQUISITES:** R0.
**EXACT WORK** (each item a separate reviewed engineering PR with a two-sided test; none changes scientific standing):
- **R1-a Source identity / CAMPAIGN self-check:** `tools/demo_run.py::_software` passes the checkout root to `source_identity()` and reads the mapping by key; a genuinely dirty tree is refused on CAMPAIGN, a clean tree passes; the committed campaign config must build a runner.
- **R1-b AEG-1 persisted-equity re-seeding:** `build_risk_engine` never calls `update_equity(capital)` after loading a persisted state; capital seeds equity only on first start; on restart the persisted equity is reconciled against the ledger and a disagreement is a dispute; two-sided tests: (a) a restart across a UTC day boundary after a ≥ 2% intraday mark move → no halt; (b) a persisted peak ≥ capital/(1 − max_drawdown_pct) → no halt on restart, while a genuine breach still halts; (c) replay PARITY across a day boundary with a restart (`risk.state_hash` equal).
- **R1-c `risk.json` continuity (AEG-4):** a missing or unreadable `risk.json` when the decision log already holds records is a dispute, not a default; the loaded snapshot is compared with the log's last `risk.state_hash`/HALT record; a `RECOVERY` record is written; the same continuity the store and ledger already have.
- **R1-d Daemon / service semantics:** a continuous READY loop (sleep to the next expected minute close + grace), graceful SIGTERM deferred past PERSISTENCE, heartbeat every 30 s, persistent metrics endpoint; the systemd unit becomes a real long-running service; `RunnerDown` becomes a liveness alert; process-per-pass supervision is removed.
- **R1-e Clock hardening:** RiskEngine and executors refuse construction without an injected clock under CAMPAIGN/SOAK profiles; no `time.time` default reachable from the demo path; the daemon's operational clock (used only for staleness, heartbeats and the stall tick) is itself injected and distinct from the decision clock; a clock-hostility test.
- **R1-f Real staleness detection:** the age is computed as the injected operational wall clock minus the newest available minute close (or as the recorder heartbeat age), never as the decision clock compared with itself (today the comparison is identically zero by construction); compared against `max_data_delay_s` in the runner's READY gate, outside the per-order decision path; a stale feed produces a recorded `FEED_STALLED` halt (positions held; funding and liquidation checks continue on the last valid state via a stall tick driven by the operational clock), automatic continuation when data resumes; two-sided test: a frozen recorder with advancing wall time halts within threshold + grace, a healthy feed never halts; `max_data_delay_s` is enforced here or removed from the schema.
- **R1-g Recorder / runner cadence:** the recorder publishes the last closed minute incrementally (seconds, not the 300 s full-day cadence); the runner decides every closed minute since its last decided minute (no three-minute cap) or records each skip with a reason; a minute whose close coincides with a funding instant is decidable only once its settlement row exists in the recorder output (or the recorder has marked the instant as no-settlement) — a late settlement row defers the decision rather than skipping the booking; two-sided test: live and replay book the same settlement in the same minute; acceptance: zero `SKIPPED_STALE` minutes while the recorder is healthy.
- **R1-h Atomic parquet publication and recorder evidence:** temp-file + fsync + rename for every parquet and manifest write; an in-process silence watchdog that reconnects on stream silence; persisted `recorder.down` / `recorder.up` records in the recorder's own log so a dead recorder leaves evidence; gen3's contract hash is unchanged by any of this (gen3 stays engineering-only).
- **R1-i Dispute lifecycle and crash/restart semantics:** every dispute kind has exactly one clearing path (`resolve` reachable from the CLI, requires a note, re-books what it must, refuses what it cannot); `resume` and `resolve` succeed from the CLI via `start()` with log-tail clock seeding; `flatten` on CAMPAIGN without `allow_dirty` and without persisting a spurious halt reason; `HedgedPosition.correct()` wired with a bounded timeout or deleted; one-legged positions liquidation-checked per leg; `resume` ordering fixed; the state-machine crash harness (kill at every persistence step, recover, assert consistency) added to CI, including a disk-full fault; on any persistence failure (ENOSPC included) the runner exits non-zero after attempting to append the halt reason to a pre-allocated emergency record, so a restart never lacks a trace.
- **R1-j Replay parity policy:** decide `seq` (per-kind sequence or exclusion of operational kinds) and `OPERATOR` (deterministic replay counterpart from a committed operator-action file); re-include deterministic risk fields in the hash; the 48 h synthetic fixture reaches PARITY with a restart and an operator action.
- **R1-k Unreachable Aegis rules:** the funding-cost entry veto made reachable (pass `funding_rate` to `execute_target`) or removed; loss-streak and cooldown limits either driven by a real `record_trade_result` caller or removed from the config schema; each wiring/removal has a breach-refused and non-breach-allowed test; the unsafe-default detector (a configured limit that no code path enforces) added to CI.
- **R1-l Unified valuation price:** one documented valuation price (mark) for equity, drawdown and liquidation; the close-vs-mark inconsistency removed from the ledger path that R8 will inherit.
- **R1-m Retire the live-capable legacy:** delete the Freqtrade path, strategies, exchange configs, its Dockerfile, the three tests that import it and the `trade` extra; keep `make smoke` by moving it off `nn.infer_service` or retire it with the inference service.
- **R1-n Supply chain and operations:** SHA-pin GitHub Actions; `--hash` the lock; digest-pin base images; an off-host dead-man check fed by recorder and runner heartbeats; documented chrony requirement with the existing skew alert; backup/restore drill for the recorder root and the state directory; separate Unix users and state roots for recorder and runner.
- **R1-o Evidence-class labels:** the `evidence_class` manifest field, the 37.1 status labels, and the `verify_research_state` checks described there.
**TECHNICAL DESIGN:** no new abstractions; fixes inside existing modules; the four-item minimum test architecture (state-machine crash harness; clock hostility; replay-determinism CI job; unsafe-default detector) plus a synthetic-soak CI job.
**SCIENTIFIC DESIGN:** none.
**DATA USED:** synthetic fixtures; gen3 engineering recorder data.
**DATA NOT USED:** burned blocks, P4-HOLD, Styx, any sealed data, for any purpose.
**TESTS / VERIFICATION:** the unit and two-sided tests named per item; the four harnesses; exact-head CI per PR.
**ACCEPTANCE:** CI green on the exact head of each PR; a 72 h SOAK-profile run unattended on an engineering recorder root with one planned restart and one `SIGKILL`, PARITY on every day, zero `SKIPPED_STALE` minutes while the recorder was healthy, zero manual state edits.
**KILL / STOP CONDITIONS:** none scientific; an item whose fix would require a design change beyond its module is split into its own reviewed PR rather than widened.
**EVIDENCE CREATED:** ENGINEERING.
**EVIDENCE NOT CREATED:** any scientific evidence.
**PROSPECTIVE STANDING:** unchanged.
**REAL-MONEY STATUS:** none.
**DEPENDENCIES:** R0.
**NEXT MAY ASSUME:** a service-shaped runner whose decisions replay across restarts, whose halts and disputes are clearable from the CLI, whose Aegis limits all enforce something, and a recorder whose files are atomic and whose last minute is fresh.
**PR BOUNDARIES:** R1-a … R1-o as separate PRs; no PR touches a scientific artifact or a contract hash.

### R2 — gen4 engineering preflight (separate host)

**PURPOSE:** measure before freezing.
**WHY:** v3's 12–20 symbols and 182 days are unmeasured guesses; storage was already under-estimated 24× once [RF].
**INPUT PREREQUISITES:** R0; a second host that is never the PR #76 VPS; no interaction with the gen3 acceptance environment.
**EXACT WORK:** multi-symbol capture under an ENGINEERING contract id (`gen4-preflight`; `prospective_from` absent by schema) of Tier A streams (1 m klines, markPrice@1s, bookTicker, aggTrade, funding, OI via REST, forceOrder, daily exchangeInfo snapshots) for ~20 USD-M perpetuals chosen by 90-day median USD volume as of a named date, plus Tier B (depth20@100ms or diff depth with snapshot bootstrap) for BTCUSDT and ETHUSDT, plus — if Lane B is elected or undecided — spot kline_1m and spot bookTicker for BTCUSDT and ETHUSDT, for ≥ 30 consecutive days. Measure: events/s, bytes/s raw and gz, CPU, RAM, disk/day, projected 182 d/365 d, replay throughput, recovery time, reconnect behaviour, REST weight use, archive availability per stream, missingness, depth sequence-continuity rate, clock skew, cross-stream consistency (kline vs aggTrade OHLCV; bookTicker vs depth top). Compute the cross-symbol daily-return correlation matrix and the effective number of symbols; compute the per-symbol cost envelope (median/p90 half-spread at decision instants, realised trade-through for a reference size, fee tier, funding interval).
**TECHNICAL DESIGN:** the gen3 recorder generalised to (symbol, stream)-keyed files; raw NDJSON.gz per stream-day immutable; normalised parquet regenerable; contract hash + code version + schema version stamped in every day manifest; the R1-h atomicity and watchdog carried over.
**SCIENTIFIC DESIGN:** none — DIAGNOSTIC only; no model, no label, no return statistic beyond correlation, volatility and cost envelopes.
**DATA USED:** live public streams on the preflight host.
**DATA NOT USED:** burned blocks; no outcome-bearing statistic of any candidate.
**TESTS / VERIFICATION:** synthetic-server tests for new parsers; parity tests full-vs-incremental normalisation per stream; manifest checksum verification.
**ACCEPTANCE:** a preflight report with every measurement above, frozen with SHA256SUMS, `evidence_class = DIAGNOSTIC`.
**KILL / STOP CONDITIONS:** ≥ 2 of the 30 days failing coverage thresholds for core streams → fix operations before R4; Tier B cost beyond budget → Tier B shrinks or is deferred (recorded).
**EVIDENCE CREATED:** DIAGNOSTIC.
**EVIDENCE NOT CREATED:** prospective; exploratory candidate evaluations.
**PROSPECTIVE STANDING:** unchanged.
**REAL-MONEY STATUS:** none.
**DEPENDENCIES:** R0.
**NEXT MAY ASSUME:** measured rates, a correlation matrix and effective-N, per-symbol cost envelopes, archive verification classes per stream.
**PR BOUNDARIES:** recorder-generalisation PRs (offline core first, then live collection), then the preflight report PR.

### R3 — Scientific design, power module and campaign contract

**PURPOSE:** choose an answerable question and freeze it, including everything that will decide R10 and R11.
**WHY:** the historical programme measured noise with an instrument that could not see its effects; v3 says power before boundary but defines no method; a confirmatory period must replicate, not re-score.
**INPUT PREREQUISITES:** R2's diagnostics (correlation matrix, cost envelopes, rates, archive classes); the standing negative priors; the Lane B election status.
**EXACT WORK:**
1. *The frozen candidate (Lane A):* instrument (USD-M perpetuals); universe rule and K (§18 procedure, provisional 10–12); decision clock (1 h); horizon H by the κ rule (likely 4–8 h, funding-aligned); the deciding feature set = features on the decision clock plus **at most two preregistered slow closed-bar variables** (e.g., 1 d realised volatility; 1 d or 4 h trend sign), each with its clock, `available_time` rule and maximum age declared; model class with fixed hyperparameters (Tier 0 references; Tier 1 linear with λ from a predeclared blocked CV on pre-boundary engineering data; optional Tier 2 challenger, m = 2); calibration map; abstention (analytic threshold) and hysteresis; target-position mapping (per-symbol side/quantity/confidence/abstain under portfolio caps); the coherence statement (cross-sectional endpoint ↔ panel policy ↔ multi-symbol single-leg runtime).
2. *Frozen deciding economic semantics (used identically in R10 and R11), as hashed code:* per-symbol fee tier and BNB-discount assumption dated to the fetch; spread rule (recorded bookTicker touch at the decision minute close); executable-price rule (buy at ask, sell at bid, at decision time + Δ); latency Δ (frozen constant; the quote used is the first at or after t + Δ); slippage: a conservative per-symbol envelope (p90 measured trade-through for the reference size from R2, as a frozen constant or a frozen deterministic function of size and spread); funding by settlement at each recorded funding timestamp with the symbol's interval (non-8 h symbols excluded from Lane A); turnover accounting (fees and spread on every change of target notional; hysteresis inside the policy); reject/partial-fill treatment (dry-run venue model: full fill at the executable price; an engineering-rejected order is a missed decision with no phantom fill; the rule is frozen); impact rule (the campaign's dry-run notional per symbol is frozen below the measured depth threshold, so impact is outside the deciding model; anything above is stress only). ±50% cost sensitivity is reported, never deciding.
3. *The ESS/MDE/inference module,* frozen as code and report (§22's three layers: analytical N_eff; stationary bootstrap with automatic block length; whole-pipeline null simulation calibrating the exact decision rule including Holm over m ≤ 2), run on R2 engineering data and null controls only; outputs: N_eff, MDE (IC, bps net, annualised Sharpe), duration to floor, **the target number N of VALID scored blocks**, block length (calendar month, UTC), the alpha-spending interim-look rule, and the R11 continuation size N′ (or its deterministic sizing rule in the measured R10 effect).
4. *The campaign contract:* endpoints and floors (§27); blocks; **block-validity rules**: (i) mechanical invalidity criteria (any required stream below its coverage threshold on more than the frozen number of days in the block; a replay-parity failure on any day; a runner halt not recovered within the frozen window or any halt without a record; a recorder identity or contract-hash change; a runtime head change other than an ENGINEERING-FIX PR; any manual state edit; a symbol delisting/halt → that symbol's block invalid, the whole block invalid if the contract's minimum symbol count is not met); (ii) the evaluator `nn/prospective/validity.py`, reading engineering records only and no prices, scores, positions or PnL; (iii) validity-before-readout ordering; (iv) the deterministic calendar-extension rule (contiguous blocks until N VALID blocks); (v) the maximum extension E (proposal: E = ⌈N/2⌉ blocks; frozen); (vi) exhaustion → NOT EVALUABLE / ENGINEERING FAILURE; (vii) the pre-committed successor rule: at most one automatic same-design successor after a NOT EVALUABLE outcome (new hash, new boundary, re-soak first, no pooling), taken once the engineering cause is fixed and never decided on outcomes; the predecessor's VALID-block readouts stay SEALED-DIAGNOSTIC until the successor closes or the candidate is retired; a second exhaustion is terminal for the design and consumes its slot; an owner stop, or a change to a contract-hashed module during accrual, is ABORTED (slot consumed, no same-design rerun, readouts sealed); (viii) INVALID blocks permanently reported; stopping rule; minimum duration; no-peek rule; engineering-invalidation vs scientific-failure split; PASS/NEGATIVE statements; negative controls; the historical-adaptivity disclosure block; the head-freeze rule; **the R11 rule**, with its non-inferiority statement and N′: R11 uses the identical candidate and identical deciding economics; new execution/testnet/live-operational information enters only (A) through a deterministic update rule preregistered here that can never make the deciding economics less conservative than R10's (e.g., "replace the slippage envelope with max(frozen p90, p90 of ≥ 500 testnet fills) computed by frozen code"), (B) as non-deciding stress/diagnostic analysis, or (C) by a new preregistration and campaign.
5. *Lane B contract (only if elected):* own hash; BTCUSDT (ETHUSDT optional); always-on or funding-gated two-leg carry rule with parameters frozen from pre-boundary funding statistics; endpoint per §27 Lane B; m = 1 in its own slot; the same validity, extension, no-peek, head-freeze and shadow-economics-seal rules and the same DECLINED consequence; a Lane B elected after this PR gets its own preregistration PR (own hash, own review) merged ≥ 7 days before R6, otherwise it waits for its own later contract and boundary.
6. *Independent pre-open review* recorded in the PR; **≥ 7 days** between the preregistration merge and the boundary.
**TECHNICAL DESIGN:** `nn/prospective/` package: contract JSON hashed as P13's pattern; the ESS/MDE code; the endpoint evaluator that reads only the recorder root and the decision log; the validity evaluator; the cost-model code imported by both the runtime and the evaluator so the same bytes decide.
**SCIENTIFIC DESIGN:** per §27 as corrected (two-stage in one accrual: information primary, economic falsification screen gating R11).
**DATA USED:** R2 engineering-root data for null calibration, cost measurement, and the one-time predeclared λ-selection blocked CV and coefficient fit of the Tier 1 candidate (procedure and outputs hashed, recorded in the historical-adaptivity disclosure block, labelled EXPLORATORY, never a go/no-go input).
**DATA NOT USED:** burned blocks for any deciding purpose; no evaluation of the candidate's endpoint or economic screen on any pre-boundary data other than the predeclared λ-selection CV and the null-calibration controls, and none of it reported or used as evidence.
**TESTS / VERIFICATION:** hash tests; two-sided synthetic controls for the endpoint code (planted signal detected; null passes at the nominal rate); two-sided controls for the validity evaluator (a planted coverage gap → INVALID; a clean block → VALID; the evaluator has no import path to price or PnL data); two-sided controls for the cost code; a two-sided coherence control (on a synthetic block the evaluator's economic readout recomputed from the decision log equals the runtime ledger's simulated PnL to the cent, and a planted mismatch between the policy's target set and the evaluator's assumed positions is detected).
**ACCEPTANCE:** contract merged with hash recorded; review recorded; power report shows MDE ≤ the floor at the planned N (or N raised until it does); the deciding economic code hashed into the contract; the R11 rule present.
**KILL / STOP CONDITIONS:** no design reaches adequate power within an accrual of ≤ 18 months including the maximum extension → do not proceed to R4 for that design; record it as declined.
**EVIDENCE CREATED:** PREREGISTERED (design); DIAGNOSTIC (power report).
**EVIDENCE NOT CREATED:** any result.
**PROSPECTIVE STANDING:** unchanged.
**REAL-MONEY STATUS:** none.
**DEPENDENCIES:** R2 (≥ 30 days). ENG-START-EARLY: the module code and contract templates may be written during R2; acceptance needs R2's outputs.
**NEXT MAY ASSUME:** a frozen, reviewed, powered contract whose deciding code, validity rules, extension rule and R11 rule cannot change without a new preregistration.
**PR BOUNDARIES:** one preregistration PR per lane (draft until reviewed); the `nn/prospective/` code PRs precede it.

### R4 — gen4 contract freeze

**PURPOSE:** create the acquisition identity for the prospective recorder.
**WHY:** a stream absent from the contract cannot serve the campaign; verification classes must match what first-party archives actually publish [EF, audit §§12, 17].
**INPUT PREREQUISITES:** R2 report; R3's universe and streams; the Lane B election (deadline: this phase).
**EXACT WORK:** the contract JSON: exact symbols; exact streams per tier with verification class — ARCHIVE_RECONCILED (klines, aggTrades, markPriceKlines, indexPriceKlines, premiumIndexKlines, fundingRate via the monthly archive with its latency rule, metrics at 5-minute cadence, bookDepth as a sampled band file only); HEALTH_GATED (markPrice@1s; bookTicker, whose daily archive ended 2024-03-30; depth diffs/snapshots; all self-attested with health metrics and no denominator); DESCRIPTIVE_ONLY (forceOrder, which carries only the largest liquidation per symbol per second; OI/long-short REST histories; exchangeInfo snapshots); **plus spot kline_1m and spot bookTicker for Lane B's symbols as HEALTH_GATED capture iff Lane B is elected**; minute-key and missingness semantics; a per-minute provenance flag in the normalised schema (LIVE / REST_FILLED / LATE) with a test that a REST-filled minute is so labelled; storage layout (raw immutable, normalised regenerable); checksum scheme; metadata policy (daily exchangeInfo snapshot; leverage brackets; fee schedule assumption recorded); source identity; `prospective_from: null`.
**TECHNICAL DESIGN:** contract schema versioned; the contract hash binds code version and schema version in every day manifest.
**SCIENTIFIC DESIGN:** none.
**DATA USED:** R2 measurements.
**DATA NOT USED:** any outcome data.
**TESTS / VERIFICATION:** contract-schema tests; a test that every stream named by R3's feature function and cost model is in the contract.
**ACCEPTANCE:** independent plan review; hash recorded; production deployment of the recorder under this pre-activation hash begins; the ≥ 30-day qualification clock starts here and accrues in parallel with R5 and R7–R9, but a day counts as qualifying only once R5's accepted reconciliation has been run over it (offline recomputation, retroactive where needed), so the qualification record closes only after R5 acceptance; any coverage or reconciliation failure restarts the clock.
**KILL / STOP CONDITIONS:** none.
**EVIDENCE CREATED:** none.
**EVIDENCE NOT CREATED:** any.
**PROSPECTIVE STANDING:** unchanged.
**REAL-MONEY STATUS:** none.
**DEPENDENCIES:** R2, R3.
**NEXT MAY ASSUME:** a frozen contract that R6 will activate; a stream added later cannot be used by the campaign whose contract lacks it, and adding a stream to a new contract does not invalidate an existing campaign.
**PR BOUNDARIES:** one contract PR; the production-deployment PR.

### R5 — Reconciliation generalisation (PR #76 → gen4), with explicit branches

**PURPOSE:** recorder validity for gen4.
**WHY:** PR #76's reconciliation and coverage code is fail-closed and well-tested offline but single-symbol and 129 commits stale [RF]; gen4 needs (symbol, stream)-keyed reconciliation with the archive classes of R4.
**INPUT PREREQUISITES:** R4; the outcome of PR #76's second acceptance campaign, whichever it is (the roadmap does not wait for it).
**EXACT WORK:**
- *IF campaign #2 PASSES:* it establishes only that the gen3 recorder captured and reconciled two days on that VPS (ENGINEERING). Merge PR #76 on a fresh current-`main` lineage with an ordinary merge (resolve the `tests/test_recorder_no_network.py` conflict; exact-head CI); then generalise `reconcile.py`/`coverage.py` to (symbol, stream) with the contract's `minute_indexed_required` as the single source; add archive layouts for aggTrades, markPriceKlines, indexPriceKlines, premiumIndexKlines, metrics, bookDepth; keep HEALTH_GATED semantics; keep the funding monthly-archive latency rule (`FUNDING_SCHEDULE_UNAVAILABLE` is not an outage). No boundary activation follows from the pass.
- *IF campaign #2 FAILS:* classify the cause (recorder outage; archive-latency mis-read; VPS operations; reconciliation defect). Keep the offline code and tests; fix the cause in a reviewed PR; harden the preflight (silence watchdog, disk/CPU headroom, supervisor); no campaign #3 until a ≥ 7-day clean SOAK on the same host; campaign #3 is declared prospectively and cannot substitute failed days. gen4 proceeds on its own host regardless; PR #76's code is integrated into gen4 once its offline tests pass on gen4 layouts even if VPS acceptance is pending; the gen4 recorder's own qualification (R4 → R6) is the operative gate.
- *In both branches:* R1–R4, R7–R9 do not wait for the outcome.
**TECHNICAL DESIGN:** reconciliation as a pure offline recomputation over raw files and archives; every `ArchiveOutcome` fail-closed.
**SCIENTIFIC DESIGN:** none.
**DATA USED:** gen4 preflight and pre-activation raw files; first-party archives.
**DATA NOT USED:** any outcome data.
**TESTS / VERIFICATION:** offline determinism tests; real-source verification of every ARCHIVE_RECONCILED stream on ≥ 2 real days; fail-closed tests per outcome.
**ACCEPTANCE:** the above green on the exact head; reconciliation running on the production pre-activation recorder; the qualification record (≥ 30 qualifying days: per-day coverage, reconciliation agreement and source identity of the production recorder) frozen with SHA256SUMS once the clock completes.
**KILL / STOP CONDITIONS:** a reconciliation defect that cannot be made deterministic → the affected stream is downgraded to HEALTH_GATED in a new contract version before R6, never silently; a new contract version is a new R4 freeze: the qualification clock restarts under the new hash, and R7/R9 acceptance is re-established if any declared stream or clock changed.
**EVIDENCE CREATED:** ENGINEERING, including the qualification record that R6 inputs (5)–(6) consume.
**EVIDENCE NOT CREATED:** any scientific evidence.
**PROSPECTIVE STANDING:** unchanged.
**REAL-MONEY STATUS:** none.
**DEPENDENCIES:** R4.
**NEXT MAY ASSUME:** every ARCHIVE_RECONCILED stream is reconciled daily and deterministically; health-gated streams carry health metrics; the qualification record exists or its clock is running.
**PR BOUNDARIES:** the PR #76 merge (or its fix PR); the generalisation PRs per stream family; the qualification-record PR.

### R7 — Multi-clock causal MarketSnapshot and the frozen feature system

**PURPOSE:** one as-of-correct input snapshot per decision, generic over every clock the contract declares.
**WHY:** the live runtime must carry the causal semantics the research code already has (`nn/mtf.py`), and the runtime must not be architecturally limited to one timeframe — while the first campaign's science stays narrow.
**INPUT PREREQUISITES:** R3 (frozen feature function and declared clocks), R4 (declared streams). ENG-START-EARLY: the generic snapshot type and the closed-bar derivation may be implemented during R2–R4 against a schema of declared clocks.
**EXACT WORK:** (a) `MarketSnapshot`: for every input and every timeframe, `source_time`, `available_time` (= close time + the observed recorder processing delay from the raw receipt stamp), `as_of`, `age`, `staleness`, `complete`, `missing`; (b) closed-bar clocks (e.g., 5 m, 15 m, 1 h, 4 h, 1 d) derived causally from the base 1 m stream — a bar exists only when every constituent minute exists and the bar's `x == true` close has been received; (c) the feature function computed from the snapshot only, with the deciding set restricted by the contract (decision clock + ≤ 2 slow closed-bar variables); other declared clocks are computed and logged as capability, never fed to the deciding model in campaign 1; (d) no regime labels, no coherence state, no voting, no "HTF wins"; (e) the same function in research, replay and live. **MULTI-TIMEFRAME CAPABILITY = REQUIRED ENGINEERING PROPERTY. HIERARCHICAL MTC AS A SCIENTIFIC MODEL = NOT REQUIRED FOR CAMPAIGN 1.**
**TECHNICAL DESIGN:** a pure function `snapshot → feature vector`; clocks declared in the contract; feature-set membership enforced by a hashed allow-list; the snapshot logged per decision.
**SCIENTIFIC DESIGN:** none beyond the frozen feature set; a richer hierarchy may return only under a new preregistered campaign.
**DATA USED:** engineering-root or pre-activation recorder data.
**DATA NOT USED:** burned blocks; no outcome-bearing evaluation.
**TESTS / VERIFICATION:** two-sided synthetic leak controls (a planted future leak is detected; a clean series passes); the `nn/mtf.py` shift control as template; replay determinism; a test that a non-allow-listed clock cannot enter the deciding vector.
**ACCEPTANCE:** feature values from live snapshots equal features recomputed offline from the recorder root, byte for byte, over ≥ 7 days on every declared clock.
**KILL / STOP CONDITIONS:** none scientific.
**EVIDENCE CREATED:** ENGINEERING.
**EVIDENCE NOT CREATED:** any scientific evidence.
**PROSPECTIVE STANDING:** unchanged.
**REAL-MONEY STATUS:** none.
**DEPENDENCIES:** R3, R4 (acceptance).
**NEXT MAY ASSUME:** a causal, multi-clock snapshot whose deciding subset is fixed by hash.
**PR BOUNDARIES:** snapshot-type PR; closed-bar derivation PR; feature-function PR (hash-tested).

### R8 — Multi-symbol single-leg futures runtime and Aegis set scope

**PURPOSE:** replace the carry-shaped runner with the runtime the primary lane needs.
**WHY:** a cross-sectional endpoint is executed as a panel of single-leg positions; no such runtime exists [RF]; equity lives only in the demoted carry ledger; Aegis approves per order, not per target set.
**INPUT PREREQUISITES:** R1, R7 (acceptance). ENG-START-EARLY: the runtime skeleton may be built on the R1 runner during R3–R7.
**EXACT WORK:**
- **R8-a `TargetPositionSet`:** per campaign symbol: side, quantity, confidence, abstain; produced by the frozen candidate; the carry lane keeps `HedgeTarget` in its own process.
- **R8-b Execution:** one `chimera/futures` state machine per symbol (MARKET + reduce-only semantics, idempotent events, finite open-order set).
- **R8-c Single authoritative account ledger:** capital, free cash, per-position margin, fees, funding paid/received, realised, unrealised at mark; derived from the FuturesStore; the CarryLedger retired from the primary path and kept for Lane B.
- **R8-d Aegis set scope:** an `evaluate_target(TargetPositionSet)` entry point evaluating the set before any leg is sent (portfolio gross/net/per-symbol caps, order rate, staleness, dispute state, kill switch, funding cap, daily loss/drawdown from persisted equity); `evaluate_entry` kept for every exposure-increasing order; a set refused in part is refused whole; reductions ungated; scientific campaign limits (max trades/day, allowed symbols) enforced by the policy layer from the contract, not by Aegis; `record_trade_result` wired or its rules removed (finishing R1-k for the new path).
- **R8-e Crash consistency:** single write order store → ledger → log; the log arbitrates disagreement; recovery fail-closed and idempotent.
- **R8-f Per-symbol dispute isolation:** one symbol's dispute halts that symbol; portfolio halt only on limits.
- **R8-g Reconciliation:** dry-run labelled self-consistency; live semantics reserved for R14.
- **R8-h Continuous position guards inside Aegis (AEG-6):** liquidation distance and maintenance margin evaluated inside Aegis every tick from executor-computed `MarginState`; the maintenance rate read from `SymbolConstraints` only and the carry-layer constants (three copies of 0.004) deleted; two-sided test (a touch is halted; a non-touch is not) and a single-source test for the maintenance rate.
**TECHNICAL DESIGN:** reuse of `chimera/futures` domain/executor/store; one Aegis engine per runtime process; no message broker, database server or orchestration platform.
**SCIENTIFIC DESIGN:** none.
**DATA USED:** engineering-root or pre-activation recorder data; synthetic fixtures.
**DATA NOT USED:** burned blocks; no outcome-bearing evaluation.
**TESTS / VERIFICATION:** the state-machine crash harness; two-sided limit tests for every Aegis rule (a breach refused; a non-breach allowed); replay parity across restarts; multi-symbol invariants (no cross-symbol exposure leakage; portfolio caps bind; a partial set refusal sends nothing).
**ACCEPTANCE:** the 16 dry-run invariants re-run under the new runtime plus the multi-symbol invariants; the 48 h synthetic fixture at PARITY with a restart, an operator action and a partial-set refusal; the R3 coherence control passes on that fixture.
**KILL / STOP CONDITIONS:** none scientific.
**EVIDENCE CREATED:** ENGINEERING.
**EVIDENCE NOT CREATED:** any scientific evidence.
**PROSPECTIVE STANDING:** unchanged.
**REAL-MONEY STATUS:** none.
**DEPENDENCIES:** R1, R7.
**NEXT MAY ASSUME:** a multi-symbol single-leg dry-run runtime with one ledger, set-scoped Aegis, and clearable disputes.
**PR BOUNDARIES:** R8-a … R8-h as reviewed PRs; the runtime never links to a live venue module.

### R9 — Autonomous demo / soak (engineering evidence only)

**PURPOSE:** prove unattended operation of the exact runtime head that will run the campaign, before the irreversible boundary.
**WHY:** the boundary cannot be undone; every defect that a soak can reveal must be found first. **SOAK PRECEDES PROSPECTIVE ACTIVATION.**
**INPUT PREREQUISITES:** R8 accepted; a gen4 recorder running on the production recorder's pre-activation root (R4) or on the R2 preflight recorder if R2 records that it keeps running (never a prospective root — none exists before R6); the R3 contract merged; the `seq` and `OPERATOR` replay policies fixed.
**EXACT WORK and ACCEPTANCE — the ENGINEERING DEMO PASS gates (§26 as corrected):** (1) ≥ 14 consecutive UTC days unattended on live recorder data under a SOAK profile, the frozen candidate running in SHADOW (targets computed, positions simulated); Lane B, if elected, soaks in its own process under its own PASS/KILL record, its outcome never enters the primary lane's R9 verdict, and a Lane B soak failure defers Lane B to its own later contract and boundary; (2) ≥ 2 planned restarts and ≥ 1 unplanned `SIGKILL` mid-minute, each recovered with a `RECOVERY` record and no state-file edit; (3) ≥ 1 injected feed outage ≥ 10 min → staleness halt as designed and automatic continuation; ≥ 1 recorder-process restart; (4) kill-switch drill persisted across a restart and cleared by `resume` with a note; ≥ 1 reconciliation-dispute drill cleared through the CLI; ≥ 1 disk-low drill, accepted only if the halt reason is present after restart; (5) daily report every day, containing engineering metrics only (no equity, PnL or position-level economics); every alert rule fired at least once (synthetic) and delivered off-host; (6) replay of every soak day = PARITY; (7) zero manual state edits, zero unexplained divergence, zero halts without a record, zero minutes skipped as stale while the recorder was healthy. **Shadow-economics seal:** the candidate's simulated PnL is logged (Aegis needs it) but is not in the soak report, not an input to the R6 decision and not readable by anyone before R10 closes: the ledger's economic fields are written to a sub-root that is hashed daily and read only by the runtime and Aegis, and the sub-root is hashed at soak end as SEALED-DIAGNOSTIC. **Head record:** the exact head, contract hash and feature-function hash of the soak are recorded; R10 runs that head (37.2).
**TECHNICAL DESIGN:** the R1 daemon, R7 snapshot, R8 runtime, unchanged; drills scripted and logged as operator actions.
**SCIENTIFIC DESIGN:** none; scoring of the candidate is forbidden during the soak.
**DATA USED:** engineering-root or pre-activation recorder data.
**DATA NOT USED:** prospective data (none exists); burned blocks.
**TESTS / VERIFICATION:** the gates above; replay parity per day; the drill log.
**ACCEPTANCE:** all seven gates met; the soak report frozen with SHA256SUMS as ENGINEERING evidence; the shadow-economics seal recorded.
**KILL / STOP CONDITIONS:** an unexplained divergence not root-caused within one repair cycle; a silent position change; a halt without a record; a recovery that required editing state; > 1% of healthy-recorder minutes skipped. A failed soak is repaired and re-run in full on the new head; soak days are never spliced.
**EVIDENCE CREATED:** ENGINEERING (the soak report); SEALED-DIAGNOSTIC (shadow economics).
**EVIDENCE NOT CREATED:** alpha; any prospective standing; any input to activation other than PASS/FAIL.
**PROSPECTIVE STANDING:** unchanged.
**REAL-MONEY STATUS:** none.
**DEPENDENCIES:** R8; a running gen4 recorder.
**NEXT MAY ASSUME:** a runtime head certified for unattended operation, restart, drills and parity — the only head R10 may run; the R6 review may now proceed.
**PR BOUNDARIES:** the soak-report PR (evidence directory + SHA256SUMS); ENGINEERING-FIX PRs found by the soak are merged before the report is frozen and trigger a full re-run.

### R6 — Prospective boundary activation (irreversible; executes AFTER R9 PASS)

**PURPOSE:** start irreversible evidence accrual under a frozen contract on a soaked runtime.
**WHY:** the boundary cannot be undone; it must be the last gate before accrual, after everything repairable has been repaired and proven. Roadmap v3 activated it fourth; this roadmap activates it last.
**INPUT PREREQUISITES (all, separately recorded):** (1) the R3 contract merged ≥ 7 days earlier with its review; (2) the R4 contract hash; (3) R5 reconciliation running on the production recorder; (4) **R9 PASS** with its frozen soak report and head record; (5) the qualification record produced by R5: ≥ 30 qualifying pre-activation days on the production deployment under the pre-activation hash, the clock starting at the R4 deployment, each day counted only once R5's accepted reconciliation has been run over it, with coverage ≥ thresholds on every required stream and reconciliation agreement on every ARCHIVE_RECONCILED stream (any failure restarts the 30-day clock; no substitution); (6) production recorder source identity verified (part of that record); (7) an independent boundary review recorded in its own governance PR **after** R9 PASS, having read the soak report, the contract, the power report and the qualification record; (8) the owner's explicit activation decision, which may not reference the sealed shadow economics.
**EXACT WORK:** the activation commit sets `prospective_from` to the earliest UTC midnight after all eight inputs; the activated hash writes to a fresh storage root; no mixed hashes; the campaign log opens with the contract hash, the runtime head, and the feature-function, model, policy, cost-model, evaluator and validity-evaluator hashes; Lane B opens on the same boundary in its own process iff it is elected, its preregistration PR was merged ≥ 7 days earlier, and its own R9 record met every gate — readiness is mechanical and may not reference its sealed shadow economics; otherwise it waits for its own later contract and boundary.
**TECHNICAL DESIGN:** the gen3 activation pattern (fresh root; hash stamped in every manifest).
**SCIENTIFIC DESIGN:** none; this phase creates the boundary and no result.
**DATA USED:** none for any decision.
**DATA NOT USED:** the sealed shadow economics; any candidate outcome; burned blocks.
**TESTS / VERIFICATION:** activation-schema tests; a check that the first prospective day is recorded under the activated hash and root; the verifier refuses activation if any of the eight inputs is missing from the PR.
**ACCEPTANCE:** activation PR merged; first prospective day recorded under the activated hash.
**KILL / STOP CONDITIONS:** a coverage failure in the qualifying period restarts the clock; a failed or unrepaired soak → no activation; an owner decision not to activate after a PASSED soak → the candidate is DECLINED with the reason recorded: its engineering soak report stays ENGINEERING evidence, its sealed shadow economics are never opened, the decline consumes the lane's candidate slot (R12), and re-election requires a new R3, a fresh soak and a fresh seal.
**EVIDENCE CREATED:** the boundary (no result).
**EVIDENCE NOT CREATED:** any result; any claim.
**PROSPECTIVE STANDING:** CHANGES — everything after the boundary is prospective for the contract's campaign(s).
**REAL-MONEY STATUS:** none.
**DEPENDENCIES:** R3, R4, R5, R9 PASS, the qualification, the review.
**NEXT MAY ASSUME:** a live boundary; a head-frozen runtime; a frozen contract; a fresh root; no pre-boundary outcome ever read.
**PR BOUNDARIES:** the boundary-review record PR (governance), then one activation PR containing nothing else.

### R10 — First prospective campaign (design per §27 as corrected)

**PURPOSE:** the first answerable market question, accrued without adaptation.
**WHY:** every historical result was adaptive or underpowered; this is the first evidence that can count.
**INPUT PREREQUISITES:** R6; the R9 head; the contract.
**EXACT WORK:** accrual under the contract on the head-frozen runtime; no online tuning; no candidate replacement; no universe change; no refit outside a predeclared calendar lane. **No-peek:** engineering dashboards only (coverage, staleness, halts, parity, block validity). **At every block end, in this order:** (1) the validity evaluator is run by the CI/verifier job (never by hand) on engineering records only; the job commits the VALID/INVALID verdict, its input manifest and its hash in the block's evidence PR, opened before the readout job runs and reviewed by a person who certifies having read no readout of that block (a manually produced verdict is itself an invalidity criterion); (2) for a VALID block only, the frozen evaluator computes the block readout, logged and hashed; INVALID blocks get no readout; (3) interim decisions only through the alpha-spending rule. **Extension:** accrual continues in contiguous calendar blocks until N VALID blocks exist or the maximum extension E is exhausted; the count is mechanical. **Head:** ENGINEERING-FIX PRs only (37.2). **Lane B**, if elected, accrues in parallel in its own process under its own contract on the same boundary and is reported separately.
**TECHNICAL DESIGN:** runtime unchanged from R9; the campaign log; the R3 evaluator and validity code; per-block evidence directories.
**SCIENTIFIC DESIGN:** per the contract: information-primary cross-sectional endpoint; economic falsification screen; negative controls; Holm over m ≤ 2; deciding economics identical to R11's.
**DATA USED:** prospective data from the activated root, read only by the frozen evaluator at block ends.
**DATA NOT USED:** any pre-boundary data for a deciding purpose; burned blocks; the outcomes of INVALID blocks (never computed).
**TESTS / VERIFICATION:** daily replay parity; the validity evaluator's determinism (re-run offline gives the same verdict); the per-block hash chain; the verifier's check that no readout precedes its validity verdict; the coherence control re-run per block.
**ACCEPTANCE (campaign end):** N VALID blocks → the preregistered PASS / NEGATIVE statements applied by the frozen evaluator (information endpoint and economic screen); or E exhausted → **NOT EVALUABLE / ENGINEERING FAILURE**.
**KILL / STOP CONDITIONS:** *engineering, mechanical:* a block INVALID → excluded from scoring, campaign extends per rule; a head change outside ENGINEERING-FIX that touches no contract-hashed module → that block INVALID and a full re-soak before the next block is scored; a change to a contract-hashed module → campaign ABORTED (the object changed; any successor is route C); an owner stop → campaign ABORTED (37.1): the reason recorded, the candidate's budget slot consumed, no same-design rerun, all block readouts left sealed, no partial claim. ABORTED is never recorded as NOT EVALUABLE, which is reserved for exhaustion of E. *Scientific:* none mid-campaign except the alpha-spending rule.
**EVIDENCE CREATED:** PROSPECTIVE (PASS or NEGATIVE), a NOT EVALUABLE engineering-failure record, or an ABORTED record; the full VALID/INVALID chronology in every case.
**EVIDENCE NOT CREATED:** an alpha claim; real-money eligibility; confirmatory standing.
**PROSPECTIVE STANDING:** consumes the boundary for this candidate at the first scored day.
**REAL-MONEY STATUS:** none.
**DEPENDENCIES:** R6.
**NEXT MAY ASSUME:** an immutable campaign outcome with a complete block chronology. PASS on both floors → R11 with the identical object. Information PASS + screen FALSIFIED → information-only result, no promotion, R12. NEGATIVE → R12 (family retired). NOT EVALUABLE → R12 (engineering cause fixed, re-soak, the pre-committed automatic same-design successor on a new boundary if the slot allows; no pooling; readouts sealed until it closes). ABORTED → R12 (slot consumed).
**PR BOUNDARIES:** one evidence PR per block (validity verdict, then readout, each with SHA256SUMS); the closure PR; ENGINEERING-FIX PRs.

### R11 — Economic confirmation and continuation (same object as R10)

**PURPOSE:** confirm, not discover: replicate the R10 result on new data under an unchanged success definition.
**WHY:** one prospective PASS is one observation; confirmation adds data and must not redefine success.
**INPUT PREREQUISITES:** R10 PASS on both floors; an independent audit reconstructing the R10 verdict from the decision log and recorder files; the continuation size N′ and the non-inferiority rule from the R3 contract (N′ fixed there, or produced by a sizing rule preregistered there as deterministic in the measured R10 effect).
**EXACT WORK:** the same frozen candidate, on the same head-frozen runtime (ENGINEERING-FIX only), under the **identical deciding economic semantics** (fee, spread, executable price, latency, slippage envelope, funding, turnover, reject/partial rule, impact rule), for ≥ N′ VALID blocks (proposal: N′ ≥ ⌈N/2⌉ and ≥ 3) with the same validity/extension mechanics. PASS = the continuation does not contradict the R10 result under the preregistered non-inferiority rule. New execution, testnet or live-operational information enters only: **(A)** through an update rule preregistered in R3 as deterministic, applied exactly and logged; **(B)** as non-deciding stress/diagnostic analyses reported separately and labelled (measured slippage distributions, testnet reject/partial rates, hypothetical impact at larger sizes, mid-referenced slippage in bps, the opened SEALED-DIAGNOSTIC shadow-vs-live comparison from R9); or **(C)** by a new preregistration and a new campaign, which R11 is not. The confirmatory period may add data; it may not redefine success.
**TECHNICAL DESIGN:** runtime and evaluator unchanged; a separate non-deciding stress module that cannot write to the deciding readout.
**SCIENTIFIC DESIGN:** replication of one preregistered object; no new hypothesis; m unchanged.
**DATA USED:** prospective data after R10's close.
**DATA NOT USED:** R10's data for re-scoring; pre-boundary data; burned blocks.
**TESTS / VERIFICATION:** as R10; a test that the evaluator, cost-model and validity-evaluator hashes equal R10's.
**ACCEPTANCE:** CONFIRMATORY PASS by the frozen evaluator; independent audit reconstructing it.
**KILL / STOP CONDITIONS:** contradiction → CONFIRMATION FAILED, immutable, eligibility withdrawn, R12; NOT EVALUABLE as in R10; any change to a contract-hashed module → the continuation ends as ABORTED (route C for any successor), without a confirmatory result.
**EVIDENCE CREATED:** CONFIRMATORY (or an immutable confirmation-failure record); DIAGNOSTIC stress reports.
**EVIDENCE NOT CREATED:** real-money authority; any claim beyond eligibility for consideration.
**PROSPECTIVE STANDING:** **Gate 1** — eligibility for real-money *consideration*, nothing more.
**REAL-MONEY STATUS:** none.
**DEPENDENCIES:** R10 PASS.
**NEXT MAY ASSUME:** a candidate with CONFIRMATORY standing; a realised-cost diagnostic record from which Gate 4 numbers may be derived; any mismatch between the frozen deciding model and observed execution is a (C) trigger, never a silent revision.
**PR BOUNDARIES:** continuation block PRs; closure PR; the stress-analysis report PR, labelled non-deciding.

### R12 — Strategy expansion governance

**PURPOSE:** decide what follows any campaign outcome without adaptive drift.
**WHY:** an adaptive sequence of campaigns can masquerade as one programme.
**INPUT PREREQUISITES:** any R10 or R11 outcome record, or a DECLINED record from R6.
**EXACT WORK:** apply the rules: (1) NEGATIVE retires the candidate family for the same (target, horizon, universe), recorded with its full disclosure block; (2) NOT EVALUABLE retires nothing scientifically — the engineering cause is fixed by reviewed PRs, the runtime re-soaked (R9), and the pre-committed successor rule of R3 applies automatically: at most one same-design successor (new hash, a new boundary with all eight R6 inputs again, no pooling), never decided on outcomes; the predecessor's VALID-block readouts stay SEALED-DIAGNOSTIC until the successor closes or the candidate is retired, and are then disclosed as ADAPTIVE context, never as a design input; a second NOT EVALUABLE is terminal for the design and consumes the slot; (2b) ABORTED and DECLINED consume the candidate's slot and permit no same-design rerun; (3) a new candidate changes ≥ 1 design axis a priori and re-enters at R3 (power report, review, ≥ 7-day gap), then must pass R7 (feature-function hash), R9 (re-soak on the new head) and the full R6 input list before its boundary; it may read a negative campaign's descriptive statistics (turnover, cost realisation, coverage) but not its per-decision outcomes; (4) budget: ≤ 2 directional candidates and ≤ 1 carry candidate before a mandatory stop-and-decide, each candidate holding one R10, at most one automatic NOT-EVALUABLE re-run and one R11; after two directional negatives, record "no deployable directional alpha under the current mandate"; (5) engineering evidence, DIAGNOSTIC statistics and cost tables are reusable; PROSPECTIVE and CONFIRMATORY evidence belongs to the candidate that earned it; (6) orchestration (R13) opens only with ≥ 2 families holding CONFIRMATORY standing.
**TECHNICAL DESIGN:** governance documents; the front-door verifier counts candidates against the budget.
**SCIENTIFIC DESIGN:** governance only.
**DATA USED:** campaign outcome records (descriptive).
**DATA NOT USED:** per-decision outcomes for design; burned blocks.
**TESTS / VERIFICATION:** verifier; CI.
**ACCEPTANCE:** a decision PR naming the outcome record's SHA and the next permitted action.
**KILL / STOP CONDITIONS:** budget exhausted → stop-and-decide review; no further campaign without a new mandate and budget.
**EVIDENCE CREATED:** none.
**EVIDENCE NOT CREATED:** any.
**PROSPECTIVE STANDING:** unchanged.
**REAL-MONEY STATUS:** none.
**DEPENDENCIES:** R10 / R11.
**NEXT MAY ASSUME:** a recorded next step inside the budget.
**PR BOUNDARIES:** one governance PR per decision.

### R13 — Portfolio / orchestration (conditional)

**PURPOSE:** allocate across lanes only once two families hold CONFIRMATORY standing.
**WHY:** premature allocation logic is a design degree of freedom and an operational burden; nothing today needs it.
**INPUT PREREQUISITES:** R12's decision; ≥ 2 CONFIRMATORY families (the second arrives from Lane B or through the R12 → R3 re-entry loop).
**EXACT WORK:** capital allocation across lanes; netting of opposing intents; gross/net caps; per-symbol concentration; drawdown budgets per lane and in total; correlation-aware exposure with a historical-simulation floor only (no VaR model beyond it); one Aegis engine over the lanes that share an account (as §7 and §29 state), one account ledger, one evidence trail; separate dry-run lanes on separate accounts keep separate engines.
**TECHNICAL DESIGN:** an extension of R8's set scope; no message broker, database server or orchestration platform.
**SCIENTIFIC DESIGN:** the allocation rules are a governance contract, not a hypothesis; they are hashed before use.
**DATA USED:** CONFIRMATORY campaign records (descriptive).
**DATA NOT USED:** any pre-boundary outcome; burned blocks.
**TESTS / VERIFICATION:** two-sided cap tests; replay parity under orchestration.
**ACCEPTANCE:** a ≥ 14-day dry-run soak under orchestration with PARITY every day.
**KILL / STOP CONDITIONS:** fewer than two CONFIRMATORY families → not opened.
**EVIDENCE CREATED:** ENGINEERING.
**EVIDENCE NOT CREATED:** alpha.
**PROSPECTIVE STANDING:** unchanged.
**REAL-MONEY STATUS:** none.
**DEPENDENCIES:** R12.
**NEXT MAY ASSUME:** a validated allocation layer, if and only if opened.
**PR BOUNDARIES:** allocation-layer PRs; the orchestration soak-report PR.

### R14 — Live-readiness qualification (Gates 2 and 3; Gate-4 contract text)

**PURPOSE:** make the execution and operational conditions of the real-money boundary true and provable.
**WHY:** nothing in the repository has ever been reconciled against a real exchange position; the only live-capable path is retired code [RF].
**INPUT PREREQUISITES:** R11 Gate 1 for the candidate that will trade (the adapter is built in this phase, after Gate 1; testnet credentials and, for the read-only reconciliation option, a read-only key with no trading permission are the only credentials permitted before R15; no trade-capable key exists before R15's token); R9 PASS on the campaign head.
**EXACT WORK:** *Gate 2 — execution:* an authenticated exchange adapter as a separate package with its own AST guard set; signed REST + user-data stream; idempotent client order ids; unknown-state resolution by order query before any retry; partial fills across restarts; cancels; reduce-only semantics; rate-limit backoff; `listenKey` keepalive and expiry; maintenance-window handling; websocket-loss and REST-ambiguity drills; reconciliation every minute with HALT on disagreement; ≥ 30 days on testnet or read-only live reconciliation; zero unresolved disputes; drill log frozen. *Gate 3 — operational:* dedicated sub-account; the trade-only key specification (withdrawals disabled, IP allow-list, daily permission re-verification) drilled on testnet — the live trade-only key is created in R15 and these items are re-verified within 7 days of go-live; secrets outside the repository and working directory; separate Unix users and state roots for recorder, runner, adapter; SHA-pinned CI, hash-pinned dependencies, digest-pinned images; backups restored in a drill; off-host dead-man switch; chrony; documented host baseline and firewall; audit log of every operator action; an out-of-process exchange-side "cancel all and close" procedure drilled on testnet. *Gate 4 text:* the capital-governance contract drafted with numbers derived from R9–R11 measurements (starting capital the owner can lose entirely; gross exposure ≤ 1× capital; leverage 1× in config and on the exchange account; max order notional; daily/cumulative loss and drawdown halts with flatten and key rotation; per-symbol cap; funding-cost cap; overnight policy; shutdown triggers; two-person enable; rollback path). *Deployable-head re-soak:* the R9 gates re-run in full (dry-run, ≥ 14 days, drills, PARITY every day) on the exact deployable head — the runtime plus the adapter package in testnet mode — frozen as ENGINEERING evidence; this report is the artifact real-money condition (2) names. *Diagnostic:* a non-deciding comparison of testnet fills against the frozen deciding cost model, which may trigger a (C) new preregistration but never alters R11.
**TECHNICAL DESIGN:** the adapter in its own package and Unix user; the runtime unchanged; dry-run and live share the same order semantics.
**SCIENTIFIC DESIGN:** none.
**DATA USED:** testnet; read-only live account data.
**DATA NOT USED:** campaign outcomes for design; burned blocks.
**TESTS / VERIFICATION:** the Gate 2 drill list, each drill logged; the Gate 3 checklist; the crash harness against the adapter's state machine.
**ACCEPTANCE:** the deployable-head re-soak PASS; Gate 2 drill log frozen as ENGINEERING evidence; Gate 3 checklist PASS, re-verified within 7 days of any go-live; Gate 4 text ready to sign.
**KILL / STOP CONDITIONS:** any unresolved dispute at the end of the drill period → the 30-day period repeats; any Gate 3 item failing at re-verification → no go-live.
**EVIDENCE CREATED:** ENGINEERING / OPERATIONAL.
**EVIDENCE NOT CREATED:** scientific evidence; authority.
**PROSPECTIVE STANDING:** unchanged.
**REAL-MONEY STATUS:** none.
**DEPENDENCIES:** R11 (Gate 1); R9 (its gates re-run here on the deployable head).
**NEXT MAY ASSUME:** Gates 2 and 3 PASS on the exact head to be deployed; a Gate-4 contract awaiting signature.
**PR BOUNDARIES:** adapter-package PRs; the drill-log evidence PR; the operations-checklist PR; the Gate-4 contract PR (draft until signed).

### ================= REAL-MONEY AUTHORITY BOUNDARY =================

No phase above may place a real order. Crossing requires **ALL FOUR**, separately recorded, each mapped to a §30 gate:
1. **SCIENTIFIC ELIGIBILITY = Gate 1:** R11 CONFIRMATORY PASS under the same object and the same deciding economics as R10, with an independent audit reconstructing the verdict.
2. **ENGINEERING ELIGIBILITY = R9 PASS + R14 re-soak + Gate 2:** R9 PASS on the campaign head, the R14 re-soak PASS on the exact deployable head, and the execution drills PASS on that head.
3. **OPERATIONAL ELIGIBILITY = Gate 3:** the operational checklist PASS, re-verified within 7 days of go-live.
4. **OWNER AUTHORISATION = Gate 4:** the signed live-authorisation contract naming capital cap, leverage 1×, symbols, order types, abort rules, the two-person enable procedure and the rollback path.

A profitable backtest, demo, soak or canary never satisfies (1). Engineering or demo success is never alpha. No historical positive result is prospective. Nothing in this roadmap, in the audit or in the corrigendum creates real-money authority.

### R15 — Owner authorisation (Gate 4 executed)

**PURPOSE:** record the owner's bounded, revocable authority to run the canary.
**WHY:** authority must be explicit, written, and one of four gates, never the only one.
**INPUT PREREQUISITES:** conditions (1)–(3) recorded as PASS; the Gate-4 contract text from R14.
**EXACT WORK:** a governance PR recording the signed contract; the acknowledgement token is created only after the PR merges; the trade-capable API key is created only after the token exists; both are logged.
**TECHNICAL DESIGN:** documents; the runtime's live gate reads the token and the contract hash.
**SCIENTIFIC DESIGN:** none.
**DATA USED:** none.
**DATA NOT USED:** any.
**TESTS / VERIFICATION:** the verifier checks that the four condition records exist and are current before the live gate can open.
**ACCEPTANCE:** PR merged; token and key existence logged.
**KILL / STOP CONDITIONS:** any of conditions (1)–(3) lapsed or contradicted → no authorisation; authority is revocable at any time by removing the token.
**EVIDENCE CREATED:** none.
**EVIDENCE NOT CREATED:** any.
**PROSPECTIVE STANDING:** unchanged.
**REAL-MONEY STATUS:** authority created for the canary only, bounded by the signed contract.
**DEPENDENCIES:** R11, R14.
**NEXT MAY ASSUME:** a bounded authority for R16 and nothing beyond it.
**PR BOUNDARIES:** one governance PR.

### R16 — Canary (Gate 5)

**PURPOSE:** prove that real execution matches the dry-run model at a size that cannot matter.
**WHY:** the first real order is irreversible; it must be small, parallel-shadowed and reconciled daily.
**INPUT PREREQUISITES:** R15.
**EXACT WORK:** ≥ 30 days; ≤ 10% of the authorised cap and above minNotional × the largest campaign order; 1×; the campaign's symbols only; MARKET + reduce-only semantics identical to dry-run; a parallel dry-run for parity; daily reconciliation of fills, fees and funding against the account statement; abort on any reconciliation dispute, unexplained divergence, daily-loss breach, key-permission change or parity failure; escalation = stop and audit, never resize.
**TECHNICAL DESIGN:** the R14 adapter; the R8 runtime; the parallel dry-run process.
**SCIENTIFIC DESIGN:** none; a profitable canary is operational evidence only.
**DATA USED:** live account data; the campaign's recorder data.
**DATA NOT USED:** any outcome for candidate design.
**TESTS / VERIFICATION:** daily reconciliation report; daily parity report.
**ACCEPTANCE:** 30 days with zero aborts, zero unreconciled items, parity every day; the canary report frozen as OPERATIONAL evidence.
**KILL / STOP CONDITIONS:** as listed; any abort ends the canary; a repeat needs a new R15 authorisation.
**EVIDENCE CREATED:** OPERATIONAL.
**EVIDENCE NOT CREATED:** scientific evidence; scale authority.
**PROSPECTIVE STANDING:** unchanged.
**REAL-MONEY STATUS:** real orders at canary size under the contract.
**DEPENDENCIES:** R15.
**NEXT MAY ASSUME:** execution fidelity at canary size, nothing about larger sizes.
**PR BOUNDARIES:** the canary evidence PR.

### R17 — Controlled real-money operation

**PURPOSE:** operate at the authorised cap under the same rules.
**WHY:** the cap, not the result, is the authority.
**INPUT PREREQUISITES:** R16 accepted.
**EXACT WORK:** continue at the authorised cap; monthly frozen reports; monthly parity; daily reconciliation; any change to the candidate is a new R3 followed by a dry-run campaign — money never runs an unconfirmed candidate.
**TECHNICAL DESIGN:** unchanged.
**SCIENTIFIC DESIGN:** none.
**DATA USED:** live account data.
**DATA NOT USED:** live outcomes for candidate design.
**TESTS / VERIFICATION:** monthly reports and parity.
**ACCEPTANCE:** ongoing; each monthly report frozen.
**KILL / STOP CONDITIONS:** the contract's halt triggers (daily/cumulative loss, drawdown, dispute, parity breach, key change); halt + flatten + key rotation.
**EVIDENCE CREATED:** OPERATIONAL.
**EVIDENCE NOT CREATED:** scientific evidence; scale authority.
**PROSPECTIVE STANDING:** unchanged.
**REAL-MONEY STATUS:** bounded by the contract.
**DEPENDENCIES:** R16.
**NEXT MAY ASSUME:** a realised-cost and parity record for R18.
**PR BOUNDARIES:** monthly evidence PRs.

### R18 — Scale-up / scale-down governance (Gate 6)

**PURPOSE:** change size only by decision, never by momentum.
**WHY:** leverage and size creep after a profitable period is the classic failure.
**INPUT PREREQUISITES:** R17 with ≥ 60 days of frozen reports since the last step.
**EXACT WORK:** increase only by a new decision with its own evidence period and audit; ≤ 2× per step; ≥ 60 days between steps; only after realised costs and parity match the frozen deciding model (a mismatch is a (C) trigger: new preregistration before any scale-up); automatic decrease or halt on drawdown, parity breach or reconciliation dispute; no profit-based acceleration; leverage stays 1× throughout this roadmap.
**TECHNICAL DESIGN:** unchanged.
**SCIENTIFIC DESIGN:** none.
**DATA USED:** frozen monthly reports.
**DATA NOT USED:** live outcomes for candidate design.
**TESTS / VERIFICATION:** the audit per step.
**ACCEPTANCE:** each step recorded in a governance PR with its evidence period.
**KILL / STOP CONDITIONS:** automatic decrease/halt as listed.
**EVIDENCE CREATED:** OPERATIONAL.
**EVIDENCE NOT CREATED:** scientific evidence.
**PROSPECTIVE STANDING:** unchanged.
**REAL-MONEY STATUS:** bounded by the current step's contract amendment.
**DEPENDENCIES:** R17.
**NEXT MAY ASSUME:** nothing beyond the current step.
**PR BOUNDARIES:** one governance PR per step.

---
## Part D — Required non-§37 edits (exact replacements)

Each entry gives the passage as it stands in the audit (ORIGINAL, verbatim) and the text that replaces it (REPLACEMENT). Only these passages change outside §37; every other sentence of the audit stands. The corrected full report applies exactly these replacements plus the Part C substitution.

### D1 — §1 finding 7  (corrigendum A8)

**ORIGINAL:**

> The causal as-of machinery is sound and is kept; the hierarchy is simplified to at most two closed-bar slow-scale variables inside the frozen feature set (§16).

**REPLACEMENT:**

> The causal as-of machinery is sound and is kept; the hierarchy is simplified to at most two closed-bar slow-scale variables inside the frozen feature set (§16). The runtime must nevertheless be multi-timeframe *capable*: a generic multi-clock causal snapshot is a required engineering property (R7), while hierarchical MTC as a scientific model is not required for campaign 1 and may return only under a new preregistered campaign.

### D2 — §1 finding 8  (corrigendum A9)

**ORIGINAL:**

> The carry mechanism, demoted by owner decision without refutation, is the only mechanism whose *economic* evidence is reachable in a six-month window and is recommended as a parallel, separately preregistered, low-cost lane on the runtime that already exists (§27).

**REPLACEMENT:**

> The primary lane is multi-symbol, single-leg, directional LONG/SHORT on USD-M perpetuals; spot is reference data, a hedge leg where a lane needs one, and support infrastructure. The carry mechanism, demoted by owner decision without refutation, is the only mechanism whose *economic* evidence is reachable in a six-month window and is recommended as a SECONDARY, OPTIONAL, separately preregistered lane on the runtime that already exists (§27): it never delays R1–R9, has its own hypothesis slot, is not a prerequisite for anything, is not orchestration, receives no priority for arriving sooner, and yields on any resource conflict.

### D3 — §13 fitness paragraph  (corrigendum A6, A7)

**ORIGINAL:**

> a no-peek rule during accrual; an interim-look policy; the engineering-invalidation versus scientific-failure split; no retroactive substitution of blocks;

**REPLACEMENT:**

> a no-peek rule during accrual; an interim-look policy; a mechanical, outcome-blind block-validity rule with a frozen target of VALID blocks, a frozen extension cap and a NOT EVALUABLE outcome (the engineering-invalidation versus scientific-failure split made non-adaptive); no retroactive substitution of blocks; deciding economic semantics frozen before the boundary and identical in confirmation;

### D4 — §14 item 3  (corrigendum A9, A16)

**ORIGINAL:**

> 3. *Carry lane (recommended, optional):* a separately preregistered, always-on or funding-gated delta-hedged carry rule on BTCUSDT (and possibly ETHUSDT) in dry-run on the existing two-leg runtime, as its own process, state directory and dry-run capital. It is the only candidate whose *economic* evidence is reachable inside the first accrual window (≈ 546 settlements per symbol in 182 days). It is not orchestration; it is a second independent dry-run campaign. If the owner declines it, nothing else in the roadmap changes.

**REPLACEMENT:**

> 3. *Carry lane (SECONDARY, OPTIONAL):* a separately preregistered, always-on or funding-gated delta-hedged carry rule on BTCUSDT (and possibly ETHUSDT) in dry-run on the existing two-leg runtime, as its own process, state directory, Aegis instance, ledger and dry-run capital. It is the only candidate whose *economic* evidence is reachable inside the first accrual window (≈ 546 settlements per symbol in 182 days). It is not orchestration; it is a second independent dry-run campaign with its own single hypothesis slot. It must not delay R1–R9, is not a prerequisite for the primary lane, does not replace futures-first and receives no priority because its evidence may arrive sooner; if resources conflict, the primary directional futures lane wins. Election is an owner decision at R0 with a deadline at R4 (its spot streams must be in the gen4 contract). If the owner declines it, nothing else in the roadmap changes.

### D5 — §14 item 4  (corrigendum A9)

**ORIGINAL:**

> Running carry *instead* would defer that build; running it *alongside* costs little because its runtime exists.

**REPLACEMENT:**

> Running carry *instead* would defer that build; running it *alongside* costs little because its runtime exists, and it is permitted only while it costs the primary lane nothing material.

### D6 — §14 coherence rule  (corrigendum A1)

**ORIGINAL:**

> The preregistration states which, and the runtime that will execute it must have passed its soak (R9) before the boundary is activated.

**REPLACEMENT:**

> The preregistration states which; the runtime that will execute it must have passed its soak (R9) before the boundary is activated, and R9 PASS is a recorded input to R6.

### D7 — §16 decision paragraph  (corrigendum A8)

**ORIGINAL:**

> **Decision: SIMPLIFY and DEFER as mandatory architecture; RETAIN the causal machinery. [REC]** The first campaign's frozen feature set may include at most two slow-scale, closed-bar, explicitly aged state variables (1 d realised volatility; 1 d or 4 h trend sign). No hierarchy, no coherence state, no regime labels. V3-7 is re-scoped to a thin runtime snapshot type (`source_time`, `available_time`, `as_of`, `age`, `staleness`, `complete`, `missing` per input) that the single-leg runtime needs anyway, tested with two-sided synthetic leak controls (a planted future leak must be detected; a clean series must pass; the `nn/mtf.py` shift control is the template). Hierarchical MTC returns only as a preregistered second campaign arm if the first shows any prospective signal. v3 §21's "causal Multi-Timeframe Coherence remains mandatory" should be revised to "causal as-of semantics remain mandatory".

**REPLACEMENT:**

> **Decision: SIMPLIFY and DEFER hierarchical MTC as a scientific model; RETAIN the causal machinery; REQUIRE multi-timeframe capability in the runtime. [REC]** MULTI-TIMEFRAME CAPABILITY = REQUIRED ENGINEERING PROPERTY. HIERARCHICAL MTC AS A SCIENTIFIC MODEL = NOT REQUIRED FOR CAMPAIGN 1. V3-7 is re-scoped to a generic multi-clock causal `MarketSnapshot` (R7): for every input and every contract-declared timeframe it carries `source_time`, `available_time`, `as_of`, `age`, `staleness`, `complete`, `missing`, and it derives multiple closed-bar clocks causally from the base 1 m stream, so the runtime is not architecturally limited to one timeframe; it is tested with two-sided synthetic leak controls (a planted future leak must be detected; a clean series must pass; the `nn/mtf.py` shift control is the template). The first campaign's deciding feature set is restricted to the preregistered decision clock plus at most two preregistered slow-scale, closed-bar, explicitly aged state variables (1 d realised volatility; 1 d or 4 h trend sign); every other available clock is capability only and creates no scientific degree of freedom during R10. No hierarchy, no coherence state, no regime labels, no voting, no "HTF always wins", no mandatory multi-layer narrative state. A richer hierarchy returns only under a new preregistered campaign after the first prospective result warrants it. v3 §21's "causal Multi-Timeframe Coherence remains mandatory" should be revised to "causal as-of semantics and multi-clock capability remain mandatory; hierarchical coherence does not".

### D8 — §22 frozen outputs  (corrigendum A6, A7)

**ORIGINAL:**

> expected calendar duration to the effect floor; cost ±50% sensitivity. The block-length *rule* and the endpoint estimator are frozen, not the numbers;

**REPLACEMENT:**

> expected calendar duration to the effect floor; cost ±50% sensitivity; the target number N of VALID scored blocks, the block length, the maximum extension E and the R11 continuation size N′ (or its deterministic sizing rule). The block-length *rule* and the endpoint estimator are frozen, not the numbers;

### D9 — §23 fidelity table and freeze rule  (corrigendum A6)

**ORIGINAL:**

> **Required cost fidelity by stage [REC].**
> 
> | stage | fidelity | how obtained |
> |---|---|---|
> | information campaign | measured median spread and taker fee per symbol; funding inside the horizon; label on executable quotes; no fill model beyond the crossing rule | gen4 preflight bookTicker/aggTrade statistics (≥ 30 d), frozen |
> | economic screen / shadow economics | recorded-quote fills at the touch + measured slippage distribution (p50/p90) + fee schedule + funding by actual settlement + turnover | recorder + fill model calibrated on aggTrade trade-through |
> | autonomous demo | as above; correctness, not realism, is the claim | same |
> | prospective economic campaign (R11) | add size-dependent impact where order size > 1% of top-5 depth; reject/partial-fill rates from a testnet adapter; base-asset spot fee for any spot leg; mid-referenced slippage in bps | Tier B depth for BTC/ETH; testnet |
> | real money | actual fills, fees, funding from the account statement reconciled daily against the model; parity report | authenticated adapter; canary |

**REPLACEMENT:**

> **Required cost fidelity by stage [REC; decision-relevant semantics frozen in R3 and identical in R10 and R11].**
> 
> | stage | fidelity | how obtained | deciding? |
> |---|---|---|---|
> | information endpoint (R10) | measured median half-spread and taker fee per symbol; funding inside the horizon; label on executable quotes; no fill model beyond the frozen crossing rule | gen4 preflight bookTicker/aggTrade statistics (≥ 30 d), frozen as code in R3 | yes |
> | economic screen (R10) and confirmation (R11) | the frozen deciding model: fee tier and BNB assumption; touch-crossing executable price at t + Δ with Δ frozen; conservative per-symbol slippage envelope (p90 trade-through for the reference size); funding by settlement per symbol interval; turnover on every target change; frozen reject/partial-fill rule; notional capped below the impact threshold | R2 measurements frozen as hashed code in R3; the same bytes in runtime and evaluator | yes — identical in R10 and R11 |
> | autonomous demo (R9) | the same model; correctness, not realism, is the claim; shadow economics sealed | same | no (engineering only) |
> | stress / diagnostic analyses (R11, R14) | measured slippage distributions; testnet reject/partial rates; hypothetical impact at larger sizes; mid-referenced slippage in bps; spot base-asset fee for any spot leg | testnet adapter; Tier B depth for BTC/ETH | never — reported separately; may trigger a new preregistration (C) |
> | real money (R16+) | actual fills, fees, funding from the account statement reconciled daily against the frozen model; parity report | authenticated adapter; canary | operational only |
> 
> **Freeze rule [REC].** Every economic semantic that affects the endpoint — fee assumptions, spread rule, executable-price rule, slippage estimator or conservative envelope, funding treatment, turnover accounting, reject/partial-fill treatment, latency assumption, impact rule and the notional cap that keeps impact out of scope — is frozen in R3 as hashed code before the boundary. R10 and R11 test the same scientific/economic object. Newly observed execution, testnet or live-operational information may enter R11 only (A) through a deterministic update rule preregistered before R10, (B) as a non-deciding stress/diagnostic analysis, or (C) by opening a new preregistration and campaign. The confirmatory period adds data; it never redefines success.

### D10 — §25 causal feature state row  (corrigendum A8)

**ORIGINAL:**

> | Causal feature state | `MarketSnapshot` with per-input `source_time`, `available_time`, `as_of`, `age`, `staleness`, `complete`, `missing`; slow-scale variables from closed bars only | runner | R7 |

**REPLACEMENT:**

> | Causal feature state | `MarketSnapshot` generic over contract-declared causal clocks: per input and per timeframe `source_time`, `available_time`, `as_of`, `age`, `staleness`, `complete`, `missing`; closed-bar clocks derived causally from the base 1 m stream (multi-timeframe capability is a required engineering property); the deciding subset fixed by a hashed allow-list (decision clock + ≤ 2 slow variables in campaign 1) | runner | R7 |

### D11 — §26 definition (data source)  (corrigendum A1, A4)

**ORIGINAL:**

> Data source: the gen4 recorder (engineering root or prospective root — the demo computes nothing that selects anything).

**REPLACEMENT:**

> Data source: the gen4 recorder's engineering root or the production recorder's pre-activation root; never a prospective root, because the soak precedes activation (R9 before R6) and no prospective data exists while it runs.

### D12 — §26 gate 1  (corrigendum A4)

**ORIGINAL:**

> (the rule may be a deterministic dummy or the frozen candidate in shadow; scoring is forbidden until R10).

**REPLACEMENT:**

> (the frozen candidate runs in SHADOW; scoring is forbidden until R10's first scored day; the candidate's shadow economics are sealed at soak end as SEALED-DIAGNOSTIC and are not an input to the activation decision).

### D13 — §26 separation  (corrigendum A1, A4, A5)

**ORIGINAL:**

> **Separation:** the soak report is frozen as ENGINEERING evidence; its PnL is descriptive and may not select anything; a passed soak changes no prospective standing.

**REPLACEMENT:**

> **Separation:** the soak report is frozen as ENGINEERING evidence and contains engineering metrics only; the candidate's shadow PnL is logged for Aegis, sealed at soak end and opened only after R10 closes as a shadow-vs-live diagnostic; it may not select anything and may not inform the R6 decision; a passed soak changes no prospective standing; the head that passed the soak is the head R10 runs (head-freeze rule, §37.2). **Order:** R9 PASS is a recorded prerequisite of R6 — SOAK PRECEDES PROSPECTIVE ACTIVATION.

### D14 — §27 cost-model bullet  (corrigendum A6)

**ORIGINAL:**

> - *Cost model:* per-symbol measured taker fee tier, median and p90 half-spread at decision instants, slippage under the frozen crossing rule, funding per hold; frozen from preflight; ±50% sensitivity reported.

**REPLACEMENT:**

> - *Cost model (deciding; identical in R10 and R11):* per-symbol fee tier and BNB assumption; touch-crossing executable price at t + Δ with Δ frozen; conservative per-symbol slippage envelope (p90 trade-through for the reference size); funding by settlement per symbol interval; turnover on every target change; frozen reject/partial-fill rule; notional capped below the impact threshold; all frozen from preflight as hashed code in R3; ±50% sensitivity reported, never deciding; changes only by (A) a preregistered deterministic update rule, (B) non-deciding stress analysis, or (C) a new preregistration (§23).

### D15 — §27 no-peek bullet  (corrigendum A7, A17)

**ORIGINAL:**

> - *No-peek policy:* engineering dashboards only (coverage, staleness, halts, parity); economic and information readouts only at block ends by the frozen evaluator, logged and hashed.

**REPLACEMENT:**

> - *No-peek policy:* engineering dashboards only (coverage, staleness, halts, parity, block validity); at each block end the validity verdict is computed from engineering records only, committed and hashed first; economic and information readouts are computed only for VALID blocks, by the frozen evaluator, logged and hashed; an INVALID block never receives a readout.

### D16 — §27 invalidation bullet  (corrigendum A7)

**ORIGINAL:**

> - *Engineering invalidation (not a result):* coverage below threshold, unexplained divergence, halt not recovered, delisting → the block is excluded and the campaign extends by one block; never substituted.

**REPLACEMENT:**

> - *Engineering invalidation (not a result; mechanical, non-adaptive):* the contract freezes the invalidity criteria (required-stream coverage below threshold on more than the frozen number of days; replay-parity failure; a halt not recovered within the frozen window or without a record; recorder identity or contract-hash change; a runtime head change outside an ENGINEERING-FIX PR; any manual state edit; delisting/halt of a symbol → that symbol's block, or the whole block if the minimum symbol count fails), the evaluator (`nn/prospective/validity.py`, engineering records only, no prices/scores/positions/PnL), the target number N of VALID blocks, the deterministic calendar-extension rule (contiguous blocks until N VALID), and the maximum extension E (proposal ⌈N/2⌉). An INVALID block stays in chronology, labelled, reported, is never a negative result and is never replaced. If E is exhausted without N VALID blocks the campaign is NOT EVALUABLE / ENGINEERING FAILURE — neither PASS nor scientific FAIL; any successor is a new campaign on a new boundary with no pooling.

### D17 — §27 outcomes bullet  (corrigendum A6, A7)

**ORIGINAL:**

> information PASS + NOT FALSIFIED → R11 confirmation sized from the measured effect.

**REPLACEMENT:**

> information PASS + NOT FALSIFIED → R11 confirmation of the identical object (same candidate, same deciding economics, same evaluator hash), with N′ VALID blocks fixed in the preregistration or by a sizing rule preregistered as deterministic in the measured effect; E exhausted → NOT EVALUABLE / ENGINEERING FAILURE, R12.

### D18 — §27 Lane B header  (corrigendum A9, A16)

**ORIGINAL:**

> **Lane B — carry economic campaign (parallel, optional, recommended).**

**REPLACEMENT:**

> **Lane B — carry economic campaign (SECONDARY, OPTIONAL; parallel; never on the critical path).** Elected by the owner at R0 with a deadline at R4 (its spot streams must be in the gen4 contract); own preregistration hash and own hypothesis slot; own process, state root, Aegis instance and ledger; runs on the same R6 boundary if ready, otherwise waits for its own later contract and boundary; never delays R1–R9 or R6; yields on any resource conflict.

### D19 — §28 rule 1  (corrigendum A7, A20, A21)

**ORIGINAL:**

> 1. A negative campaign retires the candidate family for the same (target, horizon, universe); it is recorded as an immutable negative with its full disclosure block.

**REPLACEMENT:**

> 1. A negative campaign retires the candidate family for the same (target, horizon, universe); it is recorded as an immutable negative with its full disclosure block. A NOT EVALUABLE campaign (engineering failure) retires nothing scientifically: the cause is fixed, the runtime re-soaked, and the successor rule pre-committed in the preregistration applies automatically (at most one same-design successor on a new boundary, never decided on outcomes, no pooling; the predecessor's VALID-block readouts stay sealed until the successor closes; a second exhaustion is terminal). An ABORTED campaign (owner stop, or a change to a contract-hashed module during accrual) consumes the candidate's slot and permits no same-design rerun.

### D20 — §30 Gate 1  (corrigendum A6)

**ORIGINAL:**

> - **Gate 1 — scientific eligibility.** R11 CONFIRMATORY PASS for the candidate that will trade;

**REPLACEMENT:**

> - **Gate 1 — scientific eligibility.** R11 CONFIRMATORY PASS for the candidate that will trade, under the same object and the same deciding economic semantics as R10;

### D21 — §33 row 7  (corrigendum A8)

**ORIGINAL:**

> | **REVISE** (simplify, defer the hierarchy) | P5/P6/P7 narrow negatives; as-of machinery sound; hundreds of design DoF | leakage via `available_time`; state explosion | ≤ 2 slow closed-bar variables; thin runtime snapshot | R3, R7 |

**REPLACEMENT:**

> | **REVISE** (multi-timeframe capability required in the runtime; hierarchical MTC as a scientific model not required for campaign 1) | P5/P6/P7 narrow negatives; as-of machinery sound; hundreds of design DoF | leakage via `available_time`; state explosion; a runtime limited to one timeframe | generic multi-clock causal snapshot; ≤ 2 slow closed-bar variables in the deciding set | R3, R7 |

### D22 — §33 row 15  (corrigendum A6)

**ORIGINAL:**

> | per-symbol measured fee/spread/slippage/funding table frozen from preflight | R2 → R3 |

**REPLACEMENT:**

> | per-symbol deciding cost semantics (fee, spread, executable price, latency, slippage envelope, funding, turnover, reject/partial rule, impact cap) frozen as hashed code in R3; identical in R10 and R11 | R2 → R3 |

### D23 — §33 row 21  (corrigendum A9, A16)

**ORIGINAL:**

> | **KEEP** as an optional parallel economic lane in its own process | only mechanism with reachable economic evidence; runtime exists | orchestration by stealth | drop (loses cheap evidence) | R0, R10 |

**REPLACEMENT:**

> | **KEEP** as a SECONDARY, OPTIONAL parallel economic lane in its own process; never delays or gates the primary lane; primary wins on conflict | only mechanism with reachable economic evidence; runtime exists | orchestration by stealth; priority by stealth | drop (loses cheap evidence) | R0 (election), R4 (deadline), R10 |

### D24 — §33 row 29  (corrigendum A6, A7)

**ORIGINAL:**

> | **REPLACE** with two-stage in one accrual + separately sized confirmation | §22 |

**REPLACEMENT:**

> | **REPLACE** with two-stage in one accrual + separately sized confirmation of the identical object; N VALID blocks with a frozen extension cap and a NOT EVALUABLE outcome | §22 |

### D25 — §35 items 5–6  (corrigendum A1, A2)

**ORIGINAL:**

> 5. In parallel and without waiting for R2/R3: **R7 (causal snapshot type) and R8 (single-leg multi-symbol runtime with one ledger and Aegis portfolio scope)**, then **R9 soak**.
> 6. Only after R3 and R9: **R4 contract freeze → R6 boundary activation → R10**.

**REPLACEMENT:**

> 5. **R4 gen4 contract freeze** (with the Lane B election settled), then **R5 reconciliation generalisation** and the production recorder's ≥ 30-day pre-activation qualification, accruing in parallel with the next item.
> 6. **R7 (multi-clock causal `MarketSnapshot` and the frozen feature function; contract-agnostic parts may start early) → R8 (multi-symbol single-leg runtime with one ledger and Aegis set scope) → R9 soak** (≥ 14 days, drills, PARITY every day, shadow economics sealed).
> 7. Only after R3, R4, R5, **R9 PASS**, the qualification and the independent boundary review: **R6 boundary activation → R10**. SOAK PRECEDES PROSPECTIVE ACTIVATION.

### D26 — §36 bullet 1  (corrigendum A1, A14)

**ORIGINAL:**

> - Do not activate any prospective boundary (gen3 or gen4) before R1, R3 and R9 are complete and independently reviewed.

**REPLACEMENT:**

> - Do not activate any prospective boundary (gen3 or gen4) before R3 (merged ≥ 7 days earlier), R4, R5, **R9 PASS**, the ≥ 30-day production-recorder qualification and the independent boundary review are all recorded; the soak precedes activation, never follows it.

### D27 — §36 bullet 3  (corrigendum A4)

**ORIGINAL:**

> - Do not run the autonomous demo/soak on the current runner: it cannot start a campaign, cannot be resumed, cannot detect a dead feed, and skips most minutes.

**REPLACEMENT:**

> - Do not run the autonomous demo/soak on the current runner: it cannot start a campaign, cannot be resumed, cannot detect a dead feed, and skips most minutes; run it only on the R8 runtime, on an engineering or pre-activation root, and never read its shadow economics before R10 closes.

### D28 — §36 bullet 5  (corrigendum A8)

**ORIGINAL:**

> - Do not build the hierarchical MTC fabric, orchestration, an inference service, a model registry, or any live adapter now.

**REPLACEMENT:**

> - Do not build the hierarchical MTC fabric (the multi-clock causal snapshot of R7 is required; the hierarchy as a scientific model is not), orchestration, an inference service, a model registry, or any live adapter now.

### D29 — §36 new bullets  (corrigendum A5, A6, A7, A9, A19, A22)

**ORIGINAL:**

> - Do not design and read a campaign in the same session; do not shorten the ≥ 7-day preregistration-to-boundary gap.

**REPLACEMENT:**

> - Do not design and read a campaign in the same session; do not shorten the ≥ 7-day preregistration-to-boundary gap.
> - Do not change any deciding economic semantic, validity criterion, extension cap, evaluator or contract-hashed module between R10 and R11, except a parameter update applied exactly and logged under an update rule preregistered in R3 (route A, which may never make the deciding economics less conservative); any other such change ends the campaign as ABORTED and any successor is a new preregistration and a new campaign; a runtime-head change beyond ENGINEERING-FIX that touches no hashed module invalidates the block in which it lands and requires a re-soak before the next block is scored (§37.2).
> - Do not let the carry lane delay, gate or outrank the primary directional futures lane.

### D30 — §38 sequence row  (corrigendum A1, A2, A31)

**ORIGINAL:**

> | R1 runtime remediation ∥ R2 preflight → R3 design/power → R4 freeze → R5 reconciliation ∥ R7 snapshot → R8 runtime → R9 soak → R6 boundary → R10 campaign |

**REPLACEMENT:**

> | (R1 ∥ R2, both from R0) → R2 feeds R3 design/power → R4 freeze → (R5 reconciliation ∥ [R1 + R7 multi-clock snapshot] → R8 runtime → R9 soak ∥ the ≥ 30-day recorder qualification, whose days count once R5 reconciles them) → **R6 boundary only after R9 PASS and the boundary review** → R10 campaign → R11 confirmation of the identical object (edges per §37 Part B) |

### D31 — §38 first-campaign row  (corrigendum A6, A7)

**ORIGINAL:**

> duration from the frozen power report (≥ 6 months, likely 9–12); K from measured effective N |

**REPLACEMENT:**

> duration from the frozen power report (≥ 6 months, likely 9–12); K from measured effective N; N VALID blocks with a mechanical, outcome-blind validity rule, a frozen extension cap and a NOT EVALUABLE outcome; deciding cost semantics frozen in R3 and identical in R11 |

### D32 — §38 strategy-family row  (corrigendum A9)

**ORIGINAL:**

> low-turnover LONG/SHORT policy; plus an optional parallel carry lane |

**REPLACEMENT:**

> low-turnover LONG/SHORT policy; plus a SECONDARY, OPTIONAL carry lane that never delays or outranks the primary lane |

### D33 — §38 MTC row  (corrigendum A8)

**ORIGINAL:**

> | MTC | "mandatory", "locked" | causal as-of semantics mandatory; hierarchy deferred; ≤ 2 slow variables in the frozen set | narrow negatives; premature commitment |

**REPLACEMENT:**

> | MTC | "mandatory", "locked" | causal as-of semantics and multi-timeframe *capability* mandatory in the runtime (generic multi-clock snapshot); hierarchical MTC as a scientific model deferred; ≤ 2 slow variables in the deciding set | narrow negatives; premature commitment; the runtime must not be limited to one timeframe |

### D34 — §38 demo row  (corrigendum A1, A4, A5)

**ORIGINAL:**

> | Demo | "engineering milestone" with a drill list | a daemon on live recorder data with quantified PASS/KILL gates | the current runner is a bounded pass |

**REPLACEMENT:**

> | Demo | "engineering milestone" with a drill list | a daemon on live recorder data with quantified PASS/KILL gates, run BEFORE the boundary on the head that will run the campaign, shadow economics sealed | the current runner is a bounded pass; the boundary is irreversible |

### D35 — §39 boundary bullet  (corrigendum A1, A15)

**ORIGINAL:**

> Blockers: no frozen contract; no preregistered campaign; no frozen power/inference module; no runtime that can execute the campaign's policy; no soak; no independent boundary review; no ≥ 30-day qualifying pre-activation period.

**REPLACEMENT:**

> Blockers: no frozen contract (R4); no preregistered campaign with frozen deciding economics and validity rules (R3); no frozen power/inference module (R3); no gen4 reconciliation (R5); no runtime that can execute the campaign's policy (R7, R8); **no R9 PASS — the soak must precede activation**; no independent boundary review; no ≥ 30-day qualifying pre-activation period.

### D36 — §24 Execution simulation row  (corrigendum A6, A18)

**ORIGINAL:**

> | Execution simulation | recorded-quote touch crossing + configured slippage; deterministic adverse model | queue-position models (hftbacktest), impact models | **KEEP** for demo; **REVISE** for R11 with measured slippage distributions and a size-dependent impact term for BTC/ETH from Tier B depth | § 23 fidelity table | R11 |

**REPLACEMENT:**

> | Execution simulation | recorded-quote touch crossing + configured slippage; deterministic adverse model | queue-position models (hftbacktest), impact models | **KEEP** for demo, R10 and R11 (the deciding fill model is frozen as hashed code in R3 and identical in R10 and R11); measured slippage distributions and a size-dependent impact term for BTC/ETH from Tier B depth enter only as non-deciding stress/diagnostic analyses (§23 freeze rule) or through routes (A)/(C) | § 23 fidelity table | R3 (freeze); R11/R14 (stress only) |

### D37 — §35 item 2 (R1 enumeration)  (corrigendum A13, A34)

**ORIGINAL:**

> 2. **R1 — Runtime integrity remediation**, as separate reviewed engineering PRs, in this order: (a) `_software()` identity fix with a two-sided test; (b) operator lifecycle (`resume`/`resolve` via `start()` with log-tail clock seeding; `flatten` without `allow_dirty` on CAMPAIGN); (c) staleness guard outside the decision path; (d) daemon loop with SIGTERM deferral; recorder incremental last-minute publication or cadence-aligned catch-up; atomic parquet write; (e) dispute-clearing paths (audited re-derivation) and the hedge correction timeout; (f) `resume` ordering; funding-veto wiring or removal of dead limits from the schema; unified valuation price; (g) replay-parity policy for `seq` and `OPERATOR`; re-include deterministic risk fields in the hash; (h) delete the Freqtrade path; SHA-pin Actions, hash-pin the lock, digest-pin images; external dead-man check; backup/restore drill. Acceptance: 72 h SOAK on an engineering root with a restart and PARITY on every day.

**REPLACEMENT:**

> 2. **R1 — Runtime integrity remediation**, as separate reviewed engineering PRs R1-a … R1-o in the order given in §37 R1: source-identity/CAMPAIGN self-check; AEG-1 persisted-equity re-seeding; `risk.json` continuity; daemon/service semantics; clock hardening (including the injected operational clock); real staleness detection from the operational clock; recorder/runner cadence with the funding-settlement-minute rule; atomic parquet publication and recorder down/up evidence; dispute lifecycle and crash/restart semantics (including disk-full behaviour); replay-parity policy for `seq` and `OPERATOR` with deterministic risk fields re-included in the hash; unreachable Aegis rules wired or removed; unified valuation price; Freqtrade deletion; supply chain and operations (SHA-pinned Actions, hashed lock, digest-pinned images, off-host dead-man check, chrony, backup/restore drill, separate users); evidence-class labels and the verifier check. Acceptance: 72 h SOAK on an engineering root with a planned restart and a `SIGKILL`, PARITY on every day, zero `SKIPPED_STALE` minutes while the recorder was healthy.

### D38 — §8 EXE-4 row  (corrigendum A34)

**ORIGINAL:**

> | EXE-4 | LOW | Fill-model reference for slippage attribution is the kline close, not the mid; no bps figure emitted (A12 gaps). | gen4 freeze |

**REPLACEMENT:**

> | EXE-4 | LOW | Fill-model reference for slippage attribution is the kline close, not the mid; no bps figure emitted (A12 gaps). | R3 (deciding cost semantics frozen as hashed code shared by runtime and evaluator; mid-referenced bps slippage reported as a non-deciding diagnostic per §23) |

### D39 — §9 ACC-8 row  (corrigendum A34)

**ORIGINAL:**

> | ACC-8 | LOW | Research-side `TargetSpec` fee (5 bps "spot taker") is half the demo's spot fee; A12's two recorded gaps (no bps slippage figure; slippage measured against close, not mid) remain open. | Research and execution cost models are not the same object. | gen4 freeze |

**REPLACEMENT:**

> | ACC-8 | LOW | Research-side `TargetSpec` fee (5 bps "spot taker") is half the demo's spot fee; A12's two recorded gaps (no bps slippage figure; slippage measured against close, not mid) remain open. | Research and execution cost models are not the same object. | R3 (one cost-model module imported by both the runtime and the evaluator; frozen as hashed code) |

### D40 — §11 REC-2 resolve cell  (corrigendum A7, A34)

**ORIGINAL:**

> an explicit protocol rule that a crash-ended campaign is counted, not rescued;

**REPLACEMENT:**

> an explicit protocol rule that a crash-affected block is judged only by the frozen validity evaluator (INVALID if the halt is not recovered within the frozen window or lacks a record; never a negative result; never substituted), the campaign extending per the frozen rule;

### D41 — §30 Gate 2  (corrigendum A25)

**ORIGINAL:**

> A separate authenticated adapter package exercised ≥ 30 days on testnet or read-only live reconciliation with:

**REPLACEMENT:**

> The R9 soak gates re-run in full on the exact deployable head (the runtime plus the adapter package in testnet mode) and frozen as ENGINEERING evidence; a separate authenticated adapter package exercised ≥ 30 days on testnet or read-only live reconciliation (a read-only key with no trading permission; no trade-capable key before Gate 4 is signed) with:

### D42 — §30 Gate 4  (corrigendum A25)

**ORIGINAL:**

> reviewed before any key is created,

**REPLACEMENT:**

> reviewed before any trade-capable key is created,

**Sections checked and left unchanged:** §2–§7, §10, §12, §15, §17–§21, §29, §31, §32, §34 — no sentence in them conflicts with the corrected §37; §7 and §29 ("one engine over the lanes that share an account"; "one Aegis, one execution layer, one account ledger") agree with R13 as restored; §12's statement that the 30-day gate is re-run on the production recorder identity before R6 activation and §32's "runtime soak on the same policy before boundary" already agree with the corrected order. §8, §9, §11, §24 and §30 carry the small cell edits D36–D42 found by the verification round.

---

## Part E — Final adoption verdict

**ADOPT WITH REMAINING OWNER DECISIONS**

The corrected roadmap is internally consistent, preserves every confirmed remediation requirement, keeps futures-first as the primary lane, keeps hierarchical MTC out of the first campaign while requiring multi-timeframe capability in the runtime, freezes the deciding economics before the boundary, makes block invalidation mechanical, and places the irreversible boundary after the soak. It can become the authoritative master plan on adoption at R0.

**Remaining owner decision (the only one the completed audit cannot resolve):**

1. **Lane B election (carry lane).** Elect or decline the SECONDARY, OPTIONAL carry campaign. Deadline: R4, because its spot streams (BTCUSDT and, if named, ETHUSDT spot kline_1m and bookTicker) must be in the gen4 contract. Default if undecided at R4: not elected; a later carry campaign then needs its own contract and boundary and never delays R6. Nothing else in the roadmap changes under either answer.

**Defaults that apply unless the owner overrides them (not blocking decisions):** the R1–R9 engineering effort cap (the audit suggests ≤ 10 person-weeks); the extension cap E = ⌈N/2⌉ blocks and the continuation size N′ ≥ ⌈N/2⌉ and ≥ 3 blocks, which R3's power report fixes as numbers inside the frozen rule; the interim-look rule (alpha-spending at block ends).

**What this corrigendum does not do.** It does not reopen P4-HOLD, P8, P13 or P14; it does not read Styx; it does not use burned blocks for any deciding purpose; it does not create gen4, activate a boundary, create scientific evidence, authorise real money, change ProjectChimera code, or change Git, GitHub or VPS state; it starts no scientific analysis; it does not replace futures-first; it does not restore mandatory hierarchical MTC; it adds no distributed infrastructure and no model complexity.

This corrigendum remains a PROPOSAL. It creates no scientific evidence, no prospective boundary, no alpha claim and no real-money authority. Historical positive results are not prospective; engineering or demo success is not alpha; P4-HOLD remains retired and unread; Styx remains sealed and unread; P8 remains withdrawn.
