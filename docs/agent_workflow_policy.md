# ProjectChimera — Plan-First Agent Workflow Policy

Status: **standing engineering/governance procedure**. This document does not itself open a research checkpoint, change a scientific result, authorize live trading, or supersede a frozen preregistration.

## 1. Standing workflow

Every substantive Claude Code task follows:

```text
ARCHITECT / PLAN
    ↓
PLAN CHECK
    ↓
EXECUTION
    ↓
VERIFICATION
    ↓
INDEPENDENT REVIEW where required
```

The rule is universal for repository-changing work. The depth of the plan scales with the task.

### Trivial task

A one-line documentation typo may use a very short plan:

- inspect exact file and governing context;
- state intended edit and check;
- edit;
- verify diff.

### Major task

A major implementation/research/safety task must plan:

- exact current Git state;
- authoritative documents;
- scope and non-scope;
- invariants;
- scientific/evidence boundaries;
- dependencies;
- implementation sequence;
- tests/witnesses;
- CI requirements;
- Git/PR chronology;
- stop/escalation conditions;
- independent review boundary.

## 2. Plan Mode is not scientific authorization

Planning does not permit the model to cross a scientific barrier.

For governed research:

```text
plan/design
→ preregistration committed + pushed
→ implementation/acquisition
→ first governed result
→ closure
→ independent audit
```

A planner may inspect allowed evidence needed to design the protocol, but it may not calculate/read a result whose contract requires a future preregistration boundary first.

## 3. Architect responsibilities

Before edits, the architect must:

1. verify branch/head/base and working-tree state;
2. read `CLAUDE.md` and the task-relevant authoritative docs;
3. identify what is already frozen versus genuinely undecided;
4. locate exact code/data/artifact surfaces to be touched;
5. identify likely failure modes and rollback/stop points;
6. state the minimum coherent change;
7. state how success will be proven;
8. identify which claims require current first-party documentation;
9. decide which work is serial and which is safely parallel;
10. decide whether independent post-implementation review is mandatory.

The plan should be concrete enough that execution is mostly implementation rather than rediscovery.

## 4. Approval semantics

The project uses two forms of plan approval:

### Explicit operator approval

Required when:

- the user asked to review the plan before changes;
- requirements contain a material scientific ambiguity;
- destructive action is proposed;
- live/authenticated reachability changes;
- a new research question/checkpoint is being opened;
- architecture direction is not already adopted.

### Pre-authorized execution

If the user/task contract explicitly authorizes end-to-end execution, Claude still performs the Plan phase first but may proceed directly into execution once the plan proves the task is in-scope and no stop condition is hit.

Therefore **Plan-First does not mean Ask-First for every minor decision**.

## 5. Context7 policy

Use Context7 when current third-party library/framework API behavior matters.

Examples:

- Python package APIs;
- websocket/client libraries;
- Pydantic/Polars/Pandas/DuckDB APIs;
- ML framework interfaces;
- version-specific migration behavior.

Do not use Context7 as the final authority for:

- Binance/exchange market semantics when official exchange documentation exists;
- current ProjectChimera Git state;
- scientific contracts;
- safety policy;
- active roadmap state.

Priority:

```text
ProjectChimera repository truth
or official vendor/exchange primary docs
    >
Context7 current-library retrieval
    >
model memory
```

## 6. Knowledge-vault policy

`knowledge/` is an Obsidian-compatible external research vault.

It is useful for:

- papers;
- repository/technology landscape notes;
- Binance/market microstructure source summaries;
- strategy-family surveys;
- failure-mode catalogs;
- external benchmark notes;
- research syntheses;
- independent audit inputs/outputs that are not themselves repository authority.

It is never authoritative for:

- current SHA/branch;
- active PR state;
- result values;
- preregistration hash;
- evidence eligibility;
- merge status;
- live risk state;
- active roadmap adoption.

Every important note should carry source links and, where freshness matters, a checked date.

## 7. Subagent policy

Use only genuinely non-overlapping roles.

Recommended temporary roles:

- **repo-verifier** — reconstruct Git/files/tests/CI claims read-only;
- **external-quant-researcher** — external papers/repos/market evidence;
- **accounting-auditor** — units/notional/funding/fees/PnL/margin;
- **safety-auditor** — recovery, stale state, kill switches, operator paths;
- **docs-researcher** — current third-party documentation via Context7/primary docs.

Rules:

- restate load-bearing boundaries in each critical subagent prompt;
- do not let subagents silently edit the same files in parallel;
- do not run source-rewriting mutation tools concurrently with ordinary tests in the same worktree;
- subagent outputs are evidence to reconcile, not merge approval;
- an author's subagent is not an independent post-hoc certifier.

## 8. Ultracode policy

Plan-First applies regardless of workflow mode.

Use normal workflow when the plan is mostly serial.

Use Ultracode only after the architect establishes that the task contains independent lanes that can safely run in parallel without violating scientific chronology.

Do not use Ultracode merely because a task is important or because more agents appear stronger.

## 9. Verification before completion

Before saying a task is complete, the executor must verify the checks identified in the plan. Depending on task type this may include:

- targeted tests;
- negative/positive controls;
- mutation witnesses;
- deterministic replay;
- lint/static/compile;
- full relevant suite;
- manifest/hash verification;
- independent accounting arithmetic;
- exact-head CI;
- cross-platform CI;
- source provenance checks;
- no-live/credential scans.

The final report must distinguish:

- verified;
- inferred but not verified;
- blocked;
- deliberately out of scope.

## 10. In-flight task rule

A new governance/workflow policy on `main` must not be used to widen or restart an already-running governed branch solely to pick up the new instructions.

The current task finishes under its already-frozen contract unless the owner explicitly changes it or a genuine safety/scientific blocker requires escalation.
