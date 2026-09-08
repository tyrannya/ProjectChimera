# ProjectChimera — Claude Code standing instructions

This file is intentionally concise because Claude Code loads it into every project session. Detailed procedures live in `docs/claude_code_operating_guide.md` and `docs/agent_workflow_policy.md`; read them on demand for any substantial task.

## Before any substantial task

For every task that can modify repository state, change behavior, alter scientific interpretation, create a result, change infrastructure, or open/close a PR:

1. **Start in Architect / Plan Mode. Do not write code or edit files first.**
2. Read `docs/agent_workflow_policy.md`.
3. For multi-file implementation, research checkpoint, audit, migration, or another major task, also read `docs/claude_code_operating_guide.md`.
4. Read `docs/current_development_plan.md`, the active roadmap it points to, `docs/research_roadmap.md`, and the preregistration/closure documents relevant to the checkpoint being touched.
5. Reconstruct current Git/research state from repository evidence. Do not trust chat summaries, PR prose, status prose, memory notes, or prior agent reports when the repository can answer the question.
6. Produce a concise implementation/audit plan with explicit scope, invariants, verification points, Git/PR consequences, and stop conditions.
7. Only after the planning phase is complete may execution begin inside the approved/frozen scope.

For a trivial read-only lookup, the planning phase may be one short internal/visible plan, but the inspect-before-act rule still applies.

An already-running governed session may finish under its existing frozen contract. Do not restart or widen an in-flight scientific/remediation task solely because this file changed on `main`.

## Authority order

When instructions conflict, use this order:

1. explicit current user instruction;
2. frozen research preregistration, safety contract, and executable repository guardrails;
3. this `CLAUDE.md` and project-scoped rules;
4. task-specific prompt;
5. detailed operating/workflow docs;
6. installed skills, generic subagents, community workflows, external memory, and general best practices.

A skill, MCP server, memory system, or subagent may improve execution technique. It may never weaken a scientific boundary, change a frozen decision rule, select parameters after results, open sealed evidence, expand live-trading reachability, or override repository truth.

## Scientific discipline

- Preregistration must be committed and pushed before the first result it governs is calculated or read.
- Primary research generation must begin from a clean, reconstructible source state.
- Never rescue a negative result by changing thresholds, models, costs, folds, horizons, venues, filters, or success rules after seeing the result.
- Keep negative results visible.
- Do not manufacture new independence by splitting already-read historical periods.
- `P4-HOLD` remains retired/unread and Styx remains sealed unless an explicit later contract says otherwise.
- Styx is a repository non-read historical seal with a hindsight-era ceiling, not truly prospective blind evidence.
- No real-money promotion follows from a historical backtest, smoke test, short paper run, or single positive checkpoint.
- Aegis remains the sole central risk authority unless an explicit later governance decision changes that.

## Model / effort default

Choose deliberately before a major session; do not use one model for everything.

- **Opus 5 + xhigh**: default executor for large autonomous coding, research implementation, complex refactors, and end-to-end checkpoints.
- **Fable 5.1 + xhigh**: preferred independent adversarial reviewer / deep scientific auditor. Use `max` only for a genuinely highest-stakes one-shot design, causal, accounting, or merge-blocking decision.
- **Sonnet 5 + medium/high**: routine fixes, focused implementation, docs, straightforward tests, and CI repair when the task is well bounded.
- Use `high` for normal quality-sensitive work and `xhigh` for demanding coding/agentic work. Reserve `max` for cases where maximum reasoning quality is worth the additional usage.

A named Fable 5.1 audit must actually be served by Fable 5.1; a fallback model does not silently inherit that certification.

## Session discipline

- New substantial task = new session unless it is genuinely a continuation of the same governed task.
- **Plan first, then execute, then verify** is the standing workflow for every substantive change.
- The planning phase must not itself generate governed scientific results that belong after a preregistration barrier.
- Once a frozen plan is approved, do not stop for routine choices the repository can resolve safely.
- Use `/context` to inspect context pressure. Prefer `/clear` / a new session for unrelated work.
- If compaction is unavoidable, use focused `/compact <instructions>` and explicitly preserve chronology, unresolved findings, SHAs, hashes, safety boundaries, and next permitted action.
- Use `/btw` for side questions that should not pollute the main task history.

## Durable knowledge and external research vault

Repository Git history, preregistrations, manifests, executable guards, current roadmap documents, and evidence artifacts are authoritative.

`knowledge/` is an **external research / second-brain vault only**. It may contain papers, repository notes, market-microstructure notes, source summaries, and draft syntheses. It is deliberately **NON-AUTHORITATIVE**.

Rules:

- never use `knowledge/` as the source of truth for current branch/head, scientific result, preregistration state, merge eligibility, live safety state, or active roadmap;
- when a knowledge note conflicts with Git or an authoritative project document, Git/repository truth wins;
- knowledge notes should link to primary sources and record access dates where practical;
- promotion of a claim from `knowledge/` into a scientific contract requires an explicit governance/research step.

## Skills, MCP and current documentation

- Use a small task-specific skill set rather than installing everything available.
- Enable only MCP servers/tools that materially help the task.
- **Use Context7 for current third-party library/framework documentation when version/API freshness matters.** Prefer official exchange/vendor documentation for exchange semantics and safety-critical API behavior.
- Do not treat Context7, Obsidian, auto-memory, or any other external context source as ProjectChimera repository truth.
- Inspect third-party skills/plugins before installation and avoid opaque model-routing/fallback tools for governed scientific work.

## Subagents and workflows

- Use subagents only for genuinely separable work such as high-volume search, source-provenance inspection, independent recomputation, documentation lookup, accounting checks, safety review, and alternate causal interpretation.
- Critical subagent prompts must restate load-bearing scientific/safety boundaries explicitly.
- Subagent output is evidence, not authority; the lead agent must reconcile disagreements.
- An author's subagent is not equivalent to a later independent reviewer.
- Ultracode/dynamic workflows are appropriate only for genuinely large, decomposable, high-value tasks. They are not an intelligence level and can consume substantially more usage.
- Scientific stage order remains serial even when independent work inside one stage is parallelized.

## Verification and Git

- A task is not done because code was written. Run the repository's real tests, verifiers, lint/static checks, and relevant end-to-end controls.
- Prefer two-sided synthetic controls for decision logic, not only tests against the committed outcome.
- For accounting, independently hand-trace representative LONG/SHORT or multi-leg examples and verify units, notional, fees, funding, leverage, margin, and PnL.
- Keep important Git chronology visible. Do not amend/squash/rebase/force-push away preregistration-before-result history.
- Large research PRs stay draft until independently audited. Do not merge them merely because their author reports green tests.
- Exact-head CI is required when the governing task/PR contract requires CI; CI on an older head does not certify a newer head.

## Platform awareness

The project is often operated through Claude Code on the web/mobile cloud surface or Remote Control. Do not assume a local terminal, local environment variables, local MCP servers, or machine-local auto-memory unless the current environment proves they exist.

Repository files are the durable cross-session/cross-environment memory. Auto memory and Obsidian are supplementary, not governance.

## General coding behavior

### Think before coding

- State material assumptions explicitly.
- If multiple interpretations would materially change behavior, surface them rather than silently choosing.
- Prefer the simplest adequate approach and push back on unnecessary complexity.
- Ask only when ambiguity cannot be resolved safely from the repository and would materially change the contract.

### Simplicity first

- No speculative features or abstractions beyond the task.
- Avoid configurability that has no current use.
- Match existing design and style unless the task explicitly changes them.

### Surgical changes

- Touch only what is required by the task.
- Do not refactor adjacent code merely because it could be prettier.
- Remove only dead code/imports created by your own change unless cleanup is explicitly in scope.

### Goal-driven execution

Turn work into verifiable goals. For a multi-step task, keep a short execution plan whose steps each have an observable check. Continue until the requested result and verification criteria are satisfied, not merely until an implementation exists.
