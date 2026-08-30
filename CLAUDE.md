# ProjectChimera — Claude Code standing instructions

This file is intentionally concise because Claude Code loads it into every project session. The detailed operating manual is `docs/claude_code_operating_guide.md`; read it on demand for any major task.

## Before any major task

For any multi-file implementation, research checkpoint, audit, migration, or prompt intended to launch another Claude Code session:

1. Read `docs/claude_code_operating_guide.md`.
2. Read `docs/current_development_plan.md` and `docs/current_development_plan_post_audit.md`.
3. Read `docs/research_roadmap.md` and the preregistration/closure documents relevant to the checkpoint being touched.
4. Reconstruct current Git/research state from repository evidence. Do not trust chat summaries, PR prose, or prior agent reports when the repository can answer the question.

## Authority order

When instructions conflict, use this order:

1. explicit current user instruction;
2. frozen research preregistration, safety contract, and executable repository guardrails;
3. this `CLAUDE.md` and project-scoped rules;
4. the task-specific prompt;
5. installed skills, generic subagents, community workflows, and general best practices.

A skill or subagent may improve execution technique. It may never weaken a scientific boundary, change a frozen decision rule, select parameters after results, open sealed evidence, or expand live-trading reachability.

## Scientific discipline

- Preregistration must be committed and pushed before the first result it governs is calculated or read.
- Primary research generation must begin from a clean, reconstructible source state.
- Never rescue a negative result by changing thresholds, models, costs, folds, horizons, venues, filters, or success rules after seeing the result.
- Keep negative results visible.
- Do not manufacture new independence by splitting already-read historical periods.
- `P4-HOLD` remains retired/unread and Styx remains sealed unless an explicit later contract says otherwise.
- Styx is a repository non-read historical seal with a hindsight-era ceiling, not truly prospective blind evidence.
- No real-money promotion follows from a historical backtest, smoke test, short paper run, or single positive checkpoint.

## Model / effort default

Choose deliberately before a major session; do not use one model for everything.

- **Opus 5 + xhigh**: default executor for large autonomous coding, research implementation, complex refactors, and end-to-end checkpoints.
- **Fable 5 + xhigh**: default independent adversarial reviewer / deep research auditor. Use `max` only for a rare one-shot task where maximum capability matters more than time/usage.
- **Sonnet 5 + medium/high**: routine fixes, focused implementation, docs, straightforward tests, and CI repair when the task is well bounded.
- Use `high` for normal intelligence-sensitive work and `xhigh` for demanding coding/agentic work. Avoid repeatedly changing effort inside one long cached session.

For the current structural-carry checkpoint, the executor recommendation is **Opus 5 + xhigh**; a later independent audit should normally use **Fable 5 + xhigh**.

## Session discipline

- New substantial task = new session. Do not drag unrelated history into the next checkpoint.
- For large work: inspect first, then act, then verify. Use Plan Mode when the design is not already frozen or broad edits need review before touching disk.
- In autonomous scientific runs, a detailed preregistered prompt may itself be the approved plan; do not stop for routine choices the repository can resolve.
- Use `/context` to inspect context pressure. Prefer `/clear` / a new session for unrelated work.
- If compaction is unavoidable, use focused `/compact <instructions>` and explicitly preserve chronology, unresolved findings, SHAs, and safety boundaries.
- Use `/btw` for side questions that should not pollute the main task history.

## Skills and extensions

- Use a small task-specific skill set rather than installing everything available.
- Project heuristic: **3–6 well-separated skills** for a large task; one skill per genuinely distinct capability. This is a heuristic, not an Anthropic hard limit.
- Inspect third-party skill contents/security before installation. Prefer primary-source/reputable skills with little semantic overlap.
- Skill descriptions consume startup context; skill bodies load when invoked. Avoid dozens of overlapping `research`, `review`, `planning`, or `testing` skills.
- Enable only MCP servers/tools that materially help the task. Extra tool descriptions consume context and increase accidental action surface.
- Current P13 candidate skills, to be revalidated before use: `research`, `backtest-expert`, `dimensional-analysis`, `python-testing-patterns`.

## Subagents and workflows

- Delegate high-volume searches, independent recomputation, and narrow audits to subagents so their file/log reads stay out of the main context.
- Use fresh-context reviewers for load-bearing scientific, accounting, boundary, and safety conclusions.
- Do not create redundant parallel agents merely because parallelism is available; usage can grow rapidly.
- Built-in Explore/Plan agents may not receive this `CLAUDE.md`; restate relevant scientific/safety boundaries explicitly in critical subagent prompts.
- Ultracode/dynamic workflows are appropriate for genuinely large, decomposable, high-value tasks. They are not the default for a small fix and can consume substantially more usage.

## Verification and Git

- A task is not done because code was written. Run the repository's real tests, verifiers, lint/static checks, and relevant end-to-end controls.
- Prefer two-sided synthetic controls for decision logic, not only tests against the committed outcome.
- For accounting, independently hand-trace representative LONG/SHORT or multi-leg examples and verify units, notional, fees, funding, leverage, and PnL.
- Keep important Git chronology visible. Do not amend/squash/force-push away preregistration-before-result history.
- Large research PRs stay draft until independently audited. Do not merge them merely because their author reports green tests.

## Platform awareness

The project is often operated through Claude Code on the web/mobile cloud surface. Do not assume a local terminal, local environment variables, local MCP servers, or machine-local auto memory unless the user explicitly says the session is local/Remote Control.

Repository files are the durable cross-session/cross-environment memory. Auto memory is supplementary, not the source of truth.

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

Turn work into verifiable goals. For a multi-step task, keep a short internal execution plan whose steps each have an observable check. Continue until the requested result and verification criteria are satisfied, not merely until an implementation exists.
