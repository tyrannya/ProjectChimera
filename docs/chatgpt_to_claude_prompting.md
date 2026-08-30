# ProjectChimera — ChatGPT → Claude Code prompting handoff

This file is written for a **fresh ChatGPT/planning assistant**, not for Claude Code itself.

If the user opens ProjectChimera from another ChatGPT account and asks for a Claude Code prompt, read this file first, then read `docs/claude_code_operating_guide.md`, `CLAUDE.md`, `docs/current_development_plan.md`, `docs/current_development_plan_post_audit.md`, and `docs/research_roadmap.md` before preparing any major prompt.

The repository is the source of truth. Do not assume checkpoint SHAs, numbers, model results, or roadmap state from old chat memory if the repository can answer them.

## Before writing any major Claude prompt

Decide these first:

1. exact task type;
2. Claude model;
3. effort;
4. new session vs continuation;
5. normal / Plan Mode / Ultracode;
6. task-specific Skills;
7. MCP/external tools actually needed;
8. subagent/fresh-context verifier strategy;
9. authoritative repository files;
10. scientific and safety boundaries;
11. Git chronology;
12. verification requirements;
13. stop conditions;
14. draft-PR vs merge policy;
15. independent reviewer model.

For every major Claude prompt, visibly state at least:

> **Model:** ...  
> **Effort:** ...  
> **Session:** ...  
> **Mode:** ...  
> **Skills:** ...

Do not silently use one model for every job.

## Project model defaults

### Opus 5

Default **executor** for major autonomous ProjectChimera work: new research checkpoints, complex coding, acquisition + implementation + tests + artifacts, major refactors and systems engineering.

Default: **Opus 5 + xhigh**.

### Fable 5

Default **independent adversarial reviewer / deep scientific auditor**: reconstruct claims from Git/artifacts, attack chronology, recompute results, audit accounting/safety, challenge roadmap choices, and decide whether a large PR is safe to merge.

Default: **Fable 5 + xhigh**. Use `max` only exceptionally for a one-shot highest-stakes review where usage/time are secondary.

### Sonnet 5

Default for bounded routine work: small fixes, localized tests, documentation, CI repair and mechanical cleanup.

Default: **Sonnet 5 + medium/high**.

Preferred separation for major work:

> **Opus builds → Fable independently audits → merge only after audit + green CI + user authorization.**

## Effort

- `low`: trivial/low-risk/mechanical.
- `medium`: docs, routine fixes, straightforward tests.
- `high`: normal quality-sensitive coding/analysis.
- `xhigh`: major checkpoints, agentic coding, large audits, complex repository work.
- `max`: rare one-shot maximum-depth task; not a default.

## Session discipline

New substantial task = **new Claude Code session**. Do not drag unrelated history into a new checkpoint or independent audit.

Use `/context` to inspect pressure. Prefer a new session or `/clear` for unrelated work. If `/compact` is unavoidable in a scientific run, give focused instructions preserving branch, important SHAs, preregistration SHA, first-result SHA, unresolved findings, safety boundaries, P4-HOLD/Styx state, PR state and remaining verification.

`/btw` is appropriate for side questions that should not pollute the main task. `/voice` can help formulate complex instructions. `/loop` is useful for repeated checks inside an active session, not as a permanent scheduler.

## Normal vs Plan Mode vs Ultracode

Use **normal agentic mode** when the task is already well specified by a rigorous prompt and implementation is mostly serial.

Use **Plan Mode** when requirements/architecture are not frozen, the repository is unfamiliar, the blast radius is broad, or the user wants the approach reviewed before edits. Do not turn Plan Mode into an unnecessary approval bottleneck when a detailed scientific prompt already freezes the design.

Use **Ultracode/dynamic workflows** only when the task is genuinely large and decomposable and parallel lanes improve independence or wall-clock time. It can consume substantially more usage. Do not use it for small fixes or when parallel agents would duplicate the same evidence.

Scientific stages remain serial even if work inside a stage is parallel:

```text
preregistration commit + push
        ↓
implementation/acquisition
        ↓
first governed result
        ↓
closure
        ↓
independent audit
```

## Skills

Skills are procedural instructions, not intelligence upgrades. More is not automatically better.

Project heuristic for a large task: **3–6 well-separated Skills**, roughly one per genuinely distinct capability. This is a heuristic, not an Anthropic hard limit.

Before recommending third-party Skills, inspect their source/security/dependencies and check for overlap or workflow conflicts.

Current strong candidates for structural trading research, to be revalidated before use:

- `research` — primary-source external research/API/spec work;
- `backtest-expert` — trading/backtest friction and robustness methodology;
- `dimensional-analysis` — quantity/notional/leverage/fees/funding/unit checks;
- `python-testing-patterns` — pytest and stronger positive/negative controls.

Repository contracts and explicit task instructions outrank any Skill. A Skill may improve execution technique; it may never change a frozen research rule, open sealed evidence, choose parameters after results, or weaken safety.

## MCP and subagents

Enable only MCP servers/tools that materially help. Extra tool descriptions consume context and expand action surface. Cloud/mobile sessions must not assume local MCP servers, local environment variables or machine-local memory.

Use subagents for high-volume searches, provenance work, independent recomputation and narrow audits. Use fresh-context reviewers for load-bearing scientific, accounting, boundary and safety conclusions.

Critical subagent prompts should restate scientific/safety restrictions because built-in Explore/Plan agents may not receive all project instructions automatically.

## Major prompt structure

A serious ProjectChimera prompt should normally contain:

1. role/objective;
2. repository and known base SHA, with instruction to verify `origin/main`;
3. minimal necessary current scientific state;
4. preflight files to read;
5. exactly one scoped question;
6. non-negotiable boundaries;
7. scientific design/preregistration requirements;
8. Git chronology;
9. implementation/acquisition scope;
10. tests/verifiers/accounting checks;
11. independent recomputation/audit requirements;
12. explicit STOP conditions for positive and negative outcomes;
13. PR policy;
14. required final report fields.

For a major research checkpoint, the authoring Claude should normally open a **draft PR and not merge it**.

## Independent audit style

For Fable review, avoid anchoring it with all expected conclusions. Do not ask it merely to confirm the author’s numbers.

Prefer instructions such as:

> Reconstruct what the repository actually establishes. Do not trust PR prose, STATUS files, prior Claude/ChatGPT reports, or claimed verdicts. Derive important claims independently from Git history, committed artifacts, tests, manifests and executable recomputation. Actively try to falsify the conclusions. Mark unverified claims explicitly.

Use a fresh session and fresh-context verifier subagents where useful.

## Persistent ProjectChimera scientific rules

Unless current repository contracts say otherwise:

- negative results remain visible;
- do not redefine success after seeing results;
- do not promote secondary winners post hoc;
- repeatedly-read directional folds are adaptive/burned evidence;
- do not manufacture independence by slicing known history more finely;
- `P4-HOLD` remains retired/unread, not a spare holdout;
- Styx remains sealed and has a hindsight-era ceiling because its market dates predate formal sealing;
- stronger future confirmation should eventually be prospective, using a strategy frozen before future wall-clock data occurs;
- P6 dirty-tree fit provenance must not be silently upgraded to clean-checkout reproducibility;
- P8 remains unopened until its frozen eligibility condition is genuinely satisfied;
- futures execution/paper smoke is not alpha evidence;
- describe Chimera futures live reachability separately from the deliberately gated legacy Freqtrade live capability;
- no historical backtest, smoke, short paper run or single positive checkpoint authorizes real money;
- Aegis remains central risk authority.

## Current roadmap hint

At the time this file was created, the selected next major research axis was **structural/non-directional BTC funding-basis carry feasibility** (delta-hedged long spot / short perpetual), not another immediate directional-ML rescue. P4 tested derivatives as predictive features; the structural branch tests funding/basis as the payoff mechanism itself.

This is only a handoff hint. Re-read the current roadmap before writing a prompt because the repository may have advanced.

## Final planner checklist

Before sending any large Claude prompt, confirm:

- [ ] exact task;
- [ ] model;
- [ ] effort;
- [ ] fresh session;
- [ ] normal / Plan / Ultracode;
- [ ] 3–6 relevant Skills if needed;
- [ ] MCP/tools actually needed;
- [ ] authoritative project docs;
- [ ] scientific/safety boundaries;
- [ ] Git chronology;
- [ ] parallel vs serial stages;
- [ ] tests/verifiers;
- [ ] independent recomputation;
- [ ] stop conditions;
- [ ] draft PR or merge policy;
- [ ] independent reviewer model.

If those decisions have not been made, the prompt is not ready.
