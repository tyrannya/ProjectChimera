# ProjectChimera — Claude Code operating guide

Status: **project operating manual for choosing models, effort, skills, prompting, session structure, verification, and handoff when using Claude Code on ProjectChimera.**

Last reviewed against current Anthropic documentation: **2026-08-30**.

This is an operating guide, not a research preregistration. A frozen checkpoint contract always outranks generic advice in this file.

## 1. Why this file exists

ProjectChimera uses Claude Code for long autonomous engineering and research tasks, often from the web/mobile cloud surface. The project also deliberately uses different Claude models for implementation and independent review.

The goal of this file is that a new operator — or a new Claude/ChatGPT session asked to prepare a Claude prompt — can answer, from the repository alone:

- which model to use;
- which effort level to use;
- whether to use Plan Mode, Ultracode/dynamic workflows, subagents, or a normal session;
- which Skills to install and how many;
- when to use MCP;
- how to protect the context window;
- how to structure a large prompt;
- how to preserve scientific Git chronology;
- how to verify the result;
- when to start a new session instead of continuing an old one;
- how to keep cloud/mobile sessions reproducible without relying on machine-local memory.

The repository remains the durable source of truth. Chat memory and model auto-memory are conveniences, not project governance.

---

## 2. Sources and confidence

The operating rules below combine:

### Primary / official Anthropic sources

- Claude Code memory / `CLAUDE.md`:
  https://code.claude.com/docs/en/memory
- Claude Code context window:
  https://code.claude.com/docs/en/context-window
- Claude Code Plan Mode / permission modes:
  https://code.claude.com/docs/en/permission-modes
- Claude Code Skills:
  https://code.claude.com/docs/en/slash-commands
- Claude Code subagents:
  https://code.claude.com/docs/en/subagents
- Claude Code agents / parallel work:
  https://code.claude.com/docs/en/agents
- Claude Code MCP:
  https://code.claude.com/docs/en/mcp
- Claude Code commands:
  https://code.claude.com/docs/en/commands
- Claude Code web/cloud:
  https://code.claude.com/docs/en/web-quickstart
- Remote Control / mobile:
  https://code.claude.com/docs/en/remote-control
- Claude model selection:
  https://platform.claude.com/docs/en/about-claude/models/choosing-a-model
- Claude effort:
  https://platform.claude.com/docs/en/build-with-claude/effort
- Dynamic workflows / Ultracode:
  https://claude.com/blog/introducing-dynamic-workflows-in-claude-code
  https://claude.com/blog/a-harness-for-every-task-dynamic-workflows-in-claude-code

### Community sources / practitioner evidence

- Skills catalog:
  https://www.skills.sh/
- AIHero Skills:
  https://www.aihero.dev/
- The Coding Sloth, “I Have Spent 1000+ Hours With Claude Code. This Is What I Learned”:
  https://www.youtube.com/watch?v=YAsxyoTWFDA

Community advice is treated as a heuristic unless it is independently supported by official behavior or ProjectChimera evidence.

For example, the video’s “dumb zone” description is a useful practitioner shorthand for degraded work in overlong sessions, but not an official Anthropic scientific claim. The official fact is that conversation history, instructions, file reads, tool descriptions, skill descriptions, etc. all share a finite context window, and Claude Code exposes `/context`, `/compact`, `/autocompact`, `/clear`, and subagents to manage it.

---

## 3. First decision: what kind of task is this?

Before writing a prompt, classify the task. Model, effort, tools, skills and workflow follow from this classification.

| Task type | Default model | Effort | Typical mode |
| --- | --- | --- | --- |
| Large autonomous implementation / research checkpoint | **Opus 5** | **xhigh** | normal agentic session or Ultracode if decomposition is genuinely useful |
| Independent adversarial scientific/code audit | **Fable 5** | **xhigh** | fresh session, read-only first, fresh-context verifier subagents |
| Exceptional one-shot highest-stakes audit/design | **Fable 5** | **max** if available and usage/time are secondary | deep independent review |
| Large refactor / complex systems engineering | **Opus 5** | **xhigh** | Plan Mode first; dynamic workflow only if large/decomposable |
| Ordinary feature implementation with moderate complexity | **Opus 5** | **high** | normal session |
| Focused bugfix / CI repair / localized tests | **Sonnet 5** | **medium or high** | normal session |
| Documentation / small mechanical cleanup | **Sonnet 5** | **medium** | normal session |
| Broad external research with little implementation | **Fable 5** | **high/xhigh** | research subagents / deep research where useful |
| Fast low-risk lookup/mechanical edit | **Sonnet 5** | **low/medium** | minimal tooling |

These are ProjectChimera defaults, not immutable Anthropic rules.

### Why Opus 5 is the normal executor

Anthropic’s current model guidance recommends Opus 5 for complex agentic coding, multihour autonomous coding, large-scale refactors, complex systems engineering and advanced research. It is therefore the default model that **builds** major Chimera checkpoints.

### Why Fable 5 is the normal independent reviewer

Fable 5 is Anthropic’s highest-capability widely released model and is a strong fit for deep reasoning, long-running agents and advanced research. ProjectChimera uses that extra capability where independence and falsification matter most: reconstructing a result, attacking assumptions, checking scientific chronology and deciding whether an implementation should be trusted.

This division also avoids the same model/configuration both creating and certifying the work whenever practical.

### Why Sonnet 5 still matters

Do not waste the heaviest model on every small change. Sonnet 5 is appropriate when the problem is already bounded and the main difficulty is implementation speed rather than novel scientific/system reasoning.

---

## 4. Effort selection

Effort controls how much inference/tool-use budget Claude spends. It affects reasoning, response tokens and tool calls.

### Project defaults

- `low`: trivial, highly constrained, low-risk work.
- `medium`: routine fixes, docs, simple test additions, mechanical refactors.
- `high`: normal quality-sensitive coding and analysis; also the API/default baseline for supported models.
- `xhigh`: demanding coding, agentic work, large audits, scientific checkpoints, complex repository-wide tasks.
- `max`: rare, one-shot work where maximum capability is worth unconstrained usage/latency.

### Rules

1. **Set effort deliberately at session start.**
2. Prefer `xhigh` for major Chimera implementation/audit sessions.
3. Do not use `max` by reflex. It is for rare tasks, not a “better xhigh” default.
4. Avoid switching effort repeatedly inside one long session. Anthropic notes that changing effort changes the rendered prompt and can break prompt-cache reuse.
5. If the task phase genuinely needs a different effort level, a new session is often cleaner than mutating a long-running cached conversation.

### Current P13 recommendation

- Executor: **Opus 5 + xhigh**.
- Independent post-result audit: **Fable 5 + xhigh**.

---

## 5. Ultracode / dynamic workflows

Ultracode is not a separate model. Anthropic describes it as a Claude Code setting that sets effort to `xhigh` and lets Claude decide when to create a dynamic multi-agent workflow.

Dynamic workflows can create a custom orchestration harness and fan work across many subagents. They are useful for genuinely broad tasks such as codebase-wide migrations, bug hunts, security audits and complex research/verification.

They can also consume **substantially more usage** than a normal Claude Code session.

### Use Ultracode when

- the task has several genuinely independent workstreams;
- parallel investigation will reduce wall-clock time or improve independence;
- the deliverable is high value enough to justify extra usage;
- one main context would otherwise be flooded by huge searches/logs;
- the task can benefit from independent finders and refuters.

Examples for Chimera:

- a major new end-to-end checkpoint with independent data/accounting/test/audit lanes;
- a repository-wide safety audit;
- a broad migration that touches many independent packages;
- independent reconstruction of evidence across many artifact families.

### Do not use Ultracode automatically when

- changing one or two files;
- fixing lint/CI;
- updating documentation;
- the task is inherently serial because scientific chronology matters;
- parallel agents would all inspect the same evidence and duplicate token spend.

### Important scientific rule

A workflow may parallelize **within a stage**, but it may not violate a serial research boundary.

Example:

```text
preregistration commit + push
        ↓
source acquisition / implementation / tests
        ↓
first economic result
        ↓
closure
        ↓
independent audit
```

No workflow may calculate the result before the preregistration commit exists merely because another branch of the workflow is “almost done” writing the preregistration.

---

## 6. Plan Mode

Plan Mode is a read/research-only mode until the plan is approved. In CLI it can be entered with `Shift+Tab`, `/plan`, or the plan permission mode. On web/mobile, use the corresponding mode control when available; do not tell a mobile user to rely on a physical keyboard shortcut.

### Use Plan Mode for

- unfamiliar repositories;
- large architectural changes;
- migrations/refactors with broad blast radius;
- tasks whose requirements are not already frozen;
- changes where the operator wants to inspect the approach before disk writes.

### Do not create an unnecessary approval bottleneck

For an autonomous Chimera research session with a very detailed prompt and a frozen design, the prompt may already serve as the approved high-level plan.

The agent should still inspect the repository before changing anything, but should not stop after every routine implementation decision to ask for permission if the answer is derivable from the repo and cannot change the scientific contract.

### Recommended pattern

```text
1. Scope/reconstruct repository state.
2. Identify fixed constraints and actual unknowns.
3. Produce a short implementation plan with verification points.
4. If the user explicitly requested Plan Mode/review, wait for approval.
5. Otherwise execute autonomously inside the frozen contract.
6. Verify after each load-bearing stage.
```

---

## 7. `CLAUDE.md`, auto memory and durable project knowledge

Claude Code starts each session with a fresh context window.

### `CLAUDE.md`

Official Anthropic guidance says project `CLAUDE.md` files are loaded at session start and should contain concise facts/rules that matter every session. Anthropic recommends keeping a `CLAUDE.md` around **under 200 lines** because longer files consume context and reduce adherence.

Therefore ProjectChimera uses:

- root `CLAUDE.md`: compact standing orders;
- `docs/claude_code_operating_guide.md`: detailed procedures loaded only for major work;
- `docs/current_development_plan*.md`: current project direction;
- checkpoint-specific preregistrations: scientific source of truth.

Do **not** dump every research result and historical detail into root `CLAUDE.md`.

### Auto memory

Claude Code auto memory is useful for recurring preferences and corrections, but it is not sufficiently durable for Chimera governance:

- it is environment/machine scoped;
- it may not exist in another cloud sandbox;
- it may differ across local vs cloud sessions;
- it is written by the agent itself.

Repository files are authoritative; auto memory is supplementary.

### `/init`

`/init` can generate/suggest a project `CLAUDE.md`, but ProjectChimera already has an intentionally curated file. Do not allow `/init` to blindly replace it.

---

## 8. Context-window management

The context window is a primary operational constraint.

At startup, Claude Code context can already include:

- system/tool instructions;
- `CLAUDE.md`;
- auto memory;
- MCP tool names/descriptions;
- Skill descriptions;
- subagent descriptions;
- user prompt.

As the session proceeds, file reads, command outputs, model messages and invoked Skill bodies add more.

### Project rule: new substantial task = new session

Do not continue a huge research session into an unrelated new checkpoint just because the old session “knows the repo.” The old context costs tokens every turn and can carry stale assumptions.

Start a new session and let repository documents restore the needed state.

### `/context`

Use `/context` when a session becomes long or tool-heavy. It shows what is consuming the window and gives optimization guidance.

### `/clear`

Use `/clear` or a new session when switching to unrelated work. Official Anthropic guidance explicitly recommends clearing between tasks because old conversation content crowds out the files needed next.

### `/compact`

`/compact` summarizes the conversation to free context. It is useful, but summarization can omit details.

For ProjectChimera:

- prefer a new session if a clean handoff is possible;
- if compaction is necessary, **always give focus instructions**;
- explicitly preserve:
  - current branch/head SHA;
  - preregistration SHA;
  - first-result boundary if applicable;
  - unresolved blockers/findings;
  - exact scientific decision rule;
  - P4-HOLD/Styx/live boundaries;
  - next required action;
  - files currently in flight.

Example:

```text
/compact preserve the exact Git chronology, preregistration hash/SHA,
scientific success criteria, unresolved audit findings, P4-HOLD/Styx/live
boundaries, and the next action. Drop routine command chatter and completed
file-reading details.
```

### `/autocompact`

Can be configured when a known long task is expected. Do not tune it casually during a sensitive research run without understanding what is preserved.

### `/rewind`

Can summarize or branch from selected conversation points. Useful when a wrong exploration polluted the current thread.

---

## 9. Side questions and steering

### `/btw`

`/btw` asks a side question without adding it to the main conversation history. This is excellent for questions such as:

- “why did you choose this accounting base?”
- “what file are you using as the source of truth?”
- “does this change the preregistration?”

Use it when the answer should not become a new instruction or pollute the active task context.

### Stop bad exploration early

The video’s practical advice is correct: if Claude is clearly reading irrelevant files or expanding into an unrelated research direction, stop/steer it early. A vague prompt can make an agent spend substantial context discovering what the operator could have specified in one sentence.

However, do not micromanage legitimate repository reconstruction merely to save tokens. For scientific work, reading the actual contract and evidence is not wasted exploration.

---

## 10. `/voice`

`/voice` enables voice dictation where supported. It can help the operator describe complex intent faster.

It does not improve reasoning by itself. The dictated prompt should still be structured before a high-stakes run.

For a major Chimera checkpoint, prefer writing/pasting the final reviewed prompt rather than relying on an improvised spoken instruction.

---

## 11. `/loop`

`/loop [interval] [prompt]` repeats a prompt while the session remains open.

Useful for:

- waiting for CI;
- checking whether a deployment/review completed;
- maintenance checks during an active session.

Do not treat `/loop` as durable, guaranteed multi-day automation if the session/environment may disappear.

For research chronology, never configure a loop that can automatically respond to a positive/negative result by changing parameters and re-running. That is automated post-selection.

---

## 12. Skills: what they are and how many to install

Skills are reusable `SKILL.md` procedures. Claude can invoke them automatically from their descriptions or explicitly by command.

Important context behavior:

- Skill **descriptions** are visible to Claude for discovery and therefore consume context.
- The full Skill body loads only when invoked.
- Large/old invoked Skills can eventually be truncated/dropped under context pressure.

### More Skills is not automatically better

Claude can technically operate with many Skills, but ProjectChimera should optimize for clarity rather than catalog size.

Problems with installing too many overlapping Skills:

- more startup metadata/context;
- ambiguous automatic routing;
- contradictory workflows;
- larger third-party supply-chain/security surface;
- generic “best practice” procedures competing with scientific chronology;
- more opportunities for a Skill to trigger when it is not wanted.

### Project heuristic: 3–6 Skills per major task

This is **our heuristic, not an Anthropic hard limit**.

Aim for one Skill per distinct capability. For example:

```text
external primary-source research
trading/backtest methodology
dimensional/accounting verification
Python testing
```

Four cleanly separated Skills are usually better than twenty overlapping “research/reviewer/coder/planner” Skills.

### Before installing a third-party Skill

1. Read its `SKILL.md`.
2. Check what commands/tools/scripts it may execute.
3. Check dependencies and related Skills.
4. Check security/audit signals if the catalog provides them.
5. Confirm it does not conflict with ProjectChimera Git/scientific rules.
6. Prefer project-scoped installation for Chimera-specific workflows where possible.

A green catalog security badge reduces risk; it does not make arbitrary third-party instructions trustworthy.

### Current P13 candidate set

Revalidate these at the moment of installation because community packages can change:

1. `research` — Matt Pocock / AIHero ecosystem
   - purpose: primary-source external research;
   - useful for Binance/API/market-semantics investigation.

2. `backtest-expert` — TraderMonty
   - purpose: backtest methodology, friction, robustness, failure-oriented evaluation.

3. `dimensional-analysis` — Trail of Bits
   - purpose: detect unit/scaling mistakes, especially quantity/notional/leverage/fees/PnL.

4. `python-testing-patterns` — wshobson
   - purpose: pytest patterns, positive/negative controls, parametrization and test design.

### Skills intentionally not defaulted

Do not install generic orchestration/planning Skills just because they exist if the main prompt already controls chronology. A Skill that manages its own commit lifecycle, blocks `git push`, or automatically “optimizes” thresholds can actively conflict with a preregistered research session.

### Precedence rule

```text
frozen repository contract
> explicit task constraints
> project standing instructions
> generic Skill recommendations
```

A Skill never gets authority to reinterpret evidence.

---

## 13. Subagents

Anthropic recommends subagents when a side task would flood the main context. Each subagent receives its own context and returns a summary.

### High-value uses in Chimera

- independently recompute a published metric;
- audit data provenance;
- audit accounting with a hand implementation;
- inspect safety/live reachability;
- search large code areas;
- independently challenge a proposed scientific interpretation.

### Fresh-context verification

For a load-bearing conclusion, the second reviewer should not merely read the first reviewer’s reasoning and agree. Give it the primary artifacts/code/contract and ask it to reconstruct the result independently.

### Do not over-parallelize

Parallelism is useful but expensive. Several subagents reading the same 50 files are not independent information; they can simply be duplicated token spend.

Project heuristic:

- 1–2 targeted subagents: ordinary complex implementation;
- 2–4 independent lanes: major checkpoint implementation/verification;
- 5–7: exceptional adversarial external audit where independence is the deliverable;
- tens/hundreds: only through a justified dynamic workflow on a task that truly decomposes.

These counts are heuristics, not product limits.

### Critical nuance: built-in Explore/Plan agents

Current Anthropic docs say built-in Explore and Plan agents skip the parent session’s `CLAUDE.md` and git status to keep research fast/inexpensive.

Therefore **never assume a critical subagent inherited ProjectChimera safety/science rules**.

When delegating a critical task, restate essential constraints in the subagent prompt, e.g.:

```text
Do not read P4-HOLD or Styx. Do not calculate post-cutoff economics. This is a
read-only independent audit. Use committed artifacts, not PR prose. Report
unverified claims explicitly.
```

A custom project subagent can be used when repeated high-value auditing benefits from persistent scoped instructions.

---

## 14. MCP and external integrations

MCP connects Claude Code to external tools/data.

Use MCP when direct access materially reduces copy/paste or enables a required action, e.g. GitHub, issue trackers, monitoring, databases.

### Keep MCP narrow

Every extra tool/server increases:

- tool-selection surface;
- startup context from names/descriptions;
- permission/security surface;
- potential for accidental irrelevant calls.

For a Chimera research session, enable only what is required.

Example:

- GitHub access: useful.
- Unrelated design/CRM/database MCPs: disable unless the task genuinely needs them.

### Cloud/mobile note

Claude Code on the web runs in Anthropic-managed cloud infrastructure. A local Remote Control session runs on the user’s machine and can expose local MCP servers/files.

Do not assume local MCP availability in a cloud/mobile Code session.

---

## 15. Cloud / web / mobile vs local / Remote Control

This distinction matters for prompts.

### Claude Code on the web/mobile cloud

- repo is cloned into a cloud sandbox;
- sessions can be started from browser/mobile;
- do not assume the user’s local shell, local filesystem or local environment variables;
- machine-local hacks/configuration are inappropriate unless supported by the cloud environment;
- durable project context should be committed to the repo.

### Remote Control

Remote Control is a mobile/web front end to a Claude Code session that continues running on the user’s own machine. Local filesystem, MCP servers and project configuration remain available.

### Project default assumption

Unless the user explicitly says otherwise, assume ProjectChimera work is being launched from **Claude Code web/mobile cloud**, not from a local CLI.

Therefore do not give instructions such as custom shell environment concurrency flags or local daemon setup unless needed and explicitly compatible with the chosen surface.

---

## 16. Prompt-writing standard for major Chimera tasks

A large prompt should be detailed where **constraints matter**, not bloated with facts the agent can cheaply rediscover.

### Good major prompt structure

```text
1. Role / objective
2. Repo and known starting SHA
3. Non-negotiable boundaries
4. Required preflight reconstruction
5. Scientific question / acceptance criteria
6. Data/source contract
7. Accounting/economic semantics
8. Chronology requirements
9. Implementation requirements
10. Tests / positive controls / verifiers
11. Independent audit requirements
12. Git / PR policy
13. Final report format
```

### Tell the agent what it must rediscover

Instead of feeding every previous result as truth, say:

```text
Read the current main branch and independently verify the state. Do not trust
PR summaries when repository artifacts and Git history can answer the question.
```

This reduces anchoring and stale-summary errors.

### But explicitly state non-negotiable safety/science boundaries

Do not expect an agent to infer “never open Styx” from 50 documents while simultaneously giving it permission to research broadly. Critical boundaries belong in the prompt even if they also exist in repo docs.

### Do not over-micromanage modern frontier models

For Opus/Fable 5, avoid long instructions that specify every internal thought process or trivial shell action.

Specify:

- objective;
- evidence standard;
- immutable constraints;
- exact chronology;
- what success/failure means;
- required verification;
- when to stop.

Let the model choose routine implementation mechanics.

### Explicitly allow negative outcomes

A research prompt should always say that a negative result is acceptable and should be closed rather than rescued.

### Explicitly state stop conditions

Examples:

- stop if source semantics are ambiguous and profitability would be needed to choose between designs;
- stop if a discovered defect changes already-frozen economic evidence;
- stop after a positive screen instead of automatically escalating to live/paper;
- stop after a negative screen instead of generating v2.

---

## 17. Prompt-writing checklist for ChatGPT / another planner

Before giving the user a new Claude prompt, always answer these questions first:

### A. Model

Choose explicitly among the available:

- Opus 5
- Sonnet 5
- Fable 5

State the reason.

### B. Effort

Choose explicitly:

- low
- medium
- high
- xhigh
- max (only if justified/available)

### C. Session type

Choose:

- new clean session;
- existing session continuation;
- Plan Mode;
- normal autonomous mode;
- Ultracode/dynamic workflow.

Default for a new research checkpoint: **new clean session**.

### D. Skills

Before the prompt, decide whether Skills materially help.

- Prefer 3–6 non-overlapping task-specific Skills for major work.
- Re-check community Skill content/security if it may have changed.
- Do not install a Skill that conflicts with Git/scientific chronology.

### E. MCP/tools

Enable only tools needed for the task.

### F. Context/handoff

Ensure the prompt tells Claude which repo documents are authoritative, but does not paste all of them into the conversation unnecessarily.

### G. Verification

Specify tests, independent controls, artifact reconstruction and final audit requirements before implementation begins.

### H. Git chronology

If the task is research, explicitly define which commit must precede the first result.

### I. Merge policy

Large research PR: normally **draft, do not merge** until independent review.

Routine docs/CI repair: merge may be allowed after green CI if explicitly authorized by the user.

---

## 18. Verification: the agent must prove the work

The video emphasizes self-verification; this aligns strongly with ProjectChimera.

### Minimum software verification

Depending on scope:

- unit tests;
- integration tests;
- lint;
- static/type checks used by the repo;
- pre-commit;
- config validation;
- end-to-end smoke;
- CI.

### Scientific verification

Additionally:

- preregistration hash/SHA chronology;
- no result before the prereg commit;
- data-boundary checks;
- no sealed/retired evidence access;
- causal timestamp checks;
- two-sided synthetic positive/negative controls;
- independent recomputation from primary evidence;
- explicit sample/trade counts;
- artifact manifests/checksums;
- no post-result threshold/model shopping.

### Accounting verification

Never trust a profitable backtest because tests are green.

Hand-trace units:

```text
capital
→ exposure
→ leverage
→ quantity
→ notional
→ fills
→ fees
→ funding
→ realised/unrealised PnL
→ equity
```

For a multi-leg strategy, verify both legs and the **total capital denominator**.

### Independent audit

For major research work, the author model does not get the last word. A fresh Fable 5 session should independently reconstruct critical claims before merge.

---

## 19. Git workflow for scientific tasks

Git history is part of the evidence.

### Required pattern

```text
clean base
→ design/preregistration commit
→ PUSH
→ implementation/acquisition if permitted by the design
→ first result commit
→ closure
→ audit/remediation
→ draft PR
→ independent external review
→ merge only after explicit authorization
```

### Do not

- amend away the preregistration boundary;
- squash preregistration and result into one commit;
- force-push rewritten chronology;
- regenerate primary result evidence silently after closure;
- hide failed experiments merely because they are inconvenient.

---

## 20. What to do when context/usage limits are tight

The Coding Sloth video correctly highlights that subagents, long context and frontier models can consume plan usage quickly.

Project response should be **prioritization**, not weakening scientific controls.

Order of savings:

1. new clean session instead of carrying irrelevant history;
2. precise prompt;
3. fewer irrelevant Skills/MCP servers;
4. delegate huge searches to one focused subagent instead of reading everything into main context;
5. use Sonnet for routine subproblems where quality is sufficient;
6. use medium/high rather than xhigh on genuinely simple work;
7. use focused `/compact` only when continuation is better than a fresh handoff;
8. avoid redundant parallel auditors;
9. reserve Ultracode for high-value decomposable tasks.

Never save tokens by:

- skipping boundary verification;
- weakening accounting checks;
- dropping preregistration chronology;
- trusting an author’s summary instead of evidence;
- merging without needed independent audit.

---

## 21. Common failure modes and responses

### Failure: one giant session for everything

Response: new task → new session. Repo docs carry state.

### Failure: vague prompt causes giant exploration

Response: specify objective, authoritative files, constraints, deliverables and stop conditions.

### Failure: prompt contains every historical number and anchors the reviewer

Response: provide known starting SHA and boundaries; require independent reconstruction.

### Failure: 30 overlapping Skills

Response: choose 3–6 distinct capabilities; remove overlaps.

### Failure: subagents all repeat the same analysis

Response: assign orthogonal responsibilities or fresh independent reimplementations.

### Failure: Plan Mode blocks an overnight autonomous task waiting for trivial approval

Response: use Plan Mode for unresolved design; once the scientific contract is frozen, authorize autonomous execution within it.

### Failure: `/compact` causes loss of chronology

Response: use focused compact instructions preserving SHAs, boundaries, unresolved findings and next step — or start a clean session from repo state.

### Failure: a Skill suggests tuning after a negative result

Response: repository contract outranks Skill; close negative checkpoint.

### Failure: model that built the work also certifies it

Response: independent Fable audit from clean context.

### Failure: local-shell advice in a mobile cloud session

Response: first identify platform. Prefer repo-level configuration and cloud-supported controls.

### Failure: “CI green” treated as economic proof

Response: independently check decision logic, causal semantics and accounting.

---

## 22. Recommended operating pattern for a major new checkpoint

### Phase 0 — planner prepares the launch

1. Read this guide and current project plan.
2. Re-check official Anthropic model/effort/features if the decision may have changed since this file’s review date.
3. Choose model.
4. Choose effort.
5. Choose 3–6 useful Skills or none.
6. Decide normal vs Ultracode.
7. Write the prompt with chronology, boundaries and stop conditions.

### Phase 1 — executor

Default: **Opus 5 + xhigh**, clean session.

- reconstruct main;
- inspect relevant contracts;
- freeze/preregister;
- push prereg commit;
- implement;
- run exactly the frozen experiment;
- close honestly;
- self-audit;
- open draft PR;
- do not merge.

### Phase 2 — independent reviewer

Default: **Fable 5 + xhigh**, separate clean session.

- do not trust executor summary;
- reconstruct Git chronology;
- recompute key numbers;
- probe boundaries;
- audit accounting;
- find missing two-sided tests;
- state exact interpretation;
- issue GO/NO-GO.

### Phase 3 — integration

- repair only findings that do not rewrite frozen economics;
- if a defect invalidates economics, stop and preserve the integrity issue;
- merge only after explicit owner authorization.

---

## 23. Current ProjectChimera-specific decisions

At the time this guide was written:

- P6/P6-EXT/P7 are closed negative under their exact contracts.
- P8 is preregistered but not opened.
- P4-HOLD is retired/unread.
- Styx is sealed with a disclosed hindsight-era ceiling.
- Chimera Futures remains dry-run only; legacy Freqtrade has separately gated live capability but that is not permission to trade.
- the next selected research axis is an exploratory structural BTC spot/perpetual funding/basis carry feasibility screen.
- the intended executor for that checkpoint is **Opus 5 + xhigh**.
- the intended independent reviewer is **Fable 5 + xhigh**.
- current candidate Skills are `research`, `backtest-expert`, `dimensional-analysis`, `python-testing-patterns`.

These project-state facts can become stale. The current development plan and Git/artifact tree outrank this section.

---

## 24. Update policy

Claude Code changes quickly. Commands, models, workflow features and product limits can become stale.

Before a new major prompt is finalized, re-check official Anthropic documentation when the decision depends on:

- current model positioning;
- available effort levels;
- Ultracode/dynamic workflow behavior;
- cloud/mobile feature availability;
- Skill/subagent context behavior;
- new commands;
- plan/usage restrictions.

Do not rewrite this manual every week for cosmetic product changes. Update it when the change would materially alter how ProjectChimera should launch, prompt, verify or audit Claude Code work.

---

## 25. One-page launch checklist

Before sending a major Claude Code prompt, be able to fill this out:

```text
TASK:

MODEL: Opus 5 / Sonnet 5 / Fable 5
WHY:

EFFORT: low / medium / high / xhigh / max
WHY:

SESSION: new / continue
MODE: normal / Plan / Ultracode
WHY:

SKILLS:
1.
2.
3.
4.
WHY EACH:

MCP/EXTERNAL TOOLS ENABLED:

AUTHORITATIVE REPO DOCS TO READ:

NON-NEGOTIABLE SCIENTIFIC/SAFETY BOUNDARIES:

SERIAL GIT/PREREGISTRATION BOUNDARY:

WHAT CAN RUN IN PARALLEL:

WHAT MUST BE INDEPENDENTLY RECOMPUTED:

STOP CONDITIONS:

VERIFICATION BEFORE PR:

PR POLICY: draft / non-draft
MERGE POLICY:

POST-AUTHOR INDEPENDENT REVIEW MODEL:
```

If these fields cannot be answered, the prompt is not ready yet.
