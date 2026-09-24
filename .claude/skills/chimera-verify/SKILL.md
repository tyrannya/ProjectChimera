---
name: chimera-verify
description: Verify a ProjectChimera change before any completion claim, keeping three tiers separate - local tests, exact-head CI, and independent acceptance - and reporting what is verified, inferred, blocked or out of scope.
argument-hint: "[PR number]"
---

# chimera-verify

Read-only verification. `CLAUDE.md` and `docs/agent_workflow_policy.md` §9 govern.
A task is not done because code was written, and the three tiers below are
never merged into one "green".

**Tier 1: local tests** (evidence from the author's machine)
- Record the SHA, `git status` (is the tree dirty?), the interpreter path and its version.
- Run what the change touches: targeted tests with both positive and negative controls, `python -m compileall -q .`, and `python -m pre_commit run --all-files`.
- Run the repository verifiers the change can affect: `python -m tools.verify_research_snapshot`, `tools.verify_trade_snapshot`, `tools.verify_research_state`, and `tools.freeze_evidence --verify` on every manifest.
- Run the full `python -m pytest` when behaviour may have changed.
- Report failures with their output. Never excuse a failure as environmental without evidence.

**Tier 2: exact-head CI** (evidence about one commit)
- `git rev-parse HEAD`, the remote branch head, and the PR `headRefOid` must be the same SHA.
- Every required check (lint, and tests on both OS legs) must be green on that SHA.
- A run on an older head certifies nothing about the new one. A cancelled or pending run is not green.

**Tier 3: independent acceptance** (not something the author can supply)
- Only a separate reviewer session (`/chimera-review` or `/chimera-delta-review`) or the owner can provide it.
- The author's self-review, and the author's subagents, are not independent acceptance.
- Report it as "pending" unless the review record exists for the exact head.

**Report:** a table with each claim marked verified, inferred but not verified, blocked, or out of scope. Tier 2 and Tier 3 each get their own line. Never merge.
