---
name: chimera-review
description: Owner-initiated, READ-ONLY independent review of a ProjectChimera PR or branch pinned to an exact head SHA, reconstructed from primary evidence, ending in the repository's review verdict vocabulary.
argument-hint: <PR number or branch> [head SHA]
disable-model-invocation: true
---

# chimera-review

**READ-ONLY.** No edits, commits, pushes, branch or PR-state changes, and no
remediation. Post a verdict as a PR comment only when the owner asks for it.
`CLAUDE.md` and `docs/agent_workflow_policy.md` govern.

**Independence.** This review counts as independent acceptance only when it
runs in a fresh session that did not author the work. An author's own
invocation of this skill, or any subagent the author spawns, is
never independent acceptance. Report such a run as a self-check.

1. **Pin the exact head SHA** and base. Every finding and the verdict name the SHA.
2. **Reconstruct from primary evidence**: the diff, code, tests, artifacts,
   manifests and Git history. Do not use PR prose or the author's summary. If
   a claim can't be reproduced, record it as unverified.
3. **Chronology.**
   - Check with `git log` that preregistration commits precede the results they govern and were pushed first.
   - Published history must be append-only.
   - Nothing may have been rescued after a result.
4. **Boundaries.**
   - P4-HOLD stays unread and Styx sealed.
   - No live or real-money reachability.
   - Aegis stays the sole risk authority.
   - Scope matches the task contract.
5. **Tests.**
   - They must be two-sided, and a positive and a negative control must exist for each guard.
   - Run the relevant suites yourself.
   - Check exact-head CI: CI on another head does not count.
6. **Accounting**, if touched: hand-trace units, notional, fees, funding,
   margin and PnL for representative LONG, SHORT and multi-leg cases.
7. **Optional subagents** (`repo-verifier`, `accounting-auditor`,
   `safety-auditor`) inherit the serving model and receive the boundaries
   above verbatim. Their output is evidence for you to reconcile, not a verdict.

**Findings:** each has an id, a severity (blocker, medium, low or informational), file:line, evidence, and whether it was reproduced.

**Verdict for an engineering PR**: exactly one of
- `APPROVED — <scope>`: may recommend an owner ordinary merge; never merges;
- `REQUEST CHANGES — PR #<n> NOT READY`: lists the blocking findings;
- `BLOCKED / CANNOT VERIFY`: names what could not be reconstructed and why.

An independent audit of a *research checkpoint* uses the GO/NO-GO verdict of
`docs/claude_code_operating_guide.md` §22, and only for that review type.
