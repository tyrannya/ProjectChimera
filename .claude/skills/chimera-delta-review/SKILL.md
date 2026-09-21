---
name: chimera-delta-review
description: Owner-initiated, READ-ONLY delta re-review of a ProjectChimera PR from a previously reviewed SHA to its new head, checking that each prior finding is closed and nothing else drifted.
argument-hint: <PR number> <previously reviewed SHA> [new head SHA]
disable-model-invocation: true
---

# chimera-delta-review

**READ-ONLY.** No edits, commits, pushes, branch or PR-state changes, and no
remediation. Post the verdict as a PR comment only when the owner asks for it.
`CLAUDE.md` and `docs/agent_workflow_policy.md` govern.

**Independence.** Same rule as `/chimera-review`. An author's own invocation,
or any subagent the author spawns, is never independent acceptance.

1. **Pin both SHAs**: the previously reviewed SHA from the standing review
   record, and the exact new head.
2. **Append-only check.** Run `git merge-base --is-ancestor <old> <new>`. If
   the old SHA is not an ancestor, history was rewritten. That is itself a
   blocking finding, and the delta cannot be trusted as a delta.
3. **Read the delta**: `git log --oneline <old>..<new>` and
   `git diff <old>..<new>`. Every changed line must trace to a finding or a
   stated, in-scope reason.
4. **Close findings one by one.** For each prior finding, name the commit
   that addresses it. Reproduce the fix yourself. Confirm that a regression
   test fails without the fix, for example by a monkeypatch that restores the
   defect.
5. **Drift.** Nothing outside the findings changed behaviour, evidence, frozen
   contracts or scope. Untouched areas keep their earlier verdict only if the
   diff does not reach them.
6. **Exact-head CI.** Checks must be green on the new head SHA itself.

**Verdict**: exactly one of
- `DELTA APPROVED`: every prior finding is closed, there is no regression and no drift;
- `REQUEST CHANGES`: lists the open or new findings with evidence;
- `BLOCKED / CANNOT VERIFY`: names what could not be established, e.g. rewritten history, missing CI or an unreproducible fix.
