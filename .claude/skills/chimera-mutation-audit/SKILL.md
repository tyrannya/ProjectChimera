---
name: chimera-mutation-audit
description: Owner-initiated mutation audit of a ProjectChimera fix - restore the defect and confirm the witnessing tests fail - without leaving any mutated source behind or racing ordinary test runs.
argument-hint: <PR number or test module>
disable-model-invocation: true
---

# chimera-mutation-audit

`CLAUDE.md` and `docs/agent_workflow_policy.md` govern. Mutations are temporary.
The checked-out tree must be byte-identical before and after, and nothing mutated is ever committed.

1. **Name the claim**: which defect the fix removed and which tests are said
   to witness it.
2. **Prefer in-test mutation.** A monkeypatch that restores the old behaviour
   inside the test process rewrites no source. See the witness tests in
   `tests/test_r1b_persisted_equity.py`. Each witness must fail under the restored
   defect and pass on the fixed code.
3. **Source-rewriting tools** (e.g. mutmut) run only in an isolated copy,
   never in the working tree. Never run them concurrently with ordinary tests
   in the same worktree: a suite running beside a rewritten file reports noise
   as regressions.
4. **Record** each mutant, the command run, and whether it was killed or survived.
   A surviving mutant on the claimed defect is a finding.
5. **Restore and prove it.** Check that `git status --porcelain` is empty and
   that `git diff` is unchanged from the start. Remove the isolated copy.

Report findings to the owner. The audit fixes nothing itself: remediation is a
separate `/chimera-remediate` round.
