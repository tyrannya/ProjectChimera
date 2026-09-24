---
name: chimera-reproduce
description: Owner-initiated independent reproduction of an already-published ProjectChimera result from primary evidence at an exact SHA, compared against the frozen SHA256SUMS manifests, without committing anything.
argument-hint: <checkpoint or artifact> <SHA>
disable-model-invocation: true
---

# chimera-reproduce

`CLAUDE.md`, `docs/agent_workflow_policy.md` and the checkpoint's preregistration govern.
No committed changes: regenerated output is compared, never committed or
substituted for frozen evidence.

1. **Eligibility.** Only reproduce a result that is already published and
   whose preregistration barrier has been crossed.
   - Never read P4-HOLD, open Styx, or compute a result a future preregistration still governs.
   - If eligibility is unclear, stop and ask the owner.
2. **Pin the source.** Use the exact SHA with a clean tree. A dirty tree makes
   the reproduction meaningless. Record the interpreter and dependency versions.
3. **Run the documented command only**, from `docs/research_reproduction.md` or
   the checkpoint's own document or Makefile target. Change no parameter, fold,
   cost, seed or threshold.
4. **Compare against the freeze.** Use
   `python -m tools.freeze_evidence --verify <manifest>` for committed evidence,
   then compare the regenerated artifacts value by value with the committed ones.
5. **Report** match, mismatch, or not reproducible, with the exact commands,
   SHAs and differing values. A mismatch is a finding for the owner. It is never
   "fixed" by regenerating the evidence.
6. **Clean up.** Delete scratch output, and confirm `git status` is exactly as
   before the run.
