---
name: chimera-remediate
description: Remediate findings from an independent ProjectChimera review on an existing PR branch, one finding at a time, with appended commits and a finding-to-commit map. Use when a review has returned REQUEST CHANGES or new findings.
argument-hint: <PR number>
---

# chimera-remediate

Remediation workflow. `CLAUDE.md`, `docs/agent_workflow_policy.md`, and the PR's
own frozen contract outrank this file. So does the contract of the checkpoint it serves.

1. **Pin the state.** Record the PR, its branch, the head SHA, and the SHA the
   review was taken against. If they differ, find out why before touching anything.
2. **Read the primary review record**, meaning the comment or review itself, not a
   summary of it. List every finding with its id and severity.
3. **Classify each finding**:
   - **fix**: in scope, and the defect is real;
   - **dispute**: state the evidence (file:line, a command and its output) that the finding is wrong. Assertion alone doesn't count;
   - **out of scope**: name where it belongs. It is not silently dropped.
4. **Stop instead of fixing** when a fix would change frozen evidence, a
   preregistered rule, thresholds, costs, folds or success criteria after a
   result. Stop too when the defect invalidates already-frozen economics.
   Preserve the integrity issue and report it to the owner.
5. **Fix with appended, ordinary commits**, one finding or a tight group per
   commit. Add a regression test that fails on the pre-fix code and passes after
   it. Never amend, rebase, squash or force-push; the review must be able to diff
   the old head against the new one.
6. **Map findings to commits** in the PR record (`gh api -X PATCH` if `gh pr edit`
   fails), with the verification each fix received.
7. **Re-verify.** Run `/chimera-verify` on the new head. CI on the old head
   certifies nothing about the new one.
8. **Hand back.** The PR stays a draft. Only a fresh independent
   `/chimera-delta-review` can close findings. The author never retires a
   reviewer's finding and never merges.
