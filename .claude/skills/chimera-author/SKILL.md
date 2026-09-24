---
name: chimera-author
description: Author a new governed ProjectChimera change (implementation or research checkpoint) from a clean base to a draft PR, preserving preregistration-before-result chronology and append-only Git history. Use when starting substantive work that will commit to the repository.
argument-hint: <task or phase id>
---

# chimera-author

Workflow for authoring. It adds no authority. `CLAUDE.md`,
`docs/agent_workflow_policy.md` and the frozen contract for the task always outrank it.

1. **Preflight.** Record branch, `HEAD`, `origin/main` and `git status`. Work only
   in the worktree and branch the task names, and start from a clean, reconstructible base.
2. **Governing documents.** Read `docs/current_development_plan.md`, the roadmap
   it points to (`docs/master_roadmap_r0_r18.md`), `docs/research_roadmap.md`,
   and every preregistration or closure the task touches. Trust repository
   evidence over PR prose, chat summaries and memory.
3. **Plan before editing.** Set out scope and non-scope, invariants, the
   evidence boundary, the verification points, the Git/PR consequences and
   the stop conditions. Do not calculate or read a governed result during planning.
4. **Chronology.** For research, the preregistration commit is **pushed**
   before the first result it governs is computed or read. Never change
   thresholds, models, costs, folds, horizons, venues or success rules after
   seeing a result. Negative results stay visible.
5. **Implement surgically.** Touch only what the task requires, and match
   the surrounding style.
6. **Test both directions.** Every guard or decision gets a positive and a
   negative control. Accounting changes get a hand trace of units, notional,
   fees, funding and PnL.
7. **Commit append-only.** Use ordinary commits and a normal `git push`. Never
   amend, rebase, squash, reset published history or force-push. The guardrails
   deny these, and a deny means stop and report, not reroute.
8. **Draft PR.** Open it as a draft, with the evidence and the limits stated.
   Never merge; merging is an owner decision.
9. **Verify.** Run `/chimera-verify` before claiming anything is done. Your
   own checks are not independent acceptance.

Boundaries that always hold:
- P4-HOLD stays unread and Styx sealed.
- No live or real-money reachability.
- Aegis stays the sole risk authority.
- The serving model is fixed.
