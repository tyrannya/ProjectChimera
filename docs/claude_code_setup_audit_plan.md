# ProjectChimera — Claude Code Setup Audit Plan

Status: **PROPOSED TOOLING AUDIT — READ ONLY FIRST**.

This document defines how to review Claude Code project configuration, hooks, guards, MCP use, and subagent setup without silently changing ProjectChimera behavior.

## Objective

Determine whether additional Claude Code configuration could reduce operator error, wasted usage, unsafe Git actions, stale-context mistakes, or scientific chronology violations.

## Mandatory mode

Start in **Plan Mode / read-only**.

Do not install plugins, create hooks, change permissions, modify Git configuration, or write new agent files during the audit itself.

## Inspect

- `CLAUDE.md` size/clarity and conflicting instructions;
- `.claude/settings.json` and local/user precedence implications;
- project subagents and their tool permissions;
- hooks currently present or absent;
- dangerous Git commands that could erase scientific chronology;
- commands that could cross no-live/credential boundaries;
- mutation/test concurrency hazards;
- whether exact-head CI and branch/SHA checks can be made harder to skip;
- whether preregistration-before-result barriers can be guarded mechanically;
- whether source/acquisition/live paths need command guards;
- Context7/MCP surface and whether it adds unnecessary context/action surface;
- whether any proposed guard would create false positives that brick ordinary engineering.

## Candidate guard families to evaluate

1. **Write guard**
   - protect frozen preregistrations/evidence areas from accidental edits;
   - must have an explicit legitimate amendment path.

2. **Git chronology guard**
   - detect/deny amend/rebase/squash/force-push patterns when they would erase governed chronology;
   - must not block harmless ordinary Git inspection.

3. **Scientific-stage guard**
   - prevent first-result commands before the required preregistration commit/push boundary where mechanically identifiable.

4. **Live/acquisition guard**
   - prevent authenticated/live expansion or credential-bearing commands unless a future explicit contract opens them.

5. **Completion verifier**
   - a `/verify-done`-style procedure that reconstructs branch/head, diff, tests, verifiers, artifacts, and exact-head CI before completion claims.

6. **Mutation/test concurrency guard**
   - prevent known source-rewriting mutation workflows from running concurrently with ordinary suites in the same worktree.

## Required adversarial tests for any proposed guard

For every guard, test both directions:

- it blocks the unsafe action;
- it allows the legitimate safe action.

A guard that can brick healthy work without a supported recovery path is not acceptable merely because it is conservative.

## Deliverable

Produce a recommendation table:

| candidate | problem prevented | exact mechanism | false-positive risk | bypass/recovery | tests | recommend? |
|---|---|---|---|---|---|---|

Then propose the **minimum** separate governance PR needed. Do not mix the implementation into an active scientific/remediation PR.
