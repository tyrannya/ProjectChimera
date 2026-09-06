# ProjectChimera active context

> Navigation / handoff only. This file is **not authoritative**. Current Git state,
> adopted governance documents, frozen research contracts and executable guardrails
> always override it. Verify every load-bearing claim before acting.

## Last verified repository snapshot

- Upstream `main` before this agent-context PR: `e02b871adf3e43437d7f84b17c0f81399f39143c`.
- PR-12 (#85) and PR-13 (#86) are merged.
- A11 Spot WebSocket transport correction is merged.
- Recorder contract remains `btcusdt-prospective-gen3` with
  `prospective_from = null`.

## Current roadmap position

- S3 is **not yet complete** because the demo runner does not yet make all adopted
  campaign evidence paths reachable.
- Immediate engineering task: PR-10R, completing runner funding, reconciliation,
  liquidation-touch and required recovery/stale evidence semantics, followed by
  48-hour synthetic acceptance and replay-parity verification.
- PR #76 / roadmap PR-06 remains a separate acceptance line. It still needs two
  genuine full UTC recorder days; those engineering days are not prospective S1
  evidence.
- S1 has not started.
- PR-14 / S2 scientific preregistration has not been authored.
- S4 has not started.

## Session handoff rule

At the end of a substantial task that changes durable repository state, update this
file narrowly with:

1. the latest verified main SHA (or explicitly say the work is still an unmerged PR);
2. merged/open PRs that materially affect the active roadmap;
3. unresolved load-bearing blockers;
4. the immediate next action.

Do not turn this file into a chronological diary. Durable engineering decisions go
in `docs/decisions/`; scientific/governance decisions stay in their adopted plan,
contract or amendment documents.
