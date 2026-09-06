# Known blockers and non-blocking limitations

> Handoff index only, not authority. Re-verify against current `main`, the adopted
> master plan, open PRs and executable tests before acting.

## Load-bearing blockers

### D1 — demo runner campaign-evidence paths

Current S3 blocker recorded by PR-12: the runner does not yet make all adopted
campaign evidence paths reachable. In particular, funding, reconciliation and
liquidation-touch behaviour require a PR-10 follow-up with deterministic
persistence/replay semantics and synthetic witnesses.

**Owner / next action:** PR-10R.

### PR-06 acceptance

Roadmap PR #76 remains separate from S3. Merge acceptance requires reconciliation
of two genuine full UTC recorder days against published minute-indexed data. Do
not manufacture, relabel or backfill recorder observations to satisfy it.

**Owner / next action:** persistent recorder host + two genuine UTC days, then
PR-06 acceptance review.

### S1 source readiness

The official S1 30-day gate additionally needs the adopted funding/source path to
be operational. A transport/network environment that can satisfy PR-06 minute
reconciliation is not automatically sufficient for S1 funding completeness.

**Owner / next action:** verify the eventual persistent S1 host against every
contract-required source before activation.

## Known limitations that are not current S3 blockers

- `conf/base.json` cleanup/classification is deferred to PR-16.
- Some futures cost metrics do not provide per-leg attribution; PR-12 records this
  as an observability limitation.
- Scientific classification of genuinely unclassified decision-log kinds belongs
  to PR-14/S2 unless the adopted plan already settles the case.

## Rules for this file

- Remove a blocker only when repository evidence proves it closed.
- Do not convert a limitation into a blocker without tracing the governing
  acceptance criterion.
- Do not add speculative future work.
