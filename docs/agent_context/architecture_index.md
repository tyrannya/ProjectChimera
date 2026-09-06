# ProjectChimera architecture index

> Fast navigation for agents. This index is not a specification and must not be
> used to override the files it points to.

## Authority / roadmap

1. `CLAUDE.md` — standing execution and safety instructions.
2. `docs/proposed_development_plan_post_fable_5_1_audit.md` — adopted roadmap.
3. `docs/proposed_demo_implementation_master_plan.md` — adopted engineering plan.
4. Adopted amendments under `docs/amendment_*.md` — narrow superseding governance.
5. Frozen research / recorder contracts and executable guardrails.

## Active demo path

```text
public market data
  -> chimera.recorder
  -> normalized filesystem
  -> chimera.demo.feed
  -> chimera.demo.runner / rules
  -> chimera.risk (Aegis)
  -> chimera.carry
  -> chimera.futures dry-run venue
  -> chimera.demo.decision_log
  -> replay parity / reports / observability
```

Useful entry points:

- Recorder: `chimera/recorder/`, `tools/recorder.py`, `tools/recorder_preflight.py`
- Demo feed: `chimera/demo/feed.py`
- Demo runner: `chimera/demo/runner.py`
- Rules: `chimera/demo/rules*.py`
- Risk: `chimera/risk.py`
- Carry: `chimera/carry/`
- Dry-run execution: `chimera/futures/`
- Decision log: `chimera/demo/decision_log.py`
- Replay parity: `tools/replay_parity.py`
- Reports: `chimera/demo/reports.py`, `tools/demo_report.py`
- Observability: `chimera/demo/telemetry.py`, `chimera/metrics.py`,
  `conf/alerts_demo.yml`, `grafana/provisioning/dashboards/demo.json`
- Deployment / operations: `docker-compose.yml`, `deploy/`,
  `docs/demo_runbook.md`

## Historical / disconnected runtime

The former Freqtrade / inference / modes / consensus runtime is retained for
historical reproducibility but disconnected from the active demo path. PR-13 and
its tests are the executable proof. Deletion belongs to PR-16 after S4.

## Cross-session handoff

- `docs/agent_context/active_context.md` — current short handoff.
- `docs/agent_context/known_blockers.md` — load-bearing blocker index.
- `docs/decisions/` — future durable engineering ADRs not already governed by an
  adopted scientific/roadmap document.

Always reconstruct exact Git/PR/CI state before a load-bearing action.
