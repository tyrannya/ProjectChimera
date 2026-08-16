# MiCA & DORA Checklist

> **Status: aspirational, not implemented.** This is a list of obligations a
> production crypto-asset service would have to meet. ProjectChimera is a
> dry-run research platform and implements **none** of the controls below as
> stated: there is no audit-log retention policy, no backup or restore
> procedure, no incident-reporting process, and no third-party monitoring. The
> "five years" retention line describes an intention, not a mechanism.
>
> Do not cite this document as evidence of compliance. Treat it as a to-do list
> for anyone considering operating this system as a regulated service — which
> would require substantially more than the code in this repository.

## MiCA (2024-12-30)
- Compliance with crypto-asset service provider obligations
- Risk management and user protection
- Incident reporting procedures

## DORA (2025-01-17)
- ICT risk management framework
- Operational resilience testing
- Third‑party service monitoring

### Audit Logs
- All trades and system events recorded via Freqtrade and MLflow

### Disaster Recovery
- Backups copied to secure storage; restoration through docker-compose redeploy

### Data Retention
- Parquet archives and mlruns preserved for at least five years
