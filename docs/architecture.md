# Архитектура системы

```mermaid
flowchart LR
    A[data_pipeline] --> B[train]
    B --> C[MLflow]
    C --> D[BentoML]
    D --> E[Freqtrade]
