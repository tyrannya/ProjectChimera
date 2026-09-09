# ProjectChimera — Obsidian External Knowledge Vault Setup

Status: **tooling procedure only**. This does not make Obsidian authoritative project memory.

## Goal

Use `knowledge/` as a structured external research/second-brain vault while keeping ProjectChimera's scientific and engineering source of truth in Git/repository contracts.

## Local setup

1. Install/open Obsidian on the operator machine.
2. Choose **Open folder as vault**.
3. Select the repository's `knowledge/` directory, not the repository root.
4. Start from `wiki/master-index.md`.
5. Keep raw external material in `raw/`, curated notes in `wiki/`, and completed external syntheses in `output/`.

## Authority firewall

Obsidian notes may summarize external evidence, but they must never establish:

- current Git SHA/branch;
- PR/CI state;
- active preregistration;
- scientific results;
- evidence eligibility;
- merge readiness;
- live risk state;
- roadmap adoption.

When a note and repository evidence disagree, repository evidence wins.

## Claude Code usage

Claude Code may read the vault as ordinary repository files. If an Obsidian MCP is later enabled, treat it as a convenience layer for searching/writing **only `knowledge/`**.

Do not grant an external memory/MCP workflow authority to rewrite scientific contracts, front-door roadmaps, evidence manifests, or safety state.

## MCP stance

No Obsidian MCP is enabled by this repository change.

Reason: MCP implementations are third-party operational dependencies and should be security/behavior reviewed before becoming part of a governed workflow. The vault is useful immediately without MCP because Claude Code can read the Markdown files directly.

If an MCP is later adopted, require at minimum:

- local-only or otherwise explicitly trusted transport;
- clear repository/vault path scoping;
- no secrets stored in committed config;
- read/write access constrained to `knowledge/` where possible;
- explicit review of the MCP implementation and update path;
- a rollback/uninstall procedure.
