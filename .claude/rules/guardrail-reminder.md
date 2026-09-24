# Guardrail reminder (always loaded)

- `CLAUDE.md` and `docs/agent_workflow_policy.md` govern; this file adds nothing to them.
- Published Git history is append-only. Two layers enforce part of that: native `permissions.deny` rules and the PreToolUse hook `.claude/hooks/chimera-guardrails.mjs`. They also block secret files and environment dumps. Neither layer is complete; the policy still applies to anything they cannot see.
- On a deny: stop and report the rule id to the owner. Never rephrase, split, quote, encode, alias, script or reroute a command to get past a guardrail. The owner runs a legitimately needed operation manually.
- Never weaken `.claude/settings*.json`, `.claude/hooks/` or `.claude/rules/` inside a task that is not about the guardrails themselves.
- The serving model is fixed; no skill, agent or workflow switches or routes models.
- Details, false-positive procedure and residual limits: `docs/claude_extension_layer.md`.
