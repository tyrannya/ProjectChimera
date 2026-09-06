# Engineering Architecture Decision Records

Use this directory for durable **engineering architecture** decisions that are
not already governed by a frozen research contract, adopted roadmap/master plan,
or formal amendment.

Do not duplicate scientific/governance decisions here. The original authoritative
record must remain the source of truth.

## When to write an ADR

Write one when a future maintainer would otherwise have to rediscover why a
load-bearing engineering choice was made, for example persistence ordering,
process boundaries, deterministic replay architecture, or a deliberate platform
constraint.

Do not write an ADR for:

- routine implementation details;
- temporary debugging choices;
- a decision already recorded in an adopted amendment/contract;
- post-result changes to scientific rules.

## Minimal template

```markdown
# ADR-NNN — Short title

Status: proposed | accepted | superseded
Date: YYYY-MM-DD

## Context

What durable engineering problem required a choice?

## Decision

What was chosen?

## Invariants

What must remain true regardless of future refactoring?

## Alternatives considered

What materially different option was rejected, and why?

## Consequences

What does this make easier, harder, or deliberately impossible?

## Evidence / references

Exact PR, commit, tests, and governing documents.
```

If an ADR later conflicts with an adopted scientific/governance document, the
scientific/governance document wins and the ADR must be marked superseded.
