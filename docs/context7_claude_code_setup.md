# ProjectChimera — Context7 for Claude Code

Status: **tooling procedure**.

Context7 is used only to retrieve current third-party library/framework documentation when freshness matters. It is not ProjectChimera governance or scientific evidence by itself.

## Recommended setup

From the repository root on a machine with Node.js 18+:

```powershell
./scripts/setup_context7_claude.ps1
```

Equivalent upstream setup command:

```text
npx ctx7 setup --claude --project
```

Authentication is handled by the Context7 setup flow; do not commit API keys.

## Verify after setup

In a fresh Claude Code session:

1. confirm Context7 is visible as an installed project integration/skill/MCP as expected;
2. make one harmless documentation lookup against a known library;
3. verify the result is current/version-aware;
4. if setup is unavailable in the current cloud/remote environment, continue without it rather than inventing library API facts.

## Usage policy

Use Context7 when current third-party library API behavior matters.

Prefer official first-party documentation for:

- Binance/exchange semantics;
- Anthropic/Claude behavior;
- security/safety-critical service semantics;
- any load-bearing protocol claim where the vendor's primary documentation is available.

For ProjectChimera state, Git/repository evidence always outranks Context7.

## Sources checked when this setup was introduced

- Context7 Claude Code guide: `https://context7.com/docs/clients/claude-code`
- Context7 CLI guide: `https://context7.com/docs/clients/cli`

Re-check upstream docs before changing the integration later.
