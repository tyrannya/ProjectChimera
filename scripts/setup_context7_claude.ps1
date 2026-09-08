# ProjectChimera helper: configure Context7 for Claude Code at project scope.
# This script does not store API keys in the repository.
# Run from the ProjectChimera repository root on a machine with Node.js 18+.

$ErrorActionPreference = "Stop"

Write-Host "ProjectChimera Context7 setup"
Write-Host "- project-scoped Claude Code integration"
Write-Host "- no credentials will be written by this script"

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    throw "Node.js is required. Install Node.js 18+ first."
}

$nodeVersion = node --version
Write-Host "Node: $nodeVersion"

if (-not (Get-Command npx -ErrorAction SilentlyContinue)) {
    throw "npx is required. Install a current Node.js/npm distribution first."
}

Write-Host "Launching official Context7 setup for Claude Code (project scope)..."
Write-Host "Authentication is interactive unless you already configured Context7 credentials."

# Official Context7 CLI supports Claude Code and project-scoped setup.
npx ctx7 setup --claude --project

Write-Host "Context7 setup command finished."
Write-Host "Verify the installed MCP/skill in Claude Code before relying on it."
Write-Host "For exchange semantics, continue to prefer official exchange documentation."
