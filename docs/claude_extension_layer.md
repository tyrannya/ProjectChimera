# ProjectChimera — Claude Code extension layer

Status: **engineering/governance tooling.** This document opens no research
checkpoint, changes no scientific result, roadmap state, preregistration,
Aegis behaviour, trading path, `prospective_from` or real-money authority.
`CLAUDE.md` and `docs/agent_workflow_policy.md` outrank it. It describes
what `.claude/` enforces mechanically and where that enforcement stops.

Verified against Claude Code **2.1.278** on Windows 11 on 2026-09-21 (live
probes, §4) and against the official documentation (code.claude.com: hooks,
permissions, skills, memory) on the same date.

## 1. What is in `.claude/`

| path | loaded | purpose |
|---|---|---|
| `CLAUDE.md` (repo root) | every session | standing orders; the authority for everything below |
| `.claude/rules/guardrail-reminder.md` | every session (no `paths:` frontmatter) | six lines: the guardrails exist, a deny means stop and report |
| `.claude/settings.json` | every session | `defaultMode: "plan"`, the native deny rules (§2), the hook registration (§3) |
| `.claude/hooks/chimera-guardrails.mjs` | runs on each matching tool call | the enhanced PreToolUse guardrail |
| `.claude/agents/*.md` | descriptions every session; body when spawned | read-only auditors; all `model: inherit` |
| `.claude/skills/*/SKILL.md` | see below | workflow procedures, loaded on invocation |

**Always on.** Every session loads `CLAUDE.md`, the rule file, the settings
(so both enforcement tiers), agent descriptions, and the *descriptions* of the
three model-invocable skills (`chimera-author`, `chimera-remediate`,
`chimera-verify`). The four owner-only skills set
`disable-model-invocation: true`, so their descriptions are kept out of the
model's context entirely. Claude cannot start them on its own.

**Loaded only on invocation.** A skill's body enters the context only when the
skill runs. Skills carry workflow, not authority: each one links back to
`CLAUDE.md` and the policy, and none may select a model, route to one, or merge.

## 2. Two enforcement tiers

The two tiers are independent. Neither is complete, and the system as a whole is
**not** fail-closed. §2.3 lists exactly what remains when the hook cannot run.

### 2.1 Native hard-deny coverage (`permissions.deny`)

Claude Code evaluates these rules itself, before any hook, in every permission
mode. They keep working if Node is missing, if the hook crashes, and if it
times out. Deny rules apply to any subcommand of a compound command, including
`&&`, `;`, `|`, subshells and `$( )`.

What they cover, by canonical form only:

- **Git, Bash and PowerShell:**
  - `git commit *--amend*`, `git rebase` / `git rebase *`, `git pull *--rebase*`, `git pull -r *`;
  - `git push *--force*` (this also covers `--force-with-lease` and `--force-if-includes`), `git push -f *`, `git push * +*`, `git push *--mirror*`, `git push *--delete*`, `git push -d *`, `git push *--prune*`;
  - `git reset *--hard*`, `git clean -*f*`, `git clean *--force*`;
  - `git branch -D *`, `git branch -f *`, `git branch *--force*`, `git checkout -B *`, `git switch -C *`, `git switch *--force-create*`;
  - `git worktree remove *--force*`, `git worktree remove -f *`, `git tag -f *`, `git tag *--force*`;
  - `git update-ref *`, `git replace *`, `git reflog expire|delete *`, `git filter-branch|filter-repo *`;
  - `printenv`, `gh auth token*`.
- **Secrets, via `Read(...)` and `Edit(...)`.** `Edit` governs Write, Edit and NotebookEdit, and a `Read` deny also blocks Edit and Write. Read and Edit rules also apply to `cat`/`head`/`tail`/`sed` and to shell redirect targets.
  - `//**/.env`, and every `.env.*` except `.env.example` (see below);
  - `//**/*.pem`, `//**/*.key`;
  - `~/.ssh/**`, `~/.aws/**`;
  - `~/.npmrc`, `~/.pypirc`, `~/.git-credentials`.
- **Git metadata:** `Edit(//**/.git)` and `Edit(//**/.git/**)`. No file tool and no shell redirect may write into a git directory, including a linked worktree's `.git` pointer file. Reads of `.git` stay allowed.

**Keeping `.env.example` readable.** The committed template has to stay readable, and the rule syntax has no negation: a live probe showed that `[!e]` and `[^e]` are read as *sets containing* `e`. Instead, `.env.*` is covered by a chain of positive ranges that exclude the letters of `example` one position at a time: `.env.[a-df-z…]*`, `.env.e`, `.env.e[a-wyz…]*`, … `.env.example?*`.

**Known limits of the native tier.** Rules are prefix patterns over the command text. These forms slip past them:
- `git -C <dir> …`, `git -c k=v …`, `git --git-dir …`;
- quoted or escaped spellings (`git "rebase"`, `re""base`);
- launchers (`bash -c`, `cmd /c`, `pwsh -c`, `Start-Process`, `iex`);
- short-flag clusters (`-uf`, `-df`) and trailing short flags (`git push origin x -f`);
- the deletion refspec `git push origin :branch`, which could not be expressed (see §4);
- aliases, scripts and interpreters (`python -c "subprocess…"`);
- file access made by a subprocess itself.

PowerShell rules are case-insensitive. So the case-sensitive short flags (`branch -D`, `checkout -B`, `switch -C`) are deliberately *not* native PowerShell rules: they would also block `branch -d`, `checkout -b` and `switch -c`. The hook covers them.

The native tier is also coarser than the hook in a few places:
- `git replace -l` is denied;
- `git clean -fdn` is denied (a dry run);
- a `git pull --rebase=false` is denied.

These are rare, and the owner can run them by hand.

### 2.2 Enhanced hook coverage (`chimera-guardrails.mjs`)

**Registration.** A PreToolUse command hook in exec form: `node` with the script
path as its argument. No shell is involved, so it behaves the same under Git Bash,
PowerShell and Linux. Its matcher is an explicit tool list,
`Bash|PowerShell|Monitor|Read|Write|Edit|NotebookEdit|Grep`, covering the
built-in tools whose input can carry a shell command or a file path.
- `Monitor` is included because it runs a shell `command`.
- `Glob` returns names only and is excluded.
- MCP, agent and task tools never start Node.

**What it inspects.**
- `tool_input.command`, which it lexes twice, once with bash rules and once with PowerShell/cmd rules:
  - quotes are concatenated;
  - segments split on `; & | && || ( )`, newlines, `$( )` and backticks, including inside double quotes;
  - `2>&1` is treated as a redirect, not a separator;
  - line continuations are joined;
  - heredocs and PowerShell here-strings are treated as data, so a commit message may *mention* `git push --force`.
- File fields: `file_path`, `notebook_path`, `path`, `glob`.

**What it adds over the native tier.**
- Git is recognised at any word position:
  - with `sudo`/`env`/`&` prefixes;
  - through `C:\Program Files\Git\cmd\git.exe`, `GIT.EXE`, or `git-rebase`;
  - with global options skipped (`-C`, `-c`, `--git-dir`, `--work-tree`, `--no-pager`…);
  - with long-option abbreviations (`--amen`, `--forc`);
  - with case-sensitive short clusters, so `-D` is denied and `-d` allowed.
- Launchers (`bash`, `sh`, `pwsh`, `powershell`, `cmd`, `wsl`, `eval`, `iex`, `Start-Process`, …) make it dissolve the quoting and re-inspect, up to 4 levels. `-EncodedCommand` is denied outright because it cannot be inspected.
- It adds the rules the native syntax cannot express.

Rules, by id (every deny message carries the id):

| rule | blocks | still allowed |
|---|---|---|
| `git-amend` | `commit --amend` (any abbreviation) | `commit`, `commit -am` |
| `git-rebase` | `rebase` (every form); `pull --rebase[=…]`, `pull -r` | `pull`, `pull --ff-only`, `pull --no-rebase` |
| `git-force-push` | `--force*`, `--force-with-lease*`, `--force-if-includes`, `-f` anywhere in a cluster, `+refspec`, `--mirror` | `push`, `push -u`, `push HEAD:refs/heads/x` |
| `git-push-delete` | `--delete`, `-d`, `:ref`, `--prune` | — |
| `git-hard-reset` | `reset --hard` | `reset HEAD file`, `reset --soft` (see §5) |
| `git-clean` | `clean` without `-n`/`--dry-run` | `clean -n`, `clean -nd` |
| `git-branch-force` | `branch -D/-f/-M/-C/--force`; `checkout -B`; `switch -C/--force-create` | `branch -d/-m/-c/-vv`, `checkout -b`, `switch -c` |
| `git-worktree-force-remove` | `worktree remove -f/--force` | `worktree add`, `worktree remove` |
| `git-tag-force` | `tag -f/--force` | `tag v1`, `tag -l` |
| `git-ref-rewrite` | `update-ref`; `replace` unless listing; `symbolic-ref` writes; `reflog expire/delete` | `show-ref`, `rev-parse`, `symbolic-ref --short HEAD`, `reflog`, `replace -l` |
| `git-history-rewrite` | `filter-branch`, `filter-repo` | — |
| `git-inline-alias` | `-c alias.*=…` | other `-c` options |
| `git-metadata-write` | Write/Edit/NotebookEdit into any `.git`; shell redirects into `.git`; `cp/mv/rm/tee/Set-Content/Remove-Item/…` or `sed -i` naming a `.git` path | Read/Grep/`cat`/`ls`/`Get-Content` of `.git`; `.gitignore`, `.github/` |
| `encoded-command` | `powershell`/`pwsh -EncodedCommand` (any prefix, `-ec`) | `-Command`, `-ea` |
| `secret-path` | `.env*` except `.env.example`; `*.pem`, `*.key`, `id_{rsa,dsa,ecdsa,ed25519}*`; any `.ssh` or `.aws` directory; `.npmrc`, `.pypirc`, `.netrc`, `_netrc`, `.git-credentials`. Matching is case-insensitive, also in `--env-file=.env`, `host:.ssh/…`, `.env::$DATA`, `.env.` | `.env.example`, `test_keys.py` |
| `env-dump` | `printenv`; bare `env`, `set`, `export [-p]`; the PowerShell `env:` drive; `GetEnvironmentVariables()`; `/proc/*/environ` | `echo $env:PATH`, `set -euo pipefail`, `env X=1 cmd` |
| `secret-token` | `gh auth token` | other `gh` commands |
| `internal-error` | stdin that is not JSON, a command over 2 MB, any exception inside the hook | — |

**Output contract.**
- On a block: deny JSON on stdout (`hookSpecificOutput.permissionDecision: "deny"`), the same reason on stderr, and exit code **2**. Claude Code blocks on exit 2 even if it ignores the JSON.
- The reason depends only on the rule. It never contains the command, so it can never repeat a secret someone typed inline.
- Otherwise: exit 0 with no output. The hook **never** emits `allow`, so it can never skip a permission prompt.
- Nothing is logged.

**Not blocked, by design.** Ordinary `git commit`, `git merge` and non-force `git push` are not blocked. Neither is `gh pr merge`: whether a merge is authorised is an owner decision under `CLAUDE.md`, not something this layer decides. Also unblocked: `worktree add`, test runs, and scratch files.

### 2.3 Degraded modes: what is left when the hook is not running

| situation | result | what still blocks |
|---|---|---|
| Node not installed, script missing, or the hook fails to start | Claude Code treats the non-2 exit as a **non-blocking** hook error, and the call proceeds | the native tier only (§2.1) |
| hook exceeds its 10 s timeout | the call **proceeds** | the native tier only |
| hook running; stdin malformed, command > 2 MB, internal exception | **blocked** (`internal-error`, exit 2) | both |
| owner runs a command outside Claude's tool calls | not inspected | nothing: it is the owner's action |

So the hook fails closed *inside a run*, but the layer as a whole does not. The canonical destructive forms and the secret files stay denied without the hook. The alternate forms of §2.2 do not. Tests characterise this:
- `test_a_hook_that_cannot_start_is_not_a_block`;
- `test_pathological_input_finishes_far_inside_the_timeout`, which checks that each case finishes in under 2 s against the 10 s budget;
- `test_native_floor_covers_canonical_forms` and `test_native_floor_leaves_ordinary_work_alone`.

## 3. Why the layer is built this way

- **Context7 is external documentation only.** It is used for current third-party library and framework APIs when version freshness matters. It is never authority for repository state, scientific contracts, safety policy, the roadmap, or exchange semantics where official vendor docs exist (`docs/agent_workflow_policy.md` §5). The hook and permission semantics here were checked against Anthropic's own documentation and then confirmed by live probes (§4).
- **Path-scoped rules are not safety authority.** A rule with `paths:` frontmatter loads only after Claude touches a matching file. A command that never reads such a file never sees it, and nothing enforces a rule's text in any case. That is why the only rule file here is short and unconditional, and why every safety invariant sits in `settings.json` and the hook, where Claude Code enforces it mechanically.
- **Why Node.** It runs on the same code path under Windows and Linux, needs no dependencies, and is independent of this machine's Python environment. The exec form removes any dependence on Git Bash or PowerShell quoting.
- **Why an explicit matcher.** Node starts only on the eight tools where there is something to inspect. Unrelated MCP, agent and task calls pay no latency, and a schema change to one of those tools cannot break this guardrail.

## 4. Live probe record (2026-09-21, Claude Code 2.1.278, Windows)

The probes ran against a throwaway repository with a local bare remote, and against dummy secret files in the session scratchpad. They were designed to be harmless even if not denied: `--dry-run`, nonexistent refs, dummy files. No real credential path was read. The native tier was probed **before** the hook was registered, which is the "hook unavailable" state.

- **Denied natively (Bash):**
  - `commit --amend`, `rebase`, `pull --rebase`;
  - `push --force`, `--force-with-lease`, `-f`, `+main`, `--delete`;
  - `reset --hard`, `clean -fdn`;
  - `branch -D`, `branch -f`, `worktree remove --force`, `checkout -B`, `switch -C`, `tag -f`;
  - `update-ref -d`, `reflog expire`;
  - `replace -l` and `printenv <nonexistent>`: these benign commands matched only a rule, so the rules themselves are doing the denying;
  - `gh auth token`.
- **Not denied** (positive controls, which ran): `branch -d`, `checkout -b`, `switch -c`, `push -u`, `push HEAD:refs/heads/x`, `merge`, an ordinary commit. Bash rules are case-sensitive.
- **PowerShell:** `push --force` and `reset --hard` denied; `branch -d` and `checkout -b` allowed.
- **Not expressible:** `git push origin :branch`. Both `Bash(git push * :*)` and `… :**)` failed to match, because a trailing `:*` parses as the legacy suffix. It is covered by the hook only.
- **Native blind spots confirmed:** `git -C <repo> reset --hard` and `git push origin main -f` both ran (harmlessly).
- **Files:**
  - `Read`, and `cat` in Bash, of `.env`, `.env.local` (plus eight other `.env.*` variants), `server.pem` and `tls.key` were denied; `.env.example` was readable;
  - Write, and `echo >`, into `repo/.git/` were denied; Read, `cat` and `ls` of `.git` were allowed;
  - `~/.ssh/<nonexistent>`, `~/.aws/credentials`, `~/.pypirc` and `~/.git-credentials` (all absent) were denied, while an uncovered nonexistent control returned "file does not exist";
  - `~/.npmrc` exists on the probe machine and was deliberately not probed. Its rule is characterised from the documentation.
- **After the hook was registered:**
  - `git -C <repo> reset --hard` → `[git-hard-reset]`;
  - `git push … -f` → `[git-force-push]`;
  - `git push origin :probe-side` → `[git-push-delete]`;
  - PowerShell `git branch -D` → `[git-branch-force]`;
  - `git -C <repo> status` was allowed.

## 5. Threat model and residual limits

The layer is a **seatbelt against careless or accidental agent actions**, not a
sandbox against a determined adversary. Known gaps:

- **Hook blind spots:**
  - variable indirection (`f=.e; cat ${f}nv`);
  - globs (`cat *`) and recursive `grep -r` reading a `.env`;
  - interpreters and script files (`python -c`, `.sh`/`.ps1`);
  - pre-existing git aliases or config (`pull.rebase=true`);
  - strings piped into `xargs`;
  - `cd .git && rm refs/…`;
  - `git fetch +src:dst` into local branches;
  - 8.3 short names and symlinks;
  - a heredoc whose delimiter is chosen to swallow later lines.
- `reset --soft` and `reset --mixed` can rewind *unpublished* local history. Published history is still protected, because pushing the result would need a force push.
- **Hook false positives:**
  - quoted prose that contains a path-like secret (`"see ~/.ssh/config"`);
  - `grep "\.env"`;
  - `echo git rebase`;
  - text in a command that also contains a launcher;
  - copying *out of* `.git` with `cp`.
- Commands the owner runs themselves are never inspected, by design.

## 6. Invoking the skills

| skill | who starts it | what it does |
|---|---|---|
| `/chimera-author <task>` | owner or model | new governed work: clean base → plan → preregistration before results → surgical implementation → two-sided tests → ordinary commits → draft PR |
| `/chimera-remediate <PR>` | owner or model | review findings → fix, dispute with evidence, or mark out of scope → appended commits and a finding→commit map → new CI → a fresh delta review |
| `/chimera-verify [PR]` | owner or model | local tests vs exact-head CI vs independent acceptance, kept separate |
| `/chimera-review <PR> [SHA]` | **owner only** | READ-ONLY independent review. Verdicts: `APPROVED — <scope>` / `REQUEST CHANGES — PR #<n> NOT READY` / `BLOCKED / CANNOT VERIFY` |
| `/chimera-delta-review <PR> <old SHA> [new]` | **owner only** | READ-ONLY delta re-review. Verdicts: `DELTA APPROVED` / `REQUEST CHANGES` / `BLOCKED / CANNOT VERIFY` |
| `/chimera-reproduce <checkpoint> <SHA>` | **owner only** | reproduces a published, eligible result against its frozen manifests; commits nothing |
| `/chimera-mutation-audit <PR>` | **owner only** | restores the defect and confirms the witnesses fail; the tree ends byte-identical |

An author's own invocation of a review skill, or a subagent the author spawns, is never independent acceptance.

## 7. Testing the guardrails

```text
python -m pytest tests/test_claude_guardrails.py
```

- `tests/claude_guardrails_cases.json` holds one PreToolUse call per line. Each case says whether the call is allowed or denied, which rule must fire, and whether the native tier must also cover it (`native`) or must leave it alone (`native_negative`).
- The test wraps each case in a real hook envelope and runs it through Node on both CI legs. A deny must be for the named rule, with the exact JSON schema. An allow must produce no output at all.
- Locally, a missing Node skips these tests; under `CI` a missing Node fails them.

To try a single call by hand:

```text
echo '{"hook_event_name":"PreToolUse","tool_name":"Bash","tool_input":{"command":"git -C x reset --hard"}}' | node .claude/hooks/chimera-guardrails.mjs; echo "exit=$?"
```

Every change to a rule adds a **pair** of fixtures: the unsafe form that must be denied, and the nearest legitimate form that must still pass (`docs/claude_code_setup_audit_plan.md`: a guard that can brick healthy work is not acceptable merely because it is conservative).

## 8. Diagnosing a false positive without bypassing governance

1. **Read the rule id** in the message (`[git-branch-force]`, `[secret-path]`, …) and find it in §2.2.
2. **Do not route around it.** Claude must not rephrase the command, split it, quote it differently, encode it, run it through another shell, a script, an alias or an interpreter, or reach the target another way. Any of those is a governance violation even when the underlying operation is legitimate.
3. **Stop and report** to the owner: the rule id, what the operation was for, and why it is legitimate.
4. **The owner decides.** The supported recovery path is for the owner to run the operation manually, outside Claude's tool calls.
5. **Fix a real false positive in a governance PR.** Narrow the rule in the hook and/or `settings.json`, and add the fixture pair (the false positive now allowed, plus the unsafe neighbour still denied). The PR needs normal review.
6. **Never** silently disable the layer: no `disableAllHooks`, no removed or overridden rules in `.claude/settings.local.json`, no edits to the hook inside an unrelated task. If an emergency disable is ever needed, the owner does it, and it is recorded.

## 9. Recommendation record (per `docs/claude_code_setup_audit_plan.md`)

| candidate | problem prevented | mechanism | false-positive risk | bypass / recovery | tests | status |
|---|---|---|---|---|---|---|
| Git chronology guard | amend/rebase/force/reset/clean/branch/tag/worktree/ref rewriting | native deny + hook | low; see §5 | owner runs it manually; governance PR for real FPs | fixture pairs + native coverage | **adopted** |
| Secret-access guard | reading or exfiltrating credentials | native Read/Edit deny + hook | low (`.env.example` spared) | owner runs it manually | fixture pairs + P2/P3/P5 probes | **adopted** |
| Git-metadata write guard | editing refs/HEAD/logs directly | native `Edit(.git)` + hook | low (reads allowed) | owner edits manually | fixture pairs + P4 probe | **adopted** |
| Completion verifier | "done" claims without evidence | `/chimera-verify` skill (procedure, not enforcement) | none | — | structural test | **adopted** as a skill |
| Mutation/test concurrency | mutated source racing ordinary suites | `/chimera-mutation-audit` procedure | none | — | structural test | **adopted** as a skill |
| Write guard for frozen evidence; scientific-stage guard; live/acquisition guard | edits to frozen artifacts; results before preregistration | — | — | — | — | **not in this PR**; each needs its own two-sided design |
