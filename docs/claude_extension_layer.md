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

What they cover, by canonical form only. Git accepts any unique abbreviation of a long option (`git reset --h` *is* `--hard`), so the long-option literals are cut to the shortest prefix that is still only the dangerous option:

- **Git, Bash and PowerShell:**
  - `git commit *--am*`, `git rebase` / `git rebase *`, `git pull *--reb*`, `git pull -r *`;
  - `git push *--forc*` (this also covers `--force-with-lease` and `--force-if-includes`), `git push -f *`, `git push * +*`, `git push *--m*` (mirror), `git push *--de*`, `git push -d *`, `git push *--pru*`;
  - `git reset *--h*`, `git clean -*f*`, `git clean *--f*`;
  - `git branch -D *`, `git branch -f *`, `git branch *--forc*`, `git checkout -B *`, `git switch -C *`, `git switch *--force-*`;
  - `git worktree remove *--f*`, `git worktree remove -f *`, `git tag -f *`, `git tag *--forc*`;
  - `git update-ref *`, `git replace *`, `git reflog expire|delete *`, `git filter-branch|filter-repo *`;
  - `printenv`, `gh auth token*`, `gh auth status *-t*`, `gh config *get *oauth_token*`, `git credential*`, `gh repo delete*`.
- **Secrets, via `Read(...)` and `Edit(...)`.** `Edit` governs Write, Edit and NotebookEdit, and a `Read` deny also blocks Edit and Write. Read and Edit rules also apply to `cat`/`head`/`tail`/`sed` and to shell redirect targets.
  - `//**/.env`, `//**/.envrc`, `//**/.env[-_]*`, and every `.env.*` except `.env.example` (see below);
  - `//**/*.pem`, `//**/*.key`;
  - `~/.ssh/**`, `~/.aws/**`;
  - `~/.npmrc`, `~/.pypirc`, `~/.git-credentials`;
  - `~/.claude/.credentials.json`, `~/.config/gh/hosts.yml`.
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
- `git pull --rebase=false` is denied;
- `git reset --help` is denied;
- `gh auth status` with a `-t`-containing hostname is denied.

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
  - in the PowerShell pass, unquoted `{` and `}` also split segments, so a script block's body is read as commands (`&{git …}`, `if ($x) {git …}`), and a literal `(Get-Command <name>)` or `(gcm [-Name] <name>)`, optionally followed by `.Source`/`.Path`/`.Definition`, is read as the word `<name>`, so `& (Get-Command git) push --force` is a `git` command. Braces inside quotes and here-strings stay data;
  - line continuations are joined;
  - heredocs and PowerShell here-strings are treated as data, so a commit message may *mention* `git push --force`. The exception is a heredoc fed to a launcher on its own line (`bash <<EOF`, `cat <<EOF | bash`), which is analysed as commands.
- File fields: `file_path`, `notebook_path`, `path`, `glob`.

**What it adds over the native tier.**
- Git is recognised at any word position:
  - with `sudo`/`env`/`watch`/`&` prefixes;
  - through `C:\Program Files\Git\cmd\git.exe`, `GIT.EXE`, or `git-rebase`;
  - with bash `$'…'`/`$"…"` quoting;
  - with global options skipped (`-C`, `-c`, `--git-dir`, `--work-tree`, `--no-pager`…);
  - with every long-option abbreviation down to `--` plus one letter (`--h`, `--am`, `--forc`); `switch --force-create` is the one exception, because `--force` alone is a different option;
  - with case-sensitive short clusters, so `-D` is denied and `-d` allowed.
- A segment with more than 64 `git` words is refused (`internal-error`) rather than inspected. Checking each occurrence against its tail is quadratic, and a hook that times out lets the call through.
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
| `git-remote-ref-rewrite` | `gh api` PATCH/PUT/DELETE on `…/git/refs/…`; GraphQL `deleteRef`/`updateRef`; `gh repo delete` | `gh api` reads, `gh api -X PATCH …/pulls/N` |
| `git-inline-alias` | `-c alias.*=…` | other `-c` options |
| `git-metadata-write` | Write/Edit/NotebookEdit into any `.git`; shell redirects into `.git`; `cp/mv/rm/tee/Set-Content/Remove-Item/…`, `sed -i` or a PowerShell `[IO.File]::Write…/Delete…/Move…` naming a `.git` path | Read/Grep/`cat`/`ls`/`Get-Content`/`[IO.File]::ReadAllText` of `.git`; `.gitignore`, `.github/` |
| `encoded-command` | `powershell`/`pwsh -EncodedCommand` (any prefix, `-ec`) | `-Command`, `-ea` |
| `secret-path` | every `.env*` name except `.env.example` (`.envrc`, `.env-prod`, …); `*.pem`, `*.key`, `id_{rsa,dsa,ecdsa,ed25519}*`; any `.ssh` or `.aws` directory; `.npmrc`, `.pypirc`, `.netrc`, `_netrc`, `.git-credentials`, `.credentials.json`; GitHub CLI `hosts.yml`. Matching is case-insensitive, also in `--env-file=.env`, `host:.ssh/…`, `.env::$DATA`, `.env.` | `.env.example`, `test_keys.py`, prose words with no path separator (`"id_rsa rotation runbook"`) |
| `env-dump` | at a command position (see below): `printenv`; `env` with only options (`-u NAME`, `-C DIR`, flags) and `NAME=value` assignments, so nothing left to run; bare `set`, and `set PREFIX` without `=` under `cmd`; `export [-p]`; `declare`/`typeset` with only flags; `compgen -e/-v`. Output redirects are not arguments: not the target (`env > out.txt`), which is excluded by its position alone, so a target that shares the command's name leaves the command in place (`env > env`, `set > set`, `export -p > export`), not an fd number glued to the operator (`env 2>/dev/null`, `env 1>out.txt`, PowerShell `env *>&1`), and not a dup target (`env 2>&1`, `env >&2`). An input redirect still is (`env < in.txt`, §5). Also listing the PowerShell `env:` drive; `GetEnvironmentVariables()`; `/proc/*/environ` | `echo env`, `printf "set"`, `bash -c "echo env"`, `grep printenv`, `git log \| grep -c set`, `"env: …"` commit scopes, `Get-Item Env:PATH`, `echo $env:PATH`, `set -euo pipefail`, `cmd /c set FOO=bar`, `env X=1 pytest …` (the utility after the assignments is judged instead), `python tool.py 2`, `echo hello 2>&1`, `echo env > env`, `env 2 >f` (spaced, so `2` is the command) |
| `secret-token` | `gh auth token`; `gh auth status -t/--show-token`; `gh config … get … oauth_token`, with `-h/--host` anywhere, including before `get`; `git credential …`, `git credential-*` | `gh auth status`, `gh config get git_protocol`, `gh config list`, `git config --get credential.helper` |
| `internal-error` | stdin that is not a JSON object; a command over 2 MB; more than 64 `git` words in one segment; any exception inside the hook | — |

**Command position**, for `env-dump`, is one of:
- the first word of a segment;
- the word after a prefix wrapper (`sudo`, `env`, `nohup`, `xargs`, …) that is itself at a command position;
- the utility after `env`'s options and assignments;
- the word after a launcher's command flag (`-c`, `-lc`, `/c`, `/k`, `-Command`), once a launcher (`bash`, `sh`, `cmd`, `pwsh`, …) has appeared in the segment.

Launcher strings are dissolved and re-inspected as described above, so `bash -c "env FOO=1"` and `bash -c "sh -c env"` are judged at their inner command position. `printenv` keeps its older, broader test: any word after one of those wrapper or flag words.

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
| hook running; stdin not a JSON object, command > 2 MB, > 64 `git` words in a segment, internal exception | **blocked** (`internal-error`, exit 2) | both |
| owner runs a command outside Claude's tool calls | not inspected | nothing: it is the owner's action |

So the hook fails closed *inside a run*, but the layer as a whole does not. The canonical destructive forms and the secret files stay denied without the hook. The alternate forms of §2.2 do not. Tests characterise this:
- `test_a_hook_that_cannot_start_is_not_a_block`;
- `test_pathological_input_finishes_far_inside_the_timeout`, which checks that each case finishes in under 5 s against the 10 s budget: 1 MB of words, 30k `git` words, 20k git segments, 5k launchers, deep nesting, long heredocs, and the command-position, redirect, brace and `Get-Command` paths of §4.2;
- `test_env_dump_scan_stays_linear_up_to_the_size_cap`, the same 5 s bound for a `declare`/`typeset` launcher chain just under the 2 MB cap, with a harmless tail and with a guarded one;
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
- **Round 2, after the adversarial review (§4.1):**
  - `git reset --h HEAD` was confirmed to be a real hard reset in the scratch repository. It had passed both tiers.
  - The shortened native literals were then probed with spellings that the hook deliberately allows, so only the native tier could deny them: `git reset --h__probe__`, `git commit --am__probe__` and `git credential__probe__` were each denied with the native permission message.

### 4.1 Adversarial review round

A read-only reviewer ran about 59 payloads through the hook. It ran on the same serving model in a fresh context, with the scientific and safety boundaries restated to it. It was the author's own subagent, so this is **not** independent acceptance. Each of its findings was reproduced before it was acted on:

| finding | disposition |
|---|---|
| `git reset --h` is a real hard reset and passed both tiers | fixed: abbreviation floor lowered to `--` plus one letter; native literals shortened; fixtures added |
| ~90 KB of repeated `git` words made the hook run > 10 s, i.e. fail open on timeout | fixed: refuse a segment with more than 64 `git` words; timing tests at 30k words / 20k segments / 5k launchers |
| `sudo git-rebase`, `env … git-push --force`, `watch git-clean` | fixed: `git-<sub>` is recognised at any position |
| `gh auth status -t/--show-token` | fixed, in both tiers |
| `.envrc`, `.env-prod` were allowed | fixed, in both tiers |
| false positives: `"env: …"` commit scopes, `echo printenv`, `grep printenv`, prose like `"id_rsa rotation runbook"` or `".npmrc: pin"` | fixed: env-drive listing only; `printenv` only in command position; prose words without a path separator are not paths |
| a JSON array on stdin was allowed | fixed: anything that is not a JSON object fails closed |
| (hit live by the author) a command running `bash <validator>.sh` together with a heredoc commit message that mentioned `git-rebase` was denied: a launcher *anywhere* made every heredoc body count as commands | fixed: only a heredoc fed to a launcher on its own line is analysed; two-sided fixture added (`cat <<EOF \| bash` is still denied) |

The author's own review added: bash `$'…'` quoting; `[IO.File]::` writes into `.git`; `git credential`; the GitHub CLI `hosts.yml` and Claude `.credentials.json`; `declare -p` / `compgen -e`; and `gh api` ref PATCH/DELETE / GraphQL ref mutations / `gh repo delete`, which are published-history rewrites that never call `git push`.

### 4.2 Independent review round (PR #101 at `5dbd576`)

A fresh independent review returned REQUEST CHANGES with three merge-blocking bypasses. Each was reproduced against the hook at `5dbd576` before it was fixed. The fixtures added as evidence for each fix fail on `5dbd576`, and a targeted mutation of each fix makes them fail again. `ps-if-block-multiline` is the exception: it already passed on `5dbd576`, so it is regression coverage, not evidence of the fix.

| finding | disposition |
|---|---|
| B1: `bash -c env`, `sh -c "set"`, `cmd /c set`, `env FOO=1` were allowed. Dump commands were only recognised as the first word of a segment, and `env NAME=value` was not "flags only" | fixed: `env-dump` is judged at every command position (§2.2); `env` counts as a dump when nothing is left to run after its options and assignments |
| B2: PowerShell `&{git push --force}`, `if ($true) {git push --force}`, `& (Get-Command git) push --force` were allowed. `{git` was taken as the command, and the resolved `git` sat in its own segment | fixed: in the PowerShell pass, braces split script blocks; a literal `(Get-Command <name>)` is read as `<name>` |
| B3: `gh config get -h github.com oauth_token` printed the stored token and passed both tiers | fixed, in both tiers: native `gh config *get *oauth_token*` (Bash and PowerShell), and the hook's `secret-token` rule |

The gh syntax was checked on gh 2.67.0 against an isolated `GH_CONFIG_DIR` that held a dummy sentinel for a fake host, with the token environment variables unset. No real credential was read. gh returned the sentinel for `-h H`, `--host H`, `--host=H` and `-hH`, with the host before or after the key, after `--`, and even before `get` (`gh config -h H get oauth_token`). The key is case-sensitive. `gh config list` does not print the token. The new native rule was checked against the documented rule semantics in the test model, not live-probed.

The author's first B1 draft rescanned the words after each command position. On a crafted 1 MB command (`bash -c env -c env …`) that took about 90 s, which is a timeout, and a timeout lets the call through. Timing cases for these paths were added.

The delta review of `09469a5` found two more B1 defects, both reproduced before they were fixed:
- The `declare`/`typeset`/`export` test still rescanned the rest of the segment from each command position. `bash -c -/declare -c -/declare … echo` makes every `-/declare` both a command and a flag, and the final word forces every scan to the end. That took 16 s at 0.5 MB and 47 s at 1 MB, and a guarded tail (`… git push --force`) took 23 s at 1 MB before its deny. Now the finding of command positions is one pass that visits each word once, and each rule after it is O(1) against facts computed once per segment ("the last word that is not a flag"). The same command at 1.9 MB now takes under 1 s. `test_env_dump_scan_stays_linear_up_to_the_size_cap` runs it near the 2 MB cap and fails if the rescan is restored.
- `env 2>&1`, `env 2>/dev/null`, `env 1>out.txt`, `env >&2`, `set 2>&1`, `export -p 2>&1` and `declare -p 2>&1` were allowed, because the fd number and the dup target were read as arguments. Now a digit-only word glued to `>`/`<` (and PowerShell's `*`) is part of the redirect, and so is the word after `>&`/`<&`. Numbers that are arguments stay words: `python tool.py 2` and the spaced `env 2 >f`.

## 5. Threat model and residual limits

The layer is a **seatbelt against careless or accidental agent actions**, not a
sandbox against a determined adversary. Known gaps:

- **Hook blind spots:**
  - variable indirection (`f=.e; cat ${f}nv`);
  - brace expansion (`git {push,--force}`);
  - globs (`cat *`) and recursive `grep -r` reading a `.env`;
  - interpreters and script files (`python -c`, `.sh`/`.ps1`);
  - pre-existing git aliases or config (`pull.rebase=true`);
  - strings piped into `xargs`;
  - `cd .git && rm refs/…`;
  - `git fetch +src:dst` into local branches;
  - an environment dump behind a wrapper that takes an argument (`timeout 5 printenv`, `sudo -u root env`, `nice -n 5 env`), or with its input redirected (`env < in.txt`: the file is read as an argument);
  - a PowerShell command resolved other than as a literal `(Get-Command <name>)`: a wildcard or computed name, or a variable (`$g = Get-Command git; & $g …`);
  - 8.3 short names and symlinks;
  - a heredoc whose delimiter is chosen to swallow later lines.
- **Not covered by design:** printing a single variable (`echo $VAR`, `Get-Item Env:X`) and `git stash drop`/`clear`. The stash stack is shared across worktrees, so treat `stash drop`/`clear` with care.
- `reset --soft` and `reset --mixed` can rewind *unpublished* local history. Published history is still protected, because pushing the result would need a force push.
- **Hook false positives:**
  - quoted prose that contains a path with a separator and a secret name (`"see ~/.ssh/config"`);
  - `grep "\.env"`;
  - `echo git rebase`;
  - a word exactly equal to `git-rebase` (e.g. a branch of that name);
  - quoted text (not a heredoc) in a command that also contains a launcher;
  - unquoted braces around a guarded word, which the PowerShell pass reads as a script block (`echo {set}`, `${set}`);
  - copying *out of* `.git` with `cp`.
- **Tamper resistance is governance, not mechanism.** The rule file forbids weakening `.claude/` in an unrelated task. Nothing mechanically stops an edit to the hook or `settings.json`, and those edits show up in review.
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
