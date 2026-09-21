#!/usr/bin/env node
// ProjectChimera PreToolUse guardrail -- the ENHANCED layer.
//
// The hard floor is `permissions.deny` in .claude/settings.json, which Claude Code
// enforces itself. This hook adds what those prefix rules cannot express:
// `git -C`, quoting, launchers (`bash -c`, `cmd /c`, `pwsh -c`), flag clusters,
// chained commands and Windows spellings. It is NOT fail-closed as a whole: a
// hook that cannot start, or times out, lets the call continue under the native
// rules only. Inside a running hook, unparseable input and internal errors do
// fail closed. Threat model and residual limits: docs/claude_extension_layer.md.
//
// Contract: exit 0 with no output = no opinion (the normal permission flow
// applies; this hook never emits "allow"). A block prints deny JSON on stdout,
// the same reason on stderr, and exits 2. The reason never echoes the command.

import { readFileSync } from "node:fs";

const DOC = "docs/claude_extension_layer.md";
const WHY = {
  "git-amend": "git commit --amend rewrites an existing commit",
  "git-rebase": "rebase rewrites commits",
  "git-force-push": "a forced push overwrites published history",
  "git-push-delete": "deleting remote refs removes published history",
  "git-hard-reset": "git reset --hard discards commits and work",
  "git-clean": "git clean deletes untracked files (only -n/--dry-run is allowed)",
  "git-branch-force": "force-moving, force-copying or force-deleting a branch rewrites refs",
  "git-worktree-force-remove": "forced worktree removal discards uncommitted work",
  "git-tag-force": "force-moving a tag rewrites a published ref",
  "git-ref-rewrite": "direct ref or reflog manipulation bypasses Git's history checks",
  "git-history-rewrite": "filter-branch/filter-repo rewrite history",
  "git-inline-alias": "an inline git alias (-c alias.*) cannot be inspected",
  "git-metadata-write": "writing into .git directly bypasses Git",
  "git-remote-ref-rewrite": "rewriting or deleting remote refs through the GitHub API removes published history",
  "encoded-command": "an encoded PowerShell command cannot be inspected",
  "secret-path": "this touches a credential or secret file",
  "secret-token": "this prints a stored credential",
  "env-dump": "dumping the environment exposes secrets",
  "internal-error": "the guardrail could not inspect this call, so it fails closed",
};

const LAUNCHERS = new Set([
  "bash", "sh", "zsh", "dash", "ksh", "fish", "pwsh", "powershell", "cmd", "wsl", "eval",
  "iex", "invoke-expression", "start-process", "saps", "start", "su", "sudo", "runas", "watch",
]);
const WRITERS = new Set([
  "cp", "mv", "rm", "rmdir", "tee", "dd", "truncate", "ln", "touch", "install", "rsync",
  "mkdir", "md", "set-content", "sc", "add-content", "ac", "out-file", "new-item", "ni",
  "remove-item", "ri", "del", "erase", "rd", "move-item", "mi", "move", "copy-item", "cpi",
  "copy", "rename-item", "rni", "ren", "clear-content", "clc",
]);
// Words after which the next word is in command position: prefix wrappers and
// launcher flags (`sudo printenv`, `bash -c printenv`). `echo printenv` is text.
const WRAPPERS = new Set([
  "sudo", "doas", "env", "command", "exec", "builtin", "time", "nohup", "nice", "xargs",
  "watch", "-c", "/c", "/k", "-lc", "-command",
]);
const WRITE_TOOLS = new Set(["Write", "Edit", "MultiEdit", "NotebookEdit"]);
const SECRET_NAMES = new Set([
  ".npmrc", ".pypirc", ".netrc", "_netrc", ".git-credentials", ".credentials.json",
]);
const DOTNET_WRITE = /\[(system\.)?io\.(file|directory)\]::(write|append|create|delete|move|copy|replace|open)/;
const MAX_COMMAND = 2_000_000;
// More `git` words than this in one segment is not a command anyone writes; it
// is refused rather than inspected, because inspecting each occurrence against
// its tail is quadratic and a hook that times out lets the call through.
const MAX_GIT_WORDS = 64;

// --- paths -------------------------------------------------------------------

// Every path-like reading of a string: split on `= @ , :` so `--env-file=.env`,
// `host:.ssh/id_rsa`, `C:\x` and `.env::$DATA` all expose the path; lower-case
// and `/`-separate (Windows is case-insensitive); drop Windows' trailing dots
// and spaces, which the filesystem ignores.
function pathViews(s) {
  return String(s)
    .split(/[=@,:]/)
    .map((part) =>
      part
        .toLowerCase()
        .replace(/\\/g, "/")
        .split("/")
        .map((seg) => seg.replace(/[. ]+$/, ""))
        .filter(Boolean),
    )
    .filter((segs) => segs.length);
}

function isSecret(s) {
  // Cheap superset of every trigger below; keeps megabyte commands fast.
  if (!/env|pem|key|id_|ssh|aws|rc|credential|hosts/i.test(s)) return false;
  return pathViews(s).some((segs) => {
    const base = segs[segs.length - 1];
    return (
      (base.startsWith(".env") && base !== ".env.example") ||
      /\.(pem|key)$/.test(base) ||
      /^id_(rsa|dsa|ecdsa|ed25519)/.test(base) ||
      SECRET_NAMES.has(base) ||
      (base === "hosts.yml" && (segs.includes("gh") || segs.includes("github cli"))) ||
      segs.includes(".ssh") ||
      segs.includes(".aws")
    );
  });
}

// A shell word with inner whitespace and no path separator is prose (a commit
// message, an echo), not a path: `"id_rsa rotation runbook"`, `".npmrc: pin"`.
function isSecretWord(w) {
  const t = w.trim();
  return !(/\s/.test(t) && !/[\\/]/.test(t)) && isSecret(t);
}

const isGitMeta = (s) => pathViews(s).some((segs) => segs.includes(".git"));

// Lower-case basename of an executable word: `C:\Program Files\Git\cmd\git.exe` -> git.
const exe = (w) =>
  w.replace(/\\/g, "/").split("/").pop().toLowerCase().replace(/\.(exe|cmd|bat|com)$/, "");

// --- lexer -------------------------------------------------------------------

// Split a command into segments of words the way a shell would, well enough to
// find what runs. Total: it never throws and always terminates. `ps` selects
// PowerShell/cmd escaping (backtick and caret escape, backslash is literal)
// instead of bash's (backslash escapes, backtick substitutes). Callers run both.
// Heredoc and here-string bodies are data: they are returned separately and not
// segmented, so a commit message that mentions a command does not trip a rule.
function lex(src, ps) {
  const segments = [];
  const bodies = [];
  const frames = [];
  let seg = { words: [], redirects: [] };
  let w = null;
  let redir = false;
  let mode = "cmd";
  let close = null;
  let heredocs = [];
  const add = (s) => {
    w = (w ?? "") + s;
  };
  const endWord = () => {
    if (w === null) return;
    seg.words.push(w);
    if (redir) seg.redirects.push(w);
    w = null;
    redir = false;
  };
  const endSeg = () => {
    endWord();
    if (seg.words.length) segments.push(seg);
    seg = { words: [], redirects: [] };
  };
  const push = (closer) => {
    frames.push({ seg, w, redir, mode, close });
    seg = { words: [], redirects: [] };
    w = null;
    redir = false;
    mode = "cmd";
    close = closer;
  };
  const pop = () => {
    endSeg();
    ({ seg, w, redir, mode, close } = frames.pop());
  };
  // Consume heredoc bodies after a newline. A delimiter that never appears means
  // this was not really a heredoc (e.g. `$((1<<2))`), so nothing is skipped.
  // Only a body fed to a launcher (`bash <<EOF`, `cat <<EOF | bash`: a launcher
  // in the heredoc's segment or later on its line) is returned for analysis; any
  // other body -- a commit message, a file being written -- is data.
  const takeHeredocs = (i) => {
    for (const { delim, from } of heredocs) {
      const feed = segments.slice(from).some((s) => s.words.some((x) => LAUNCHERS.has(exe(x))));
      let j = i;
      let found = -1;
      while (j <= src.length) {
        const nl = src.indexOf("\n", j);
        const end = nl === -1 ? src.length : nl;
        if (src.slice(j, end).trim() === delim) {
          found = end;
          break;
        }
        if (nl === -1) break;
        j = nl + 1;
      }
      if (found === -1) break;
      if (feed) bodies.push(src.slice(i, found - delim.length));
      i = Math.min(found + 1, src.length);
    }
    heredocs = [];
    return i;
  };

  let i = 0;
  while (i < src.length) {
    const c = src[i];
    const next = src[i + 1];
    if (mode === "sq") {
      if (c === "'") mode = "cmd";
      else add(c);
      i++;
      continue;
    }
    if (mode === "dq") {
      if (c === '"') mode = "cmd";
      else if (ps && c === "`") {
        if (next !== "\n") add(next ?? "");
        i++;
      } else if (!ps && c === "\\" && next !== undefined && '$`"\\\n'.includes(next)) {
        if (next !== "\n") add(next);
        i++;
      } else if (c === "$" && next === "(") {
        push(")");
        i++;
      } else if (!ps && c === "`") push("`");
      else add(c);
      i++;
      continue;
    }
    // mode === "cmd"
    if (close && c === close) {
      pop();
      i++;
      continue;
    }
    if (c === " " || c === "\t" || c === "\r") {
      endWord();
      i++;
    } else if (c === "\n") {
      endSeg();
      i = heredocs.length ? takeHeredocs(i + 1) : i + 1;
    } else if (c === "'") {
      add("");
      mode = "sq";
      i++;
    } else if (c === '"') {
      add("");
      mode = "dq";
      i++;
    } else if ((!ps && c === "\\") || (ps && (c === "`" || c === "^"))) {
      if (next !== "\n" && next !== undefined) add(next);
      i += 2;
    } else if (c === "$" && next === "(") {
      endWord();
      push(")");
      i += 2;
    } else if (!ps && c === "$" && (next === "'" || next === '"')) {
      i++; // bash $'...' and $"..." quoting: the quote that follows does the work
    } else if (!ps && c === "`") {
      endWord();
      push("`");
      i++;
    } else if (ps && c === "@" && (next === "'" || next === '"') && /^[ \t]*\r?\n/.test(src.slice(i + 2))) {
      // PowerShell here-string: data up to a line starting with '@ or "@.
      const start = src.indexOf("\n", i) + 1;
      const endMark = src.indexOf("\n" + next + "@", start - 1);
      if (endMark === -1) {
        add(c);
        i++;
      } else {
        add(src.slice(start, endMark));
        i = endMark + 3;
      }
    } else if (c === ";" || c === "|" || c === "(" || c === ")") {
      endSeg();
      i++;
    } else if (ps && c === ",") {
      endWord();
      i++;
    } else if (c === "&") {
      if (src[i - 1] === ">" || src[i - 1] === "<") redir = false; // 2>&1, >&2: fd dup
      else if (next === ">") endWord(); // &> file
      else endSeg();
      i++;
    } else if (c === ">") {
      endWord();
      redir = true;
      i += next === ">" || next === "|" ? 2 : 1;
    } else if (c === "<" && next === "<" && src[i + 2] !== "<") {
      endWord();
      let j = i + 2;
      if (src[j] === "-") j++;
      while (src[j] === " " || src[j] === "\t") j++;
      let delim = "";
      while (j < src.length && !/[\s;|&<>()]/.test(src[j])) delim += src[j++];
      delim = delim.replace(/["'\\]/g, "");
      if (delim) heredocs.push({ delim, from: segments.length });
      i = j;
    } else if (c === "<") {
      endWord();
      i += next === "<" ? 3 : 1; // `<<<` here-string: the next word is plain data
    } else {
      add(c);
      i++;
    }
  }
  while (frames.length) pop();
  endSeg();
  return { segments, bodies };
}

// --- git ---------------------------------------------------------------------

// Global options that take the next word as their value.
const GIT_VALUE_OPTS = new Set([
  "-c", "-C", "--git-dir", "--work-tree", "--namespace", "--config-env", "--super-prefix",
  "--attr-source",
]);

// `t` is an accepted abbreviation of long option `full`. Git accepts any unique
// prefix, and for `reset` even `--h` is `--hard`, so the floor is `--` plus one
// letter; an ambiguous prefix is an error in git anyway, so denying it costs
// nothing. Only a *complete* benign option that prefixes a dangerous one
// (`switch --force` vs `--force-create`) needs a higher floor. Case-folded.
function long(t, full, min = 3) {
  const key = t.split("=")[0].toLowerCase();
  return key.startsWith("--") && key.length >= min && full.startsWith(key);
}

// Letters of a short-option cluster, stopping at the first option that takes a
// value (the rest of the cluster is that value). Case-sensitive, like git.
function cluster(t, stop = "") {
  if (!/^-[^-]/.test(t)) return "";
  let out = "";
  for (const ch of t.slice(1)) {
    out += ch;
    if (stop.includes(ch)) break;
  }
  return out;
}

function parseGit(words, j) {
  for (; j < words.length; j++) {
    const t = words[j];
    if (!t.startsWith("-") || t === "-") break;
    let value = null;
    if (t.startsWith("-c") && t.length > 2 && !t.startsWith("--")) value = t.slice(2);
    else if (t.toLowerCase().startsWith("--config-env=")) value = t.slice(13);
    else if (GIT_VALUE_OPTS.has(t)) value = words[++j] ?? "";
    if ((t === "-c" || t.startsWith("-c") || t.toLowerCase().startsWith("--config-env")) && /^alias\./i.test(value ?? "")) {
      return { rule: "git-inline-alias" };
    }
  }
  return { sub: (words[j] ?? "").toLowerCase(), args: words.slice(j + 1) };
}

function gitRule(sub, args) {
  const L = (full, min) => args.some((t) => long(t, full, min));
  const S = (chars, stop = "") => args.some((t) => [...cluster(t, stop)].some((ch) => chars.includes(ch)));
  const pos = args.filter((t) => !t.startsWith("-"));
  switch (sub) {
    case "commit":
      return L("--amend") ? "git-amend" : null;
    case "rebase":
      return "git-rebase";
    case "pull":
      return args.some((t) => long(t, "--rebase") && !/=(false|no|off|0)$/i.test(t)) || S("r", "sX")
        ? "git-rebase"
        : null;
    case "push":
      if (L("--force-with-lease") || L("--force-if-includes") || L("--mirror") || S("f", "o") || pos.some((t) => t.startsWith("+"))) {
        return "git-force-push";
      }
      if (L("--delete") || L("--prune") || S("d", "o") || pos.some((t) => t.startsWith(":"))) {
        return "git-push-delete";
      }
      return null;
    case "reset":
      return L("--hard") ? "git-hard-reset" : null;
    case "clean":
      return args.some((t) => long(t, "--dry-run") || cluster(t, "e").includes("n")) ? null : "git-clean";
    case "branch":
      return S("DfMC", "u") || L("--force") ? "git-branch-force" : null;
    case "checkout":
      return S("B") ? "git-branch-force" : null;
    case "switch":
      return S("C") || L("--force-create", 8) ? "git-branch-force" : null;
    case "worktree":
      return (args[0] ?? "").toLowerCase() === "remove" &&
        args.slice(1).some((t) => long(t, "--force") || cluster(t).includes("f"))
        ? "git-worktree-force-remove"
        : null;
    case "tag":
      return S("f", "mFu") || L("--force") ? "git-tag-force" : null;
    case "update-ref":
      return "git-ref-rewrite";
    case "replace": {
      const listing = args.length === 0 || args.some((t) => t === "-l" || long(t, "--list"));
      const mutating = args.some(
        (t) =>
          ["-d", "-f", "-e", "-g"].includes(t) ||
          ["--delete", "--force", "--edit", "--graft", "--convert-graft-file"].some((o) => long(t, o)),
      );
      return listing && !mutating ? null : "git-ref-rewrite";
    }
    case "symbolic-ref":
      return args.some((t) => t === "-d" || long(t, "--delete")) || pos.length >= 2 ? "git-ref-rewrite" : null;
    case "reflog":
      return ["expire", "delete"].includes((pos[0] ?? "").toLowerCase()) ? "git-ref-rewrite" : null;
    case "filter-branch":
    case "filter-repo":
      return "git-history-rewrite";
    default:
      // `git credential fill` and the credential-* helpers print stored secrets.
      return sub === "credential" || sub.startsWith("credential-") ? "secret-token" : null;
  }
}

// GitHub CLI routes around `git push`: ref PATCH/DELETE (force-update or delete
// a published branch), GraphQL ref mutations, and deleting the repository.
function ghRule(lower, k) {
  const [a, b] = [lower[k + 1], lower[k + 2]];
  const rest = lower.slice(k + 2);
  if (a === "auth" && (b === "token" || (b === "status" && rest.some((x) => x === "-t" || x === "--show-token")))) {
    return "secret-token";
  }
  if (a === "repo" && b === "delete") return "git-remote-ref-rewrite";
  if (a !== "api") return null;
  let method = "";
  rest.forEach((x, i) => {
    if (x === "-x" || x === "--method") method = rest[i + 1] ?? "";
    else if (x.startsWith("--method=")) method = x.slice(9);
    else if (/^-x[a-z]+$/.test(x)) method = x.slice(2);
  });
  const refs = rest.some((x) => x.includes("git/refs"));
  if ((refs && ["delete", "patch", "put"].includes(method)) || rest.some((x) => /deleteref|updateref/.test(x))) {
    return "git-remote-ref-rewrite";
  }
  return null;
}

// --- segments ----------------------------------------------------------------

function checkSegment({ words, redirects }) {
  if (words.some(isSecretWord)) return "secret-path";
  const names = words.map(exe);
  const lower = words.map((x) => x.toLowerCase());
  const atCommand = (k) => k === 0 || WRAPPERS.has(lower[k - 1]);
  if (names.some((n, k) => n === "printenv" && atCommand(k))) return "env-dump";
  if (
    lower.some(
      (x) =>
        /^env:[\\/]?(\S*[*?]\S*)?$/.test(x) || // the PowerShell env: drive, listed
        x.includes("getenvironmentvariables") ||
        /\/proc\/[^/]*\/environ$/.test(x.replace(/\\/g, "/")),
    )
  ) {
    return "env-dump";
  }
  const rest = lower.slice(1);
  const flagsOnly = rest.every((x) => x.startsWith("-"));
  if (
    (names[0] === "env" && flagsOnly) ||
    (names[0] === "set" && rest.length === 0) ||
    (names[0] === "export" && rest.every((x) => x === "-p")) ||
    ((names[0] === "declare" || names[0] === "typeset") && flagsOnly) ||
    (names[0] === "compgen" && rest.some((x) => x === "-e" || x === "-v"))
  ) {
    return "env-dump";
  }
  if (redirects.some(isGitMeta)) return "git-metadata-write";
  const inPlace = lower.some((x) => /^-[a-z]*i/.test(x) || x.startsWith("--in-place"));
  const writes = names.some((n) => WRITERS.has(n) || ((n === "sed" || n === "perl") && inPlace));
  if (writes && words.some(isGitMeta)) return "git-metadata-write";
  let gitWords = 0;
  for (let k = 0; k < words.length; k++) {
    if (names[k] === "gh") {
      const r = ghRule(lower, k);
      if (r) return r;
      continue;
    }
    let sub;
    let args;
    if (names[k] === "git" || names[k].startsWith("git-")) {
      if (++gitWords > MAX_GIT_WORDS) return "internal-error";
    }
    if (names[k] === "git") {
      const g = parseGit(words, k + 1);
      if (g.rule) return g.rule;
      ({ sub, args } = g);
    } else if (names[k].startsWith("git-")) {
      sub = names[k].slice(4);
      args = words.slice(k + 1);
    } else continue;
    const r = sub && gitRule(sub, args);
    if (r) return r;
  }
  return null;
}

const isEncoded = ({ words }) =>
  words.some((x) => ["pwsh", "powershell"].includes(exe(x))) &&
  words.some((x) => {
    const k = x.slice(1).toLowerCase();
    return /^[-/]/.test(x) && k.length > 0 && (k === "ec" || "encodedcommand".startsWith(k));
  });

function analyze(cmd, depth, seen) {
  if (depth > 4 || seen.has(cmd)) return null;
  seen.add(cmd);
  for (const ps of [false, true]) {
    const { segments, bodies } = lex(cmd, ps);
    for (const seg of segments) {
      const r = checkSegment(seg);
      if (r) return r;
    }
    // `[IO.File]::WriteAllText('.git/HEAD', …)`: the parenthesis puts the path in
    // its own segment, so a .NET write is judged over the whole command.
    if (DOTNET_WRITE.test(cmd.toLowerCase()) && segments.some((s) => s.words.some(isGitMeta))) {
      return "git-metadata-write";
    }
    if (segments.some((s) => s.words.some((x) => LAUNCHERS.has(exe(x))))) {
      if (segments.some(isEncoded)) return "encoded-command";
      // A launcher runs strings as commands: dissolve the quoting and look again.
      const flat = segments.map((s) => s.words.join(" ")).join("\n");
      for (const text of [flat, ...bodies]) {
        const r = analyze(text, depth + 1, seen);
        if (r) return r;
      }
    }
  }
  return null;
}

// --- entry -------------------------------------------------------------------

function decide(raw) {
  const input = JSON.parse(raw);
  if (!input || typeof input !== "object" || Array.isArray(input)) return "internal-error";
  const tool = String(input.tool_name ?? "");
  const ti = input.tool_input && typeof input.tool_input === "object" ? input.tool_input : {};
  if (typeof ti.command === "string") {
    if (ti.command.length > MAX_COMMAND) return "internal-error";
    const r = analyze(ti.command, 0, new Set());
    if (r) return r;
  }
  for (const key of ["file_path", "notebook_path", "path", "glob"]) {
    const v = ti[key];
    if (typeof v !== "string") continue;
    if (isSecret(v)) return "secret-path";
    if (WRITE_TOOLS.has(tool) && isGitMeta(v)) return "git-metadata-write";
  }
  return null;
}

let rule;
try {
  rule = decide(readFileSync(0, "utf8"));
} catch {
  rule = "internal-error";
}
if (rule) {
  const reason =
    `ProjectChimera guardrail [${rule}]: ${WHY[rule]}. Blocked by repository governance. ` +
    "Do not rephrase, split, encode or route around this; stop and report it to the owner, " +
    `who can run the operation manually if it is legitimately needed. See ${DOC}.`;
  process.stdout.write(
    JSON.stringify({
      hookSpecificOutput: {
        hookEventName: "PreToolUse",
        permissionDecision: "deny",
        permissionDecisionReason: reason,
      },
    }) + "\n",
  );
  process.stderr.write(reason + "\n");
  process.exitCode = 2;
}
