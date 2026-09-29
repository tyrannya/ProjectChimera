"""R1-j's acceptance witness: the 48 h synthetic fixture reaches PARITY with a
restart and an operator action.

The roadmap's item, verbatim (``docs/master_roadmap_r0_r18.md``):

    **R1-j Replay parity policy:** decide `seq` (per-kind sequence or exclusion
    of operational kinds) and `OPERATOR` (deterministic replay counterpart from a
    committed operator-action file); re-include deterministic risk fields in the
    hash; the 48 h synthetic fixture reaches PARITY with a restart and an
    operator action.

Everything here is invented by ``chimera.demo.fixtures``: synthetic engineering
data, no market data, no burned or sealed block, no economic meaning.

**The live side is the CLI, process by process.** Each step is a separate
``tools.demo_run.main`` call, so each builds a fresh runner from the files on
disk and goes through ``start()``, exactly as a restarted service or an operator
command does. The recorder's output grows between them (R1-g's incremental
publication), so each ``run --once`` catches up to what has been published:

    minutes are indexed from 2026-09-19T00:00Z (0) to 2026-09-20T23:59Z (2879)

    process 1  run --once   decides minutes    0 ..  719  (D1 00:00 .. 11:59)
    process 2  run --once   decides minutes  720 .. 1799  (restart; D1 12:00 .. D2 05:59,
                                                            crossing the UTC day)
    process 3  flatten      after minute 1799 (D2 05:59), with the operator's note
    process 4  run --once   decides minutes 1800 .. 2879  (D2 06:00 .. 23:59)

The restart (process 2) comes 18 hours before the flatten, and the flatten 18
hours before the end, so the fixture shows ``seq`` staying comparable across a
restart, the flatten's effect reproduced after it, the risk state (the day roll,
the order window) deterministic throughout, and every later decision matching.

**The replay side is the tool.** ``tools.replay_parity.main`` copies the files,
replays them from an empty state directory and applies the COMMITTED
operator-action file ``tests/fixtures/r1j_operator_actions_48h.json``. The live
note below is written out here, independently of that file, so a file that
disagrees with what the operator did fails.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from chimera.demo.fixtures import MinuteShape, SyntheticFeed
from chimera.demo.risk_continuity import RISK_HASH_POLICY
from chimera.recorder.contract import load_recorder_contract
from tests.demo_harness import CARRY_PARAMS, MOMENTUM_PARAMS, REPO, SHADOW_PARAMS
from tools import demo_run, replay_parity

pytestmark = pytest.mark.replay_determinism

D1, D2, D3 = "2026-09-19", "2026-09-20", "2026-09-21"
MINUTES = 2 * 1440
RESTART_AT = 720
FLATTEN_AT = 1800
ACTIONS = REPO / "tests" / "fixtures" / "r1j_operator_actions_48h.json"

#: What the operator typed, written independently of the committed file.
LIVE_NOTE = (
    "R1-j synthetic 48 h fixture: reduce the hedge to flat after the planned "
    "restart, to prove the replay counterpart"
)

#: An operational clock in 2100. It may reach telemetry and nothing else.
OPERATIONAL_2100 = 4_102_444_800.0

FIRST_MS = int(pd.Timestamp(D1, tz="UTC").timestamp() * 1000)


def _minute_iso(index: int) -> str:
    return pd.Timestamp(FIRST_MS + index * 60_000, unit="ms", tz="UTC").isoformat()


def _publish(feed: SyntheticFeed, through: int) -> None:
    """The recorder's normalized days, published up to minute ``through`` - 1."""
    for offset, day in enumerate((D1, D2)):
        start = offset * 1440
        if through <= start:
            continue
        hidden = {
            slot: MinuteShape(present=False) for slot in range(max(0, through - start), 1440)
        }
        for market in ("um", "spot"):
            feed.write_day(market, day, hidden)


def _settlements(feed: SyntheticFeed) -> None:
    """Every settlement the 48 hours book: D1 and D2 at 00/08/16, and D3 00:00,
    the instant the last minute (D2 23:59) closes on. Nothing later."""
    path = feed.write_settlements([D1, D2, D3])
    end = FIRST_MS + MINUTES * 60_000
    kept = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["funding_time_ms"] <= end
    ]
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")


def _config(tmp_path: Path, state_dir: Path) -> Path:
    base = json.loads((REPO / "conf" / "demo" / "pvc1.json").read_text(encoding="utf-8"))
    base.update(
        {
            "profile": "TEST",
            "runner": {"state_dir": str(state_dir)},
            "rules": {
                "R1_carry": dict(CARRY_PARAMS),
                "R2_frozen_logistic": dict(SHADOW_PARAMS),
                "R3_daily_momentum": dict(MOMENTUM_PARAMS),
            },
        }
    )
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(base), encoding="utf-8")
    return path


def _records(log_dir: Path) -> list[dict]:
    out: list[dict] = []
    for path in sorted(log_dir.glob("*.ndjson")):
        out += [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]
    return out


@pytest.fixture(scope="module")
def fixture_run(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("r1j_48h")
    root = tmp_path / "recorder"
    state_dir = tmp_path / "state"
    feed = SyntheticFeed(root, load_recorder_contract("btcusdt-prospective-gen3"))
    _settlements(feed)
    config = _config(tmp_path, state_dir)
    argv = ["--config", str(config), "--root", str(root), "--profile", "TEST"]

    def cli(*command: str) -> int:
        return demo_run.main(argv + list(command), operational_clock=lambda: OPERATIONAL_2100)

    _publish(feed, RESTART_AT)
    assert cli("run", "--once") == demo_run.EXIT_OK  # process 1
    _publish(feed, FLATTEN_AT)
    assert cli("run", "--once") == demo_run.EXIT_OK  # process 2: the restart
    assert cli("flatten", "--note", LIVE_NOTE) == demo_run.EXIT_OK  # process 3
    _publish(feed, MINUTES)
    assert cli("run", "--once") == demo_run.EXIT_OK  # process 4
    live = _records(state_dir / "decision_log")

    replay_argv = argv + [
        "--live-log", str(state_dir / "decision_log"),
        "--days", D1, D2, D3,
        "--scratch", str(tmp_path / "scratch"),
        "--operator-actions", str(ACTIONS),
        "--json",
    ]  # fmt: skip
    return {
        "tmp_path": tmp_path,
        "root": root,
        "live": live,
        "replay_argv": replay_argv,
        "state_dir": state_dir,
    }


def test_the_48_hour_fixture_reaches_parity_with_a_restart_and_an_operator_action(
    fixture_run, capsys
):
    code = replay_parity.main(
        fixture_run["replay_argv"], operational_clock=lambda: OPERATIONAL_2100
    )
    report = json.loads(capsys.readouterr().out)

    assert code == replay_parity.EXIT_PARITY, report
    assert report["label"] == "PARITY" and report["parity"] is True
    assert report["divergences"] == 0
    assert report["live_only"] == [] and report["replay_only"] == []
    assert report["explained_exclusions"] == []
    assert report["operator_actions"]["file_hash"] == (
        "sha256:" + hashlib.sha256(ACTIONS.read_bytes()).hexdigest()
    )
    assert report["seq_policy"] == replay_parity.SEQ_POLICY
    # Every minute of the 48 hours has a DECISION, plus the funding, the
    # reconciliations and the two OPERATOR records.
    assert report["records_compared"] >= MINUTES + 2
    with capsys.disabled():
        print(
            "\nR1J-48H-PARITY "
            + json.dumps(
                {k: report[k] for k in ("label", "parity", "records_compared", "divergences")}
                | {
                    "live_only": len(report["live_only"]),
                    "replay_only": len(report["replay_only"]),
                    "explained_exclusions": len(report["explained_exclusions"]),
                },
                sort_keys=True,
            )
        )


def test_the_live_run_really_restarted_and_really_flattened(fixture_run):
    """The witness is not vacuous: four processes, one flatten, 48 hours."""
    live = fixture_run["live"]
    kinds = Counter(r["kind"] for r in live)
    assert kinds["STARTUP"] == 4, "three restarts: the run, the flatten, the run"
    assert kinds["SHUTDOWN"] == 3, "the flatten process writes none, like the CLI"
    assert "RECOVERY" not in kinds, "clean restarts: nothing to recover, nothing excluded"

    decided = sorted({r["minute"] for r in live if r["kind"] == "DECISION"})
    assert decided == [_minute_iso(i) for i in range(MINUTES)], "every minute, once"

    operator = [r for r in live if r["kind"] == "OPERATOR"]
    assert [(r["operator"]["command"], r["operator"]["phase"]) for r in operator] == [
        ("flatten", "requested"),
        ("flatten", "completed"),
    ]
    assert {r["minute"] for r in operator} == {_minute_iso(FLATTEN_AT - 1)}
    assert operator[0]["operator"]["note"] == LIVE_NOTE
    assert operator[1]["operator"]["request_seq"] == operator[0]["seq"]
    # The flatten reduced a real position, and the next minute re-opened it.
    before = operator[0]["operator"]["position_before"]
    assert Decimal(before["perp_qty"]) > 0 and Decimal(before["spot_qty"]) > 0
    after = operator[1]["position_after"]
    assert Decimal(after["perp_qty"]) == 0 and Decimal(after["spot_qty"]) == 0
    reopened = next(
        r for r in live if r["kind"] == "DECISION" and r["minute"] == _minute_iso(FLATTEN_AT)
    )
    assert reopened["execution"], "the minute after the flatten trades"

    # A restart shifts the persisted seq: the raw numbers the pre-R1-j
    # comparison demanded cannot be the replay's.
    starts = [r["seq"] for r in live if r["kind"] == "STARTUP"]
    assert starts[1] > starts[0] + 1


def test_the_new_hash_policy_and_real_risk_transitions_are_exercised(fixture_run):
    """Every restated hash is chimera.risk-hash/2, and the fields it re-included
    moved during the run, so the policy is exercised rather than asserted."""
    live = fixture_run["live"]
    stated = [r["risk"] for r in live if isinstance(r.get("risk"), dict)]
    assert stated and all(block["hash_policy"] == RISK_HASH_POLICY for block in stated)
    risk = json.loads((fixture_run["state_dir"] / "risk.json").read_text("utf-8"))
    # The last minute (D2 23:59) closes at D3 00:00, the decision clock's last
    # instant, so Aegis's day has rolled twice: D1 -> D2 -> D3.
    assert risk["day"] == D3, "the Aegis day follows the recorded decision clock"
    days = {
        r["minute"][:10]
        for r in live
        if r["kind"] == "DECISION" and r["ledger_effect"] and r["execution"]
    }
    assert {D1, D2} <= days, "orders on both days, so the order window moved"
    # Distinct hashes across the run: the state moved, and was restated each time.
    assert len({block["state_hash"] for block in stated}) > 10


def test_the_replay_counterpart_is_attributed_to_the_committed_file(fixture_run, tmp_path):
    argv = list(fixture_run["replay_argv"])
    scratch = argv.index("--scratch") + 1
    argv[scratch] = str(tmp_path / "scratch-attribution")
    assert replay_parity.main(argv, operational_clock=lambda: OPERATIONAL_2100) == 0
    replayed = _records(Path(argv[scratch]) / "replay_state" / "decision_log")
    operator = [r for r in replayed if r["kind"] == "OPERATOR"]
    digest = "sha256:" + hashlib.sha256(ACTIONS.read_bytes()).hexdigest()
    assert [r["operator"]["replay_action"] for r in operator] == [
        {"id": "r1j-48h-flatten-after-restart", "file_hash": digest}
    ] * 2
    kinds = Counter(r["kind"] for r in replayed)
    assert kinds["STARTUP"] == 1, "the replay itself never restarts"


@pytest.mark.parametrize("omit", ["operator_file", "flatten_in_file"])
def test_the_fixture_fails_without_its_operator_counterpart(
    fixture_run, tmp_path, capsys, omit
):
    """Two-sided: the same 48 hours without the committed counterpart diverge."""
    argv = list(fixture_run["replay_argv"])
    argv[argv.index("--scratch") + 1] = str(tmp_path / "scratch")
    position = argv.index("--operator-actions")
    if omit == "operator_file":
        del argv[position : position + 2]
    else:
        empty = tmp_path / "empty.json"
        empty.write_bytes(
            (
                json.dumps(
                    {"actions": [], "schema": "chimera.operator-actions/1"},
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("ascii")
        )
        argv[position + 1] = str(empty)
    code = replay_parity.main(argv, operational_clock=lambda: OPERATIONAL_2100)
    report = json.loads(capsys.readouterr().out)
    assert code == replay_parity.EXIT_DIVERGED
    assert any("OPERATOR" in entry and "flatten" in entry for entry in report["live_only"])
