"""P6-EXT: the extension's design, its chronology, and its evidence.

The property this file exists to protect is that **P6-EXT is P6's design on two
further clocks and not a second, drifting one**. Everything except the clock set
is imported from `nn.p6_preregistration`, so the test that matters most is the
one asserting the two payloads agree field by field on every shared key.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from nn import p6_extension_preregistration as ext
from nn import p6_preregistration as p6
from nn.p6 import registration

REPO = Path(__file__).resolve().parents[1]
DOCUMENT = REPO / "docs" / "p6_extension_preregistration.md"
BENCHMARK = REPO / "artifacts" / "benchmark"
DECISION = BENCHMARK / "btc_p6ext_decision" / "decision.json"
MANIFEST = REPO / "artifacts" / "btc_p6ext_SHA256SUMS.txt"

FROZEN_HASH = "sha256:f0ce8bb4281389df5c877f20c88228350b2a20477ce36ad77da4acb7719c5804"

CELLS = [(clock, model) for clock in ext.CLOCKS for model in p6.MODELS]


def _flat(text: str) -> str:
    stripped = text.replace("`", "").replace("*", "").replace(">", " ")
    return re.sub(r"\s+", " ", stripped).lower()


# --------------------------------------------------------------------------- #
# A. it is P6's design, on two more clocks
# --------------------------------------------------------------------------- #


def test_the_two_clocks_are_the_ones_p6_did_not_cover():
    assert ext.CLOCKS == ("4h", "1d")
    assert set(ext.CLOCKS).isdisjoint(p6.CLOCKS)
    assert set(ext.CLOCKS) | set(p6.CLOCKS) == set(
        ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    )
    from nn.multiclock import ALL_CLOCKS

    assert set(ALL_CLOCKS) == set(ext.CLOCKS) | set(p6.CLOCKS)


def test_every_shared_design_field_is_p6s_own_object():
    """Imported, not restated: the two payloads agree on every shared key."""
    theirs, ours = p6.payload(), ext.payload()
    shared = set(theirs) & set(ours)
    # The keys that must differ are exactly the ones naming the checkpoint.
    differ = {
        "checkpoint",
        "question",
        "hypothesis",
        "evidence_ceiling",
        "research_classification",
        "clocks",
        "horizons",
        "measured_universe",
        "forbidden_after_results",
        "stopping_rule",
        "artifact_policy",
    }
    for key in sorted(shared - differ):
        assert theirs[key] == ours[key], f"{key} drifted between P6 and P6-EXT"
    assert shared - differ, "nothing is shared, so nothing was imported"


def test_the_gate_is_p6s_gate_unchanged():
    assert ext.payload()["viability_gate"] == p6.VIABILITY_GATE
    assert ext.payload()["horizon_bars"] == p6.HORIZON_BARS == 6
    assert ext.payload()["seq_len"] == p6.SEQ_LEN == 1
    assert ext.payload()["fold_periods"] == [dict(row) for row in p6.FOLD_PERIODS]


def test_the_horizons_are_six_native_bars():
    import pandas as pd

    assert ext.HORIZONS == {"4h": "1 day", "1d": "6 days"}
    assert pd.Timedelta(ext.HORIZONS["4h"]) == 6 * pd.Timedelta("4h")
    assert pd.Timedelta(ext.HORIZONS["1d"]) == 6 * pd.Timedelta("1d")


def test_it_names_the_p6_it_extends_by_hash():
    assert ext.EXTENDS["preregistration_hash"] == p6.preregistration_hash()
    assert ext.EXTENDS["clocks"] == list(p6.CLOCKS)


def test_it_says_plainly_that_it_is_not_p5():
    flat = _flat(ext.NOT_P5)
    assert "context columns attached to a 1h row" in flat
    assert "different objects" in flat
    assert _flat("It is not P5") in _flat(DOCUMENT.read_text())


def test_the_thinness_of_the_1d_universe_is_recorded_before_the_verdict():
    assert ext.MEASURED_UNIVERSE["1d"]["region_rows"] < 2_000
    flat = _flat(ext.THINNESS_NOTE)
    assert "changes no condition of the gate" in flat
    assert "reason to distrust a positive" in flat


def test_the_hash_is_frozen_and_the_document_publishes_it():
    assert ext.preregistration_hash() == FROZEN_HASH
    assert FROZEN_HASH in DOCUMENT.read_text()


def test_the_runner_executes_this_registration_and_no_other_clock():
    registered = registration("p6ext")
    assert registered.checkpoint == "P6-EXT"
    assert registered.clocks == ext.CLOCKS
    assert registered.preregistration_hash == FROZEN_HASH
    assert registered.prefix == "btc_p6ext"
    with pytest.raises(SystemExit, match="unknown registration"):
        registration("p9")


def test_every_forbidden_item_appears_in_the_document():
    section = _flat(DOCUMENT.read_text().split("## 7. Forbidden")[1].split("## 8.")[0])
    for item in ext.FORBIDDEN_AFTER_RESULTS:
        assert _flat(item) in section, f"the document does not forbid: {item}"


# --------------------------------------------------------------------------- #
# B. chronology and evidence
# --------------------------------------------------------------------------- #


def test_p6_ext_was_registered_after_p6_closed():
    p6_decision = BENCHMARK / "btc_p6_decision" / "decision.json"
    assert p6_decision.is_file()
    assert (
        json.loads(p6_decision.read_text())["preregistration_hash"]
        == p6.preregistration_hash()
    )


P6EXT_EVIDENCE_EXPECTED = True


def test_no_p6ext_artifact_exists_before_the_evidence_commit():
    if P6EXT_EVIDENCE_EXPECTED:
        pytest.skip("P6-EXT evidence is expected; its own tests check it")
    found = sorted(p.name for p in BENCHMARK.glob("btc_p6ext*"))
    assert (
        found == []
    ), f"P6-EXT artifacts exist while its preregistration is all there is: {found}"


@pytest.mark.skipif(not DECISION.is_file(), reason="P6-EXT has not been run")
def test_the_decision_reports_both_clocks_under_the_frozen_design():
    payload = json.loads(DECISION.read_text())
    assert payload["checkpoint"] == "P6-EXT"
    assert payload["preregistration_hash"] == FROZEN_HASH
    assert [row["clock"] for row in payload["clocks"]] == list(ext.CLOCKS)
    assert payload["decided_by"] == p6.PRIMARY_MODEL
    assert "best_clock" not in payload


@pytest.mark.skipif(not DECISION.is_file(), reason="P6-EXT has not been run")
@pytest.mark.parametrize("clock,model", CELLS, ids=[f"{c}-{m}" for c, m in CELLS])
def test_each_extension_cell_declares_the_frozen_design(clock, model):
    cell = json.loads((BENCHMARK / f"btc_p6ext_{clock}_{model}" / "p6.json").read_text())
    assert cell["checkpoint"] == "P6-EXT"
    assert cell["preregistration_hash"] == FROZEN_HASH
    assert cell["clock"] == clock and cell["model"] == model
    assert cell["horizon_bars"] == 6
    assert cell["config"] == {"seed": 42, "seq_len": 1, "min_trades": 10}


@pytest.mark.skipif(not MANIFEST.is_file(), reason="P6-EXT has not been frozen")
def test_the_extension_evidence_is_frozen_and_still_hashes():
    from tools.freeze_evidence import check

    assert check(MANIFEST) == []


# --------------------------------------------------------------------------- #
# D. what a re-run reproduces, and the one field it does not
# --------------------------------------------------------------------------- #


def test_the_decision_regenerates_from_the_frozen_cells_except_one_prose_field():
    """The disclosed divergence, pinned so it cannot quietly widen.

    P6-EXT's decision record was produced while `nn.p6_decision` read the
    stopping rule from `nn.p6_preregistration` for every registration, so the
    committed `interpretation` is **P6's** `on_all_fail` — which names P7, and
    was written into a record closed after P7 had already closed. The module now
    reads the registration's own stopping rule.

    A closed checkpoint does not rewrite its evidence to make a sentence right,
    so the committed record keeps the wrong sentence and this test states exactly
    what differs: one prose field, no number, no condition, no verdict, no
    outcome. `docs/p6_extension_preregistration.md`'s closure says the same in
    prose.
    """
    from nn.p6_decision import build

    committed = json.loads(DECISION.read_text())
    regenerated = build(
        sorted((BENCHMARK.glob("btc_p6ext_4h_*"))) + sorted(BENCHMARK.glob("btc_p6ext_1d_*")),
        registration("p6ext"),
    )

    differing = [key for key in committed if committed[key] != regenerated.get(key)]
    assert differing == ["interpretation"], differing
    assert set(regenerated) == set(committed)

    assert committed["interpretation"] == p6.STOPPING_RULE["on_all_fail"]
    assert regenerated["interpretation"] == ext.STOPPING_RULE["on_fail"]
    assert "P7" not in regenerated["interpretation"]


def test_p6s_own_decision_still_regenerates_byte_for_byte():
    """The same edit must not have moved P6, which is closed and reproduced."""
    from nn.p6_decision import build

    committed = json.loads((BENCHMARK / "btc_p6_decision" / "decision.json").read_text())
    runs = [
        directory
        for clock in p6.CLOCKS
        for directory in sorted(BENCHMARK.glob(f"btc_p6_{clock}_*"))
    ]
    assert build(runs, registration("p6")) == committed
