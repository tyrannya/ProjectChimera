"""R1-n's backup drill, and the four ways a backup lies.

A drill that only ever ran against a healthy archive would prove nothing about
the day the archive is wrong, so every test here has a twin: the round trip that
must pass, and a deliberately damaged restore that must be reported. The damage
is planted by editing the *restored* tree after extraction and re-running the
comparison, which is the same thing a corrupt archive, a partial write or a
silently dropped file would produce -- and it is checkable without needing a
genuinely broken tar.

The unsafe-target guard gets its own section. It is the only part of this tool
that can destroy something: an operator who points ``--restore-to`` at the live
recorder root would have the drill extract over the data it was protecting. Both
directions are tested, including the sibling-directory case that a naive string
prefix check gets wrong.

Nothing here touches a real recorder root; every tree is built under
``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.backup_drill import (
    Source,
    archive,
    compare,
    digest_tree,
    drill,
    main,
    restore,
    unsafe_target,
)


def build_tree(root: Path) -> Path:
    """A miniature of the two real roots: nested dirs, binary and text, a log."""
    (root / "raw/2026-10-01").mkdir(parents=True)
    (root / "normalized").mkdir(parents=True)
    (root / "health").mkdir(parents=True)
    (root / "raw/2026-10-01/kline_1m.ndjson.gz").write_bytes(bytes(range(256)) * 8)
    (root / "normalized/kline_1m.parquet").write_bytes(b"PAR1" + b"\x00" * 512)
    (root / "health/heartbeat.json").write_text(
        '{"schema": "x", "heartbeat_ns": 1}', encoding="utf-8"
    )
    (root / "health/lifecycle.ndjson").write_text(
        '{"event": "recorder.up"}\n', encoding="utf-8"
    )
    (root / "manifest.json").write_text('{"day": "2026-10-01"}', encoding="utf-8")
    return root


@pytest.fixture
def roots(tmp_path: Path) -> tuple[list[Source], Path, Path]:
    recorder = build_tree(tmp_path / "live/data")
    state = tmp_path / "live/state"
    state.mkdir(parents=True)
    (state / "carry_ledger.json").write_text('{"positions": []}', encoding="utf-8")
    (state / "risk.json").write_text('{"equity": "1000000"}', encoding="utf-8")
    (state / "decisions.ndjson").write_text('{"kind": "DECISION"}\n', encoding="utf-8")
    sources = [Source("recorder-root", recorder), Source("state-dir", state)]
    return sources, tmp_path / "backup/drill.tar.gz", tmp_path / "restore"


# --------------------------------------------------------------------------- #
# the round trip
# --------------------------------------------------------------------------- #


def test_a_clean_round_trip_reports_nothing(roots: tuple[list[Source], Path, Path]) -> None:
    sources, archive_path, restore_to = roots
    assert drill(sources, archive_path, restore_to) == []
    assert archive_path.is_file()


def test_every_file_in_both_roots_is_in_the_archive(
    roots: tuple[list[Source], Path, Path]
) -> None:
    """The positive control: a drill that archived an empty tree would pass.

    ``compare`` returns no problems when both sides are empty, so without this
    the suite could not tell a working backup from one that captured nothing.
    """
    sources, archive_path, restore_to = roots
    live_counts = {s.label: len(digest_tree(s.path)) for s in sources}
    assert live_counts == {"recorder-root": 5, "state-dir": 3}

    archive(sources, archive_path)
    restore(archive_path, restore_to)
    for source in sources:
        restored = digest_tree(restore_to / source.arcname)
        assert len(restored) == live_counts[source.label]
        assert restored == digest_tree(source.path)


def test_a_missing_source_root_is_reported_and_nothing_is_claimed(
    roots: tuple[list[Source], Path, Path]
) -> None:
    sources, archive_path, restore_to = roots
    sources[1] = Source("state-dir", sources[1].path.parent / "never-created")
    problems = drill(sources, archive_path, restore_to)
    assert problems == ["state-dir: source root does not exist"]
    assert not archive_path.exists(), "an archive was written for a root that is absent"


# --------------------------------------------------------------------------- #
# the four ways a restore can disagree
# --------------------------------------------------------------------------- #


def test_a_file_dropped_from_the_restore_is_caught(
    roots: tuple[list[Source], Path, Path]
) -> None:
    sources, archive_path, restore_to = roots
    assert drill(sources, archive_path, restore_to) == []
    (restore_to / "recorder-root/manifest.json").unlink()
    problems = compare(sources, restore_to)
    assert problems == ["recorder-root: manifest.json is missing from the restore"]


def test_a_byte_changed_in_the_restore_is_caught(
    roots: tuple[list[Source], Path, Path]
) -> None:
    """The one a file-count or a byte-total check would miss."""
    sources, archive_path, restore_to = roots
    assert drill(sources, archive_path, restore_to) == []
    victim = restore_to / "state-dir/risk.json"
    content = victim.read_text(encoding="utf-8")
    victim.write_text(content.replace("1000000", "9000000"), encoding="utf-8")
    assert len(victim.read_text(encoding="utf-8")) == len(
        content
    ), "the control changed the length too"

    problems = compare(sources, restore_to)
    assert len(problems) == 1
    assert "risk.json restored with a different digest" in problems[0]


def test_a_truncated_file_in_the_restore_is_caught(
    roots: tuple[list[Source], Path, Path]
) -> None:
    sources, archive_path, restore_to = roots
    assert drill(sources, archive_path, restore_to) == []
    victim = restore_to / "recorder-root/normalized/kline_1m.parquet"
    victim.write_bytes(victim.read_bytes()[:-1])
    problems = compare(sources, restore_to)
    assert any("kline_1m.parquet restored with a different digest" in p for p in problems)


def test_a_file_the_live_tree_does_not_have_is_caught(
    roots: tuple[list[Source], Path, Path]
) -> None:
    """A restore with extras is also a restore that is not this tree."""
    sources, archive_path, restore_to = roots
    assert drill(sources, archive_path, restore_to) == []
    (restore_to / "state-dir/stowaway.json").write_text("{}", encoding="utf-8")
    problems = compare(sources, restore_to)
    assert problems == ["state-dir: stowaway.json is in the restore but not live"]


def test_a_restore_that_produced_no_directory_is_caught(
    roots: tuple[list[Source], Path, Path]
) -> None:
    sources, archive_path, restore_to = roots
    restore_to.mkdir(parents=True)
    problems = compare(sources, restore_to)
    assert len(problems) == 2
    assert all("the restore produced no" in p for p in problems)


# --------------------------------------------------------------------------- #
# the unsafe-target guard
# --------------------------------------------------------------------------- #


def test_restoring_onto_a_live_root_is_refused(roots: tuple[list[Source], Path, Path]) -> None:
    sources, _archive, _restore = roots
    reason = unsafe_target(sources, sources[0].path)
    assert reason is not None
    assert "itself" in reason


def test_restoring_inside_a_live_root_is_refused(
    roots: tuple[list[Source], Path, Path]
) -> None:
    sources, _archive, _restore = roots
    reason = unsafe_target(sources, sources[0].path / "health/drill")
    assert reason is not None
    assert "inside" in reason


def test_restoring_into_a_parent_of_a_live_root_is_refused(
    roots: tuple[list[Source], Path, Path]
) -> None:
    """The dangerous direction: extracting there writes over the live tree."""
    sources, _archive, _restore = roots
    reason = unsafe_target(sources, sources[0].path.parent)
    assert reason is not None
    assert "would write over the live tree" in reason


def test_a_scratch_directory_beside_the_roots_is_allowed(
    roots: tuple[list[Source], Path, Path]
) -> None:
    """The other side, and the sibling-prefix case.

    ``/live/data-restore`` shares a prefix with ``/live/data`` without being
    inside it. A guard that compared strings would refuse the correct target and
    would then be loosened to pass, which is how a safety check stops checking.
    """
    sources, _archive, restore_to = roots
    assert unsafe_target(sources, restore_to) is None
    sibling = sources[0].path.parent / "data-restore"
    assert unsafe_target(sources, sibling) is None


# --------------------------------------------------------------------------- #
# the CLI
# --------------------------------------------------------------------------- #


def test_the_cli_refuses_an_unsafe_target(
    roots: tuple[list[Source], Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    sources, archive_path, _restore = roots
    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                "--recorder-root",
                str(sources[0].path),
                "--archive",
                str(archive_path),
                "--restore-to",
                str(sources[0].path),
            ]
        )
    assert exit_info.value.code == 2
    assert "refusing to restore" in capsys.readouterr().err


def test_the_cli_refuses_to_drill_nothing(
    roots: tuple[list[Source], Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    _sources, archive_path, restore_to = roots
    with pytest.raises(SystemExit) as exit_info:
        main(["--archive", str(archive_path), "--restore-to", str(restore_to)])
    assert exit_info.value.code == 2
    assert "at least one of" in capsys.readouterr().err


def test_the_cli_exit_code_follows_the_comparison(
    roots: tuple[list[Source], Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    sources, archive_path, restore_to = roots
    argv = [
        "--recorder-root",
        str(sources[0].path),
        "--state-dir",
        str(sources[1].path),
        "--archive",
        str(archive_path),
        "--restore-to",
        str(restore_to),
    ]
    assert main(argv) == 0
    assert "backup drill passed" in capsys.readouterr().out

    # And exit 1 when the restore does not match. A dirty scratch directory is
    # the realistic way this happens: extraction adds files, it does not empty
    # the target first, so yesterday's leftovers survive into today's restore
    # and the restored tree is no longer this tree.
    dirty = restore_to.parent / "dirty"
    (dirty / "state-dir").mkdir(parents=True)
    (dirty / "state-dir/leftover_from_yesterday.json").write_text("{}", encoding="utf-8")
    assert main(argv[:-1] + [str(dirty)]) == 1
    captured = capsys.readouterr().err
    assert "BACKUP DRILL FAILED" in captured
    assert "leftover_from_yesterday.json is in the restore but not live" in captured
    assert "Do not rely on it" in captured
