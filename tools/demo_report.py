"""Section 11.4's reports on the command line: one day, or one month that refuses.

    python -m tools.demo_report --config conf/demo/pvc1.json --day 2026-10-14
    python -m tools.demo_report --config conf/demo/pvc1.json --day 2026-10-14 --markdown
    python -m tools.demo_report --config conf/demo/pvc1.json --month 2026-10 --freeze

The configuration is read for exactly one thing -- where the runner's state
directory is, and therefore where the decision log is -- plus, for a month, the
campaign identity the artifact would be filed under. Nothing here reads a
market, opens a socket, or touches a credential, and the daily report writes no
file at all: it prints, and an operator redirects it.

**`--month --freeze` refuses on this repository, today, and that is the point.**
``conf/demo/pvc1.json`` carries ``protocol_hash: null`` and the prospective
protocol it names has not been written, so :func:`_binding_from` can build no
protocol binding and :func:`chimera.demo.reports.monthly_report` refuses before
anything is computed. The refusal is reached before the first ``mkdir``, so a
refused run leaves the filesystem exactly as it found it -- which is asserted by
a test rather than promised here. Nothing under ``artifacts/prospective/`` is
committed by this change, and no row is added to ``artifacts/README.md``'s
research table: an engineering artifact answers no research question, and a
prospective one cannot exist until a protocol has been preregistered for it.

``tools/demo_run.py`` keeps its own ``report --day`` subcommand and this module
does not replace it. That one counts the records in one day FILE and is pinned
by ``tests/test_demo_cli.py``; this one selects by each record's own ``minute``,
because the runner files the 23:59 minute into the next day's file. The two
therefore answer different questions and would disagree by one record most days,
and silently redirecting the older one would change numbers a committed test
already states.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from chimera.demo.config import ConfigProfile, DemoConfig, DemoConfigError, load_demo_config
from chimera.demo.reports import (
    ProtocolBinding,
    ReportError,
    ReportRefused,
    daily_report,
    monthly_report,
    render_daily_markdown,
    render_monthly_status,
)

EXIT_OK = 0
#: The inputs are unreadable: a torn line the verifier cannot bound, a money
#: field that is not a decimal, a day that is not a date. A defect to fix.
EXIT_ERROR = 1
#: The inputs are readable and the report must not be produced from them. Kept
#: apart from EXIT_ERROR because an operator's response is different: there is
#: nothing here to repair.
EXIT_REFUSED = 2

#: Where a frozen month would be filed, relative to the working directory. The
#: manifest that covers it goes beside the other checkpoint manifests, so a
#: reader of ``artifacts/`` finds prospective evidence pinned the same way the
#: historical evidence is.
PROSPECTIVE_ROOT = Path("artifacts") / "prospective"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="demo_report",
        description="Section 11.4's reports over the decision log.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--profile",
        default=ConfigProfile.CAMPAIGN.value,
        choices=[profile.value for profile in ConfigProfile],
        help="which profile the configuration must declare",
    )
    parser.add_argument(
        "--parity-dir",
        type=Path,
        default=None,
        help="--day only: where <day>.json replay-parity verdicts are kept",
    )
    parser.add_argument(
        "--markdown", action="store_true", help="render Markdown instead of JSON"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--day", metavar="YYYY-MM-DD")
    group.add_argument("--month", metavar="YYYY-MM")
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="--month only: write the artifact and its checksum manifest",
    )
    return parser


def _binding_from(config: DemoConfig) -> ProtocolBinding | None:
    """The protocol binding a month would be computed under, or ``None``.

    ``None`` is the committed state and is reached through the ordinary path
    rather than through a special case, so the refusal an operator meets is the
    one :func:`chimera.demo.reports.monthly_report` states rather than a second
    message written here.

    The other branch refuses outright, and deliberately. A binding needs the
    recorder contract's hash and prospective boundary, the campaign start, the
    source identity, the protocol-authorised quantity names, the excluded minutes
    and a treatment for each of HALT, RESUME and RECOVERY. None of those is in a
    campaign configuration today, so a ``protocol_hash`` appearing there without
    the rest would mean the protocol was frozen somewhere this tool cannot see --
    and filling the remaining fields in from defaults would be this tool choosing
    the terms of the evidence. The change that freezes the protocol is the change
    that teaches this function to read it.
    """
    if config.protocol_hash is None:
        return None
    raise ReportRefused(
        f"{config.campaign_id} carries a protocol_hash but this tool cannot build the "
        "rest of the binding: the recorder contract hash, the prospective boundary, the "
        "campaign start, the source identity, the protocol-named quantities, the "
        "excluded minutes and the treatment of HALT/RESUME/RECOVERY are not in a "
        "campaign configuration. Filling them in from defaults would be choosing the "
        "terms of the evidence here. Nothing was written."
    )


def _run_day(args: argparse.Namespace, config: DemoConfig) -> int:
    """Print one day's report. Writes no file, whatever the flags say."""
    state_dir = Path(config.runner_setting("state_dir"))
    report = daily_report(state_dir, args.day, parity_dir=args.parity_dir)
    if args.markdown:
        sys.stdout.write(render_daily_markdown(report))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return EXIT_OK


def _run_month(args: argparse.Namespace, config: DemoConfig) -> int:
    """Compute one month, and only then consider writing anything.

    The order is the whole guarantee: :func:`monthly_report` is called before the
    first ``mkdir``, so a refusal leaves no directory, no partial artifact and no
    manifest behind. ``tools.freeze_evidence.freeze`` is called as a library
    function rather than shelled out to, and it refuses to overwrite an existing
    manifest -- a frozen month is a statement about what a past run produced, and
    regenerating it in place is how a result quietly becomes whatever the code
    does today.
    """
    state_dir = Path(config.runner_setting("state_dir"))
    report = monthly_report(state_dir, args.month, binding=_binding_from(config))
    status = render_monthly_status(report)
    if args.markdown:
        sys.stdout.write(status)
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    if not args.freeze:
        return EXIT_OK

    from tools.freeze_evidence import freeze, refuse_existing, relative

    out_dir = PROSPECTIVE_ROOT / config.campaign_id / args.month
    manifest = Path("artifacts") / f"{config.campaign_id}_{args.month}_SHA256SUMS.txt"

    # Both of `freeze`'s refusals, asked BEFORE anything is on disk. `freeze`
    # asks them too, but it runs after the two writes below, and by then the
    # damage is done: re-freezing a month that is already frozen would replace
    # the frozen report.json with today's bytes and only then refuse, leaving a
    # committed manifest that names a digest no file matches -- a frozen result
    # quietly becoming whatever the code does today, which is the one thing the
    # refusal exists to prevent. The second call is what makes a run from a
    # working directory outside the repository refuse before it scatters an
    # artifact carrying `evidence_class: prospective` somewhere unmanifested.
    refuse_existing(manifest)
    relative(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    # newline="\n" on both: a frozen artifact is identified by the SHA-256 of
    # its bytes, and the default text mode would translate every newline to
    # "\r\n" on Windows. The same report frozen on two machines would then
    # carry two different digests, and `tools.freeze_evidence` would read that
    # as evidence having moved.
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (out_dir / "STATUS.md").write_text(status, encoding="utf-8", newline="\n")
    freeze([out_dir], out=manifest)
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.freeze and args.month is None:
        raise SystemExit("--freeze applies to --month only")
    if args.parity_dir is not None and args.day is None:
        raise SystemExit("--parity-dir applies to --day only")

    try:
        config = load_demo_config(args.config, expected_profile=ConfigProfile(args.profile))
    except DemoConfigError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_ERROR

    try:
        if args.day is not None:
            return _run_day(args, config)
        return _run_month(args, config)
    except ReportRefused as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_REFUSED
    except ReportError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
