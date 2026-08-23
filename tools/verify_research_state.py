"""Check that the front-door documents agree with the committed evidence.

    python -m tools.verify_research_state            # check, exit non-zero on drift
    python -m tools.verify_research_state --write    # rewrite the state blocks

Run by ``make check`` and by ``tests/test_research_state.py``. See
:mod:`nn.research_state` for what is checked and why it is a generated block
rather than a set of forbidden sentences.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nn.research_state import (
    FRONT_DOOR_DOCUMENTS,
    checkpoint_states,
    existing_block,
    render_block,
    verify,
)

ROOT = Path(__file__).resolve().parent.parent


def write_blocks(root: Path) -> list[str]:
    """Replace each document's state block with the current one.

    Only rewrites a document that already has a block: inserting one requires
    choosing where it belongs on the page, which is an editorial decision and
    not this tool's to make.
    """
    expected = render_block(checkpoint_states(root))
    written: list[str] = []
    for name in FRONT_DOOR_DOCUMENTS:
        path = root / name
        if not path.is_file():
            continue
        text = path.read_text()
        block = existing_block(text)
        if block is None or block == expected:
            continue
        path.write_text(text.replace(block, expected))
        written.append(name)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--write",
        action="store_true",
        help="rewrite the state blocks in place instead of only reporting drift",
    )
    args = parser.parse_args(argv)

    if args.write:
        for name in write_blocks(args.root):
            print(f"rewrote the research-state block in {name}")

    problems = verify(args.root)
    if problems:
        print("research state REJECTED")
        for problem in problems:
            print(f"  {problem}")
        return 1
    states = checkpoint_states(args.root)
    print(
        "research state verified: "
        + ", ".join(f"{name}={state}" for name, state in states.items())
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
