"""Does this network receive the streams the acceptance run needs? Ask before running.

PR-05's acceptance criterion is a 60-minute run that records **all** the streams
the gen3 contract declares. An hour is a long time to discover that one of them
was never going to arrive, and a recorder that ran for an hour and captured five
of six streams is not a failure of the recorder — so this asks first, in under a
minute.

**It imports nothing from this repository.** No ``chimera``, no ``tools``, no
contract, no parser. That is the whole point: when it says the venue delivered a
stream, the claim is about the venue and the network and cannot be an artefact of
the recorder's own code, and when it says a stream is absent, the recorder is not
what is being blamed. It speaks the websocket protocol through the same library
the recorder uses and reads the raw frames itself.

Public market data only. No API key is created, read, stored or sent, no request
is signed, and no authenticated endpoint is named.

Usage::

    python -m tools.recorder_preflight            # 25 s per host
    python -m tools.recorder_preflight --seconds 60

Exit code 0 means every required stream delivered at least one frame and the
60-minute acceptance run may begin. Exit code 1 means it may not.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import time
from typing import Any, Mapping, Sequence

from websockets.asyncio.client import connect

#: The two public market-data hosts, as section 4.1 names them.
UM_WS = "wss://fstream.binance.com/ws"
SPOT_WS = "wss://stream.binance.com:9443/ws"

#: What the acceptance run needs from each host: the venue's stream name, the
#: event type its frames carry, and the recorder stream id it feeds. A stream
#: with no frames in the window is a stream the run cannot record.
UM_REQUIRED: tuple[tuple[str, str, str], ...] = (
    ("btcusdt@kline_1m", "kline", "um.kline_1m"),
    ("btcusdt@markPrice@1s", "markPriceUpdate", "um.markPrice"),
    ("btcusdt@bookTicker", "bookTicker", "um.bookTicker"),
)
SPOT_REQUIRED: tuple[tuple[str, str, str], ...] = (
    ("btcusdt@kline_1m", "kline", "spot.kline_1m"),
    # Spot's bookTicker publishes an update id and the four book fields and no
    # event type at all, so it is recognised by its shape. That asymmetry is
    # real, and a preflight that looked only at `e` would report it missing.
    ("btcusdt@bookTicker", "", "spot.bookTicker"),
)

DEFAULT_SECONDS = 25.0
RECV_TIMEOUT_S = 6.0
OPEN_TIMEOUT_S = 15.0


def classify(frame: Mapping[str, Any]) -> str:
    """The event type, or the empty string for a frame that carries none."""
    kind = frame.get("e")
    if isinstance(kind, str):
        return kind
    if {"u", "b", "B", "a", "A"} <= set(frame):
        return ""
    return "?"


async def probe(url: str, names: Sequence[str], seconds: float) -> collections.Counter:
    """Subscribe and count frames by event type until ``seconds`` have passed."""
    counts: collections.Counter = collections.Counter()
    async with connect(url, ping_interval=None, open_timeout=OPEN_TIMEOUT_S) as socket:
        await socket.send(json.dumps({"method": "SUBSCRIBE", "params": list(names), "id": 1}))
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                message = await asyncio.wait_for(socket.recv(), timeout=RECV_TIMEOUT_S)
            except asyncio.TimeoutError:
                break
            frame = json.loads(message)
            if not isinstance(frame, Mapping) or "result" in frame or "error" in frame:
                continue
            counts[classify(frame)] += 1
    return counts


async def check(label: str, url: str, required, seconds: float) -> bool:
    print(f"{label:<6} {url}")
    try:
        counts = await probe(url, [name for name, _, _ in required], seconds)
    except Exception as exc:  # noqa: BLE001 - the reason is what the reviewer needs
        print(f"   could not probe: {type(exc).__name__}: {exc}")
        print(f"   {'ALL STREAMS':<26} {'':<18} {'':>8}  FAIL")
        return False
    ok = True
    for name, kind, stream_id in required:
        frames = counts.get(kind, 0)
        verdict = "PASS" if frames else "FAIL"
        shown = kind or "(no event type)"
        print(f"   {name:<22} {shown:<18} {frames:>8} frames  {verdict}  -> {stream_id}")
        ok = ok and frames > 0
    return ok


async def run(seconds: float) -> int:
    print(
        "Preflight for the PR-05 acceptance run. Public market data only, no "
        "credentials, and nothing from this repository is imported.\n"
    )
    um = await check("USD-M", UM_WS, UM_REQUIRED, seconds)
    spot = await check("spot", SPOT_WS, SPOT_REQUIRED, seconds)
    print()
    if um and spot:
        print(
            "PREFLIGHT PASS — every required stream delivered. This network may run\n"
            "the 60-minute acceptance run:\n"
            "    make recorder-acceptance RECORDER_SECONDS=3600 RECORDER_BASE_DIR=<scratch>"
        )
        return 0
    missing = []
    if not um:
        missing.append("USD-M")
    if not spot:
        missing.append("spot")
    print(
        f"PREFLIGHT FAIL — {' and '.join(missing)} did not deliver every required "
        "stream.\nDo NOT start the acceptance run here: it would spend an hour "
        "recording an\nincomplete set and could not satisfy the criterion. Run it from a "
        "network\nthat passes this check. Nothing about the recorder is implicated by this\n"
        "result — no recorder code took part in it."
    )
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.recorder_preflight",
        description=(
            "Check that this network receives every public market-data stream the "
            "PR-05 acceptance run records. Imports no recorder code."
        ),
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=DEFAULT_SECONDS,
        help=f"how long to listen to each host (default: {DEFAULT_SECONDS:.0f})",
    )
    args = parser.parse_args(argv)
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    return asyncio.run(run(args.seconds))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
