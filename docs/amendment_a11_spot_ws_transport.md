# ADOPTED A11 — spot websocket transport correction

Status: **ADOPTED 2026-09-06, pre-acquisition.**

This document is a narrow governance amendment to the authoritative engineering plan
`docs/proposed_demo_implementation_master_plan.md`. It supersedes exactly one endpoint named in
section 4.1's stream table and in the arrow table of section 3, and adds the corresponding inline
amendment block beside amendment A3, which this amendment mirrors in form and in scope.

## Correction

Section 4.1 named the spot websocket source as:

`wss://stream.binance.com:9443/ws`

That endpoint is reset by the egress network before any WebSocket frame is exchanged. This was
confirmed directly, from a Full-network Claude Code Cloud environment, four separate ways:

1. `tools/recorder_preflight.py` against the real venue: USD-M market and USD-M public both
   delivered genuine frames; spot failed with `ConnectionResetError: [Errno 104] Connection reset
   by peer`.
2. A raw WebSocket-upgrade handshake to the same host:port over plain HTTP/1.1 with the correct
   `Upgrade`/`Sec-WebSocket-*` headers: reset before any response.
3. A plain HTTPS `GET /` to the same host:port with no WebSocket semantics at all: reset
   identically.
4. The same probe against an *unrelated* host on the same port (`api.binance.com:9443`): reset
   identically, which is what identifies this as a network-layer block on the non-standard port
   9443 rather than a Binance-side refusal of this particular host.

The same host on the standard port 443 does **not** reset the connection; it completes TLS and the
WebSocket upgrade request, and Binance's own edge answers with `HTTP 451` ("restricted location").
That is a separate, still-open matter (Binance's own eligibility check on this egress IP), not a
network-layer block, and this amendment does not address it or claim it is resolved.

The corrected spot websocket source is:

`wss://data-stream.binance.vision:443/ws`

## Evidence this is the same first-party stream family

`data-stream.binance.vision` is documented in Binance's own `binance-spot-api-docs` repository,
`web-socket-streams.md`: "The base endpoint `wss://data-stream.binance.vision` can be subscribed to
receive only market data messages," with the one documented restriction being that "User data
stream is NOT available from this URL" — it is the same public Spot Market Streams family as
`stream.binance.com`, restricted to exactly the market-data traffic this recorder ever subscribes
to (it holds no user-data stream and no key).

Verified directly from this environment: a combined `SUBSCRIBE` to `btcusdt@kline_1m` and
`btcusdt@bookTicker` against `wss://data-stream.binance.vision:443/ws` completed a genuine
101-Switching-Protocols handshake, was acknowledged (`{"result": null, "id": 1}`), and delivered
live frames of both kinds over a 20-second window (10 `kline` frames, 1199 `bookTicker` frames).
Representative shapes:

- `kline`: `{"e":"kline","E":...,"s":"BTCUSDT","k":{"t":...,"T":...,"i":"1m","o":...,"c":...,"h":...,"l":...,"v":...,"n":...,"x":false,"q":...,"V":...,"Q":...,"B":"0"}}`
- `bookTicker`: `{"u":...,"s":"BTCUSDT","b":...,"B":...,"a":...,"A":...}` (no `e` field, exactly the
  shape `chimera/recorder/events.py`'s spot bookTicker parser already expects, since spot's
  bookTicker documented shape has never carried an event type)

Both shapes are byte-for-byte the documented Binance spot `kline` and `bookTicker` frame shapes.
Nothing about the payload, the field names, the event semantics or the timestamp discipline
differs from the retired endpoint; only the host differs.

## What this amendment does not establish

- **Spot REST gap-fill is untouched and unasserted.** `GET /api/v3/klines` on
  `https://api.binance.com` is a separate source (section 4.1's `spot.kline_1m` row) and remains
  subject to the same `HTTP 451` response `api.binance.com` returns generally from this egress. This
  amendment does not authorise substituting `data-api.binance.vision` (or any other host) for it —
  that would be a second amendment, decided on its own evidence, not folded into this one.
- **USD-M REST is untouched.** `fapi.binance.com` remains blocked by the same `HTTP 451` response.
  This amendment names no USD-M endpoint change and none is authorised by it.
- **This is not a claim that PR-06's acceptance criterion is met.** Reachability is a precondition
  for recording two genuine days; it is not the recording itself.

## Classification and scope

This is a **pre-acquisition transport correction**, in the same class as amendment A3: no
economically or scientifically meaningful value changes, only which host a socket is opened
against.

Unchanged:

- `contract_id` and every field of `RecorderContract`;
- the contract canonical material and `contract_hash` (`canonical_material` in
  `chimera/recorder/contract.py` hashes no endpoint; the websocket base is a Python constant in
  `chimera/recorder/streams.py`);
- every recorder stream id;
- payload semantics and parsing;
- event/canonical timestamp semantics;
- coverage requirements (`chimera/recorder/coverage.py`) and the gate thresholds;
- every scientific criterion, gate, and firewall in `docs/current_development_plan.md` and
  `docs/research_roadmap.md`;
- `prospective_from = null`.

This amendment authorises exactly one engineering change: `SPOT_WS_BASE` in
`chimera/recorder/streams.py` and the corresponding constant in `tools/recorder_preflight.py` move
from `wss://stream.binance.com:9443/ws` to `wss://data-stream.binance.vision:443/ws`, together with
the narrowly required test and documentation updates that name the old endpoint. It authorises no
other behavioural change, no REST endpoint substitution, no venue substitution, and no change to
PR-06 (`chimera/recorder/reconcile.py`, `chimera/recorder/coverage.py`) or to PR #76 itself.
