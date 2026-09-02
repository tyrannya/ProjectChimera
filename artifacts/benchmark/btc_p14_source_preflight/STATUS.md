# CURRENT

## P14 — SOURCE PREFLIGHT

**SOURCE VALIDITY ONLY. NO SIGNAL WAS COMPUTED. NO FOLD WAS SCORED. NO PREDICTIVE
OR ECONOMIC NUMBER EXISTS. NO LIVE MONEY.**

| | |
| --- | --- |
| Checkpoint | **P14** |
| Preregistration | [`docs/p14_preregistration.md`](../../../docs/p14_preregistration.md) |
| Result state | `P14 NATIVE 1m TRADE-FLOW SCREEN: NOT YET RUN` |
| Archive host | `data.binance.vision` (canonical), listed through its S3 origin |
| Verdict | **PASS** |

This directory answers one question and no other: **can the sources the frozen
P14 contract names support every observation that contract will read?** They can.
It contains no return, no signal value, no correlation, no hit rate, no PnL and
no gate evaluation, and none may be inferred from it.

**It was produced before the design was frozen.** That ordering is deliberate and
it is the one operational lesson `P13` supplied: a frozen design can be defeated
by the coverage of its own preregistered sources, so source sufficiency is
established *first* from now on. `P13` had to acquire 260 objects, verify all 260,
and only then discover that 192 hours the design must hold through carry no
authorised mark. P14 establishes the equivalent fact before there is a design to
defeat.

## 1. What the contract needs

One archive family, and it is the family `multiclock_v1` already acquired and
checksum-verified for `P6`:

```
data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-YYYY-MM.zip
```

65 monthly objects, **2020-01 through 2025-05**. The start is inherited from the
committed multiclock source every clock in this repository is cut from; it was
not chosen for P14 and no later start was considered. The exclusive research
boundary is unchanged at **`2025-05-19T08:00:00+00:00`**.

Four columns are read: `open_time` (0), `close` (4), `volume` (5) and
`taker_buy_base_asset_volume` (9).

## 2. Objects and integrity

| | |
| --- | --- |
| zip objects | **65** |
| `.CHECKSUM` companions | **65** |
| **checksums verified** | **65 of 65** |
| checksums unverified | **0** |
| checksum mismatches | **0** |
| archive bytes | **142,054,937** |
| extracted member bytes | **428,752,979** |
| members per archive | 1 |

| structural check | count |
| --- | --- |
| columns, every month | **12**, no schema change across all 65 |
| header rows | none |
| timestamp units | **60 months milliseconds, 5 months microseconds** |
| duplicate `open_time` | **0** |
| contradictory duplicates | **0** |
| malformed rows | **0** |
| non-finite values | **0** |
| negative volume or taker volume | **0** |
| rows where `taker_buy_base > volume` | **0** |
| off-grid rows | **0** |

Every object was fetched with its published `.CHECKSUM` companion and its bytes
independently re-hashed; identity is Binance's own digest, not the hostname that
served it.

**The two timestamp eras are confirmed independently.** `multiclock_v1` §2.1
recorded 60 millisecond months and 5 microsecond months from its own acquisition.
This preflight resolved the unit per file from the magnitude of `open_time` and
found exactly the same split.

## 3. Coverage, against a 2,848,320-minute reference grid

| | |
| --- | --- |
| rows parsed | **2,845,995** |
| rows at or after the boundary, truncated at load | **18,240** |
| rows before the boundary | **2,827,755** |
| missing minutes | **2,325** |
| missing intervals | **15** |
| longest missing run | **354 minutes (5h54m)**, `2020-02-19T11:36Z` → `2020-02-19T17:30Z` |
| last missing interval | `2023-03-24T12:40Z` |

The fifteen intervals are Binance's own outages. `multiclock_v1` §5 already
enumerates **15 discontinuities, the longest 5h54m on 2020-02-19 and the last on
2023-03-24**, from a separate acquisition of the same objects. This preflight
reproduces that count, that maximum and that final date exactly.

## 4. Sufficiency — the load-bearing statement

```
archive rows before the boundary   2,827,755
committed 1m price grid rows       2,827,755
```

`data/research/btc_usdt_multiclock_gen2_1m_pre_boundary.parquet` holds 2,827,755
rows, and the archives hold exactly 2,827,755 rows before the boundary. **Every
bar of the committed price grid is one row of these archives, and the taker-side
split is present and finite on every one of them.**

So the P14 information family has **no gap surface of its own**. It cannot be
missing where the price grid is present, because it is carried on the same row.
A minute Binance did not publish produces no bar in either source, so no decision
row is created for it by one and not the other.

That is the P13 failure mode made **structurally impossible** rather than merely
measured. P13's mark-price family lived in a separate archive with its own
independent holes — 192 hours absent from months whose object *was* published,
the longest run 96 hours — and those holes fell inside windows the design had to
audit. Nothing of that shape can arise here.

**211 zero-volume minutes** exist in the span, 72 of them inside a scored outer
block, the first at `2020-12-21T13:20Z` and the last at `2023-03-24T12:39Z`. The
frozen definition gives such a bar `tfi_ratio = 0` and excludes it from every
agreement denominator. That rule is in the preregistration; the counts are
recorded here so it can never be mistaken for a response to one.

## 5. The sign convention, proved rather than assumed

The whole design rests on one semantic claim: that
`taker_buy_base_asset_volume` is aggressive *buying*. It is proved against the
publisher's own trade tape rather than read off a column name.

Three days were fixed before the check ran — one per timestamp era and one
mid-history — and for each, the daily `aggTrades` archive and the daily 1m kline
archive were both fetched, both checksum-verified, and compared minute by minute:

| | |
| --- | --- |
| days | `2020-01-15`, `2023-06-15`, `2025-03-14` |
| archives checksum-verified | **6 of 6** |
| aggregated trades read | **2,694,256** |
| `agg_trade_id` strictly increasing | yes |
| minutes compared | **4,320** |
| minutes where `volume` agrees | **4,320** |
| minutes where `taker_buy_base` agrees | **4,320** |
| tolerance | relative `1e-8` |
| max relative difference, `volume` | `3.4e-14` |
| max relative difference, `taker_buy_base` | `1.3e-14` |
| minutes present in only one source | **0** |

Summing `aggTrades` `quantity` over the rows where `is_buyer_maker` is **false**
reproduces `taker_buy_base_asset_volume` on every minute checked, to float64
summation noise, in both eras. This is an **identity, not an approximation**, and
it establishes three things at once: that the kline field is the aggressive-buy
volume, that `is_buyer_maker = false` is an aggressive BUY, and that reading the
signal from the kline archive rather than re-summing 3.4 billion trades changes
nothing about what the number means.

## 6. What was deliberately not done

No other venue, no other symbol, no other market type, no REST endpoint standing
in for the archive, no third-party mirror, no interpolated row, no synthetic
series, and no relaxation of the boundary. No predictive or economic quantity was
computed. `P4-HOLD` was not read and Styx was not approached.

**The raw archives are not committed.** They are described here by published
path, published digest and byte size, and are re-fetchable from the canonical
base URL. Per-object provenance is in
[`preflight_manifest.json`](preflight_manifest.json).
