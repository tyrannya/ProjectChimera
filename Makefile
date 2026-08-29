.DEFAULT_GOAL := help
.PHONY: help setup lint format test smoke sample backfill features \
	verify-research-snapshot verify-research-state train research experiment walkforward \
	wf-diagnostics benchmark benchmark-compare \
	p2b-cell p2b-btc p2b-compare p2b-ablation p2b-regimes \
	p2c-cell p2c-btc p2c-compare freeze-evidence \
	trade-plan trade-probe trade-snapshot verify-trade-snapshot \
	p3-cell p3-btc p3-compare \
	futures-dry-run futures-dry-run-verify \
	derivatives-plan derivatives-probe derivatives-snapshot \
	verify-derivatives-snapshot p4-status p4-cell p4-btc p4-compare \
        infer dry-run docker-build docker-up docker-down docker-logs check clean

PYTHON  ?= python
EXCHANGE ?= binance
PAIR     ?= BTC/USDT
TIMEFRAME ?= 1h
START    ?= 2023-01-01
PAIR_SAFE = $(subst /,_,$(PAIR))
CANDLES  ?= data/raw/$(EXCHANGE)/$(PAIR_SAFE)_$(TIMEFRAME).parquet
DATASET  ?= data/datasets/$(EXCHANGE)_$(PAIR_SAFE)_$(TIMEFRAME).parquet
MODELS   ?= artifacts/models
# Committed research contract the research targets run under. A selector, not a
# boundary: only ids under nn/research_contracts/ are accepted, and a new
# research generation is a new contract file.
CONTRACT ?= btc-usdt-1h-gen1
EPOCHS   ?= 30
# Run seed for the P2a benchmark; the checkpoint runs 42, 142, 242, 342, 442.
SEED     ?= 42
SEQ_LEN  ?= 64
STRATEGY ?= NNPredictorStrategy
MODE     ?= test

help:  ## Show this help
	@echo "ProjectChimera — dry-run research platform"
	@echo ""
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Common variables: EXCHANGE=$(EXCHANGE) PAIR=$(PAIR) TIMEFRAME=$(TIMEFRAME)"
	@echo "                  EPOCHS=$(EPOCHS) SEQ_LEN=$(SEQ_LEN) MODE=$(MODE)"
	@echo "                  CONTRACT=$(CONTRACT)"
	@echo ""
	@echo "LIVE TRADING IS OFF. See README 'Live trading protection'."

setup:  ## Install everything and the pre-commit hooks
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	$(PYTHON) -m pip install -e ".[all]"
	pre-commit install

lint:  ## Run all pre-commit hooks over the repository
	pre-commit run --all-files

format:  ## Auto-format with black
	black chimera nn strategies tools tests

test:  ## Run the test suite
	pytest

smoke:  ## End-to-end smoke: data -> features -> training -> service -> risk
	$(PYTHON) -m tools.smoke

check: ## Everything the Definition of Done requires
	$(PYTHON) -m compileall -q .
	$(PYTHON) -m tools.verify_research_snapshot
	$(PYTHON) -m tools.verify_trade_snapshot
	$(PYTHON) -m tools.verify_research_state
	$(PYTHON) -m tools.futures_dry_run --verify $(FUTURES_DRY_RUN_DIR)
	pytest
	pre-commit run --all-files
	docker compose config --quiet
	@echo "All acceptance checks passed."

sample:  ## Generate synthetic candles (no network needed)
	$(PYTHON) -m tools.make_sample_data --rows 5000 --out data/raw/synthetic/SYNTH_USDT_1h.parquet

backfill:  ## Download candles. Args: EXCHANGE PAIR TIMEFRAME START
	$(PYTHON) -m tools.backfill --exchange $(EXCHANGE) --pair $(PAIR) \
		--timeframe $(TIMEFRAME) --start $(START)

features:  ## Build a training dataset from downloaded candles
	$(PYTHON) -m tools.build_features --candles $(CANDLES) --out $(DATASET) \
		--exchange $(EXCHANGE) --pair $(PAIR) --timeframe $(TIMEFRAME)

verify-research-snapshot:  ## Check the committed research snapshot: hashes, seal, coverage
	$(PYTHON) -m tools.verify_research_snapshot

verify-research-state:  ## Check the front-door documents against the committed evidence
	$(PYTHON) -m tools.verify_research_state

train:  ## Train a model. Args: DATASET EPOCHS SEQ_LEN CONTRACT
	$(PYTHON) -m nn.train --dataset $(DATASET) --models-dir $(MODELS) \
		--research-contract $(CONTRACT) --epochs $(EPOCHS) --seq-len $(SEQ_LEN)

research:  ## Train on validation only, leaving the test split sealed
	$(PYTHON) -m nn.train --dataset $(DATASET) --models-dir $(MODELS) \
		--research-contract $(CONTRACT) --epochs $(EPOCHS) --seq-len $(SEQ_LEN) \
		--validation-only

experiment:  ## Grid search scored on validation only. Args: DATASET EPOCHS SEQ_LEN
	$(PYTHON) -m nn.experiment --dataset $(DATASET) --epochs $(EPOCHS) \
		--research-contract $(CONTRACT) \
		--seq-len $(SEQ_LEN) --seed 1 2 3 --lr 1e-4 3e-4 1e-3 \
		--out artifacts/experiments

walkforward:  ## Nested walk-forward validation (train -> inner val -> outer val)
	$(PYTHON) -m nn.walkforward --dataset $(DATASET) --folds 4 --epochs $(EPOCHS) \
		--research-contract $(CONTRACT) \
		--seq-len $(SEQ_LEN) --out artifacts/walkforward

wf-diagnostics:  ## Audit and compare walk-forward runs. Args: RUNS="dir1 dir2 ..."
	$(PYTHON) -m nn.wf_diagnostics $(RUNS)

benchmark:  ## P2a: untuned simple models on MTST's own samples. Args: DATASET SEED
	$(PYTHON) -m nn.benchmark --dataset $(DATASET) --folds 4 \
		--research-contract $(CONTRACT) --seq-len $(SEQ_LEN) --seed $(SEED) \
		--out artifacts/benchmark/btc_p2a_seed_$(SEED)

benchmark-compare:  ## P2a vs the frozen MTST evidence. Args: BENCH="..." MTST="..."
	$(PYTHON) -m nn.benchmark_compare --benchmark $(BENCH) --mtst $(MTST) \
		--dataset $(DATASET) --out artifacts/benchmark/btc_p2a_comparison

# --- Information-set checkpoints: P2b (market structure), P2c (chart structure)
# These run from the committed research snapshot under data/research/, not from
# a locally built dataset, so a fresh clone reproduces them with no VPS, no
# private data and no access to the sealed block.
#
# `nn.p2b` verifies that snapshot itself — all 23 checks of
# `tools.verify_research_snapshot` — before it fits anything, so the
# `verify-research-snapshot` target below is a way to *see* the result, not the
# thing that makes these targets safe. Running the module directly is equally
# safe, which is the point: an integrity guarantee that depends on the operator
# choosing the right make target is a convention, not a guarantee.
#
# `--checkpoint` is required. `ohlcv14` is the control of both checkpoints, so
# the arms cannot say which research question a cell is answering and the cell
# has to be told.
P2B_SETS   ?= ohlcv14 smc_v1 ohlcv14_plus_smc_v1
P2C_SETS   ?= ohlcv14 chart_structure_v1 ohlcv14_plus_chart_structure_v1
P3_SETS    ?= ohlcv14 microstructure_v1 ohlcv14_plus_microstructure_v1
P2B_MODELS ?= logistic_regression lightgbm xgboost
P2B_DIR    ?= artifacts/benchmark
# The nine cells are independent and each estimator is pinned to one thread
# inside the runner, so running several cells at once is both reproducible and
# roughly four times faster on a four-core machine. `p2b-btc` runs them in
# sequence; see docs/research_reproduction.md for the parallel invocation.
P2B_RUNS   = $(foreach s,$(P2B_SETS),$(foreach m,$(P2B_MODELS),$(P2B_DIR)/btc_p2b_$(s)_$(m)))
P2C_RUNS   = $(foreach s,$(P2C_SETS),$(foreach m,$(P2B_MODELS),$(P2B_DIR)/btc_p2c_$(s)_$(m)))
P3_RUNS    = $(foreach s,$(P3_SETS),$(foreach m,$(P2B_MODELS),$(P2B_DIR)/btc_p3_$(s)_$(m)))
# P4's three arms, in the order docs/p4_preregistration.md §2 reports them.
FUTURES_DRY_RUN_DIR ?= artifacts/futures_dry_run_v1

P4_SETS    = ohlcv14 derivatives_v1 ohlcv14_plus_derivatives_v1
P4_RUNS    = $(foreach s,$(P4_SETS),$(foreach m,$(P2B_MODELS),$(P2B_DIR)/btc_p4_$(s)_$(m)))
# Where `trade-snapshot` stages one archive at a time. Point it at a large disk:
# the archives are deleted as they are folded, but one of them has to fit.
TRADE_WORKDIR ?= /tmp/chimera-trades
# Where `derivatives-snapshot` stages one archive at a time. Smaller than the
# trade archives by three orders of magnitude, but the same discipline applies.
DERIVATIVES_WORKDIR ?= /tmp/chimera-derivatives
PROBE_MONTHS  ?= 2

p2b-btc: verify-research-snapshot  ## P2b: all nine cells (3 information sets x 3 models x 4 folds)
	@for s in $(P2B_SETS); do for m in $(P2B_MODELS); do \
		echo "--- $$s x $$m ---"; \
		$(PYTHON) -m nn.p2b --checkpoint P2b --information-set $$s --model $$m \
			--out $(P2B_DIR)/btc_p2b_$${s}_$${m} || exit 1; \
	done; done

p2b-cell:  ## One P2b cell. Args: SET=ohlcv14 MODEL=xgboost
	$(PYTHON) -m nn.p2b --checkpoint P2b --information-set $(SET) --model $(MODEL) \
		--out $(P2B_DIR)/btc_p2b_$(SET)_$(MODEL)

p2b-compare:  ## Join the P2b cells: parity proof, recomputation, deltas
	$(PYTHON) -m nn.p2b_compare --runs $(P2B_RUNS) \
		--out $(P2B_DIR)/btc_p2b_comparison

p2c-btc: verify-research-snapshot  ## P2c: all nine cells (chart structure vs OHLCV14)
	@for s in $(P2C_SETS); do for m in $(P2B_MODELS); do \
		echo "--- $$s x $$m ---"; \
		$(PYTHON) -m nn.p2b --checkpoint P2c --information-set $$s --model $$m \
			--out $(P2B_DIR)/btc_p2c_$${s}_$${m} || exit 1; \
	done; done

p2c-cell:  ## One P2c cell. Args: SET=chart_structure_v1 MODEL=xgboost
	$(PYTHON) -m nn.p2b --checkpoint P2c --information-set $(SET) --model $(MODEL) \
		--out $(P2B_DIR)/btc_p2c_$(SET)_$(MODEL)

p2c-compare:  ## Join the P2c cells: parity proof, recomputation, deltas
	$(PYTHON) -m nn.p2b_compare --runs $(P2C_RUNS) \
		--out $(P2B_DIR)/btc_p2c_comparison

p2b-ablation:  ## Post-hoc: leave-one-family-out. Args: MODEL=xgboost
	$(PYTHON) -m nn.p2b_ablation --full $(P2B_DIR)/btc_p2b_ohlcv14_plus_smc_v1_$(MODEL) \
		--ablations $(P2B_DIR)/btc_p2b_ohlcv14_plus_smc_v1_minus_*_$(MODEL) \
		--control $(P2B_DIR)/btc_p2b_ohlcv14_$(MODEL) \
		--out $(P2B_DIR)/btc_p2b_ablation_$(MODEL)

p2b-regimes:  ## Descriptive: what the four outer periods were
	$(PYTHON) -m nn.p2b_regimes --runs $(P2B_RUNS) --out $(P2B_DIR)/btc_p2b_regimes

# --- P3: trade-level microstructure ----------------------------------------
# P3's source is not the hourly candle. `trade-snapshot` streams Binance's
# public spot aggTrades archive month by month, folds each one into hourly
# sufficient statistics and deletes it before requesting the next, so a five-year
# acquisition needs one archive of disk rather than a billion rows of it.
#
# Run `trade-plan` (no network) and then `trade-probe` (two real archives) before
# committing to the bulk download: the probe reports compressed bytes, extracted
# rows and the epoch unit actually in use, and projects the whole acquisition
# from measurements rather than from an estimate.
#
# The acquisition window is derived from the committed OHLCV snapshot and closes
# at the last hour the research spine reaches — over three months before Styx —
# so no archive that could contain a sealed trade is ever requested.
trade-plan:  ## List the aggTrades archives the acquisition would fetch. No network.
	$(PYTHON) -m tools.export_trade_snapshot --plan

trade-probe:  ## Measure real archives and project the full acquisition. Args: PROBE_MONTHS=2
	$(PYTHON) -m tools.export_trade_snapshot --probe --months $(PROBE_MONTHS)

trade-snapshot:  ## Acquire the P3 trade source and export the hourly snapshot
	$(PYTHON) -m tools.export_trade_snapshot --workdir $(TRADE_WORKDIR)

verify-trade-snapshot:  ## Check the committed trade snapshot: hashes, seal, coverage
	$(PYTHON) -m tools.verify_trade_snapshot

# As with P2b and P2c, `nn.p2b` verifies both snapshots itself before it fits
# anything. These targets are a way to see the result, not the thing that makes
# them safe.
p3-btc: verify-research-snapshot verify-trade-snapshot  ## P3: all nine cells (microstructure vs OHLCV14)
	@for s in $(P3_SETS); do for m in $(P2B_MODELS); do \
		echo "--- $$s x $$m ---"; \
		$(PYTHON) -m nn.p2b --checkpoint P3 --information-set $$s --model $$m \
			--out $(P2B_DIR)/btc_p3_$${s}_$${m} || exit 1; \
	done; done

p3-cell:  ## One P3 cell. Args: SET=microstructure_v1 MODEL=xgboost
	$(PYTHON) -m nn.p2b --checkpoint P3 --information-set $(SET) --model $(MODEL) \
		--out $(P2B_DIR)/btc_p3_$(SET)_$(MODEL)

p3-compare:  ## Join the P3 cells: parity proof, recomputation, deltas
	$(PYTHON) -m nn.p2b_compare --runs $(P3_RUNS) \
		--out $(P2B_DIR)/btc_p3_comparison

# --- P4: derivatives positioning and carry ----------------------------------
# P4's source is not the spot tape at all. `derivatives-snapshot` streams three
# Binance USD-M archives — monthly funding, DAILY open-interest metrics, monthly
# 1h perpetual klines — reduces each to the hourly point-in-time observation the
# features are defined over, and deletes it before requesting the next.
#
# Run `derivatives-plan` (no network) and then `derivatives-probe` before
# committing to the bulk download. The probe is where the open-interest
# availability window stops being an unknown: it binary-searches the archive's
# first published day from HEAD requests alone, and matches each field's schema
# against the preregistered allow-list. It is also where P4-HOLD's own coverage
# is established — one HEAD per day of the region, no archive body — and written
# to `data/research/p4_holdout_coverage.json`, which is the only file the probe
# writes and the one §8.0's availability gate reads for the region.
#
# The window is derived from the committed OHLCV snapshot and closes at the last
# hour the research spine reaches — which is also the hour before P4-HOLD begins,
# so stage 1's own source file structurally cannot contain the region stage 1
# decides whether to open.
#
# **No P4 model runs from these targets.** `p4-cell` and `p4-btc` go through
# `nn.p4_stage1`'s interlock: `data/research/p4_stage1_authorisation.json` must
# say `authorised` *and* `--authorise-fit` must be passed. `p4-status` reports
# what would run and what is currently stopping it, and fits nothing.
derivatives-plan:  ## List the derivatives archives the acquisition would fetch. No network.
	$(PYTHON) -m tools.export_derivatives_snapshot --plan

derivatives-probe:  ## Establish the OI window, each schema and P4-HOLD's coverage. Bounded.
	$(PYTHON) -m tools.export_derivatives_snapshot --probe

derivatives-snapshot:  ## Acquire the P4 derivatives source and export the hourly snapshot
	$(PYTHON) -m tools.export_derivatives_snapshot --workdir $(DERIVATIVES_WORKDIR)

verify-derivatives-snapshot:  ## Check the committed derivatives snapshot: hashes, bounds, coverage
	$(PYTHON) -m tools.verify_derivatives_snapshot

p4-status:  ## What P4 stage 1 would run, and what is stopping it. Fits nothing.
	$(PYTHON) -c "import json, nn.p4_stage1 as s; print(json.dumps(s.describe(), indent=2))"

p4-btc: verify-research-snapshot verify-derivatives-snapshot  ## P4: all nine cells. Needs an authorised interlock.
	@for s in $(P4_SETS); do for m in $(P2B_MODELS); do \
		echo "--- $$s x $$m ---"; \
		$(PYTHON) -m nn.p2b --checkpoint P4 --information-set $$s --model $$m \
			--authorise-fit --out $(P2B_DIR)/btc_p4_$${s}_$${m} || exit 1; \
	done; done

p4-cell:  ## One P4 cell. Args: SET=derivatives_v1 MODEL=xgboost
	$(PYTHON) -m nn.p2b --checkpoint P4 --information-set $(SET) --model $(MODEL) \
		--authorise-fit --out $(P2B_DIR)/btc_p4_$(SET)_$(MODEL)

p4-compare:  ## Join the P4 cells: parity proof, recomputation, deltas
	$(PYTHON) -m nn.p2b_compare --runs $(P4_RUNS) \
		--out $(P2B_DIR)/btc_p4_comparison

# Covers primary evidence only — cells and their per-sample predictions.
# Comparisons and ablation tables are derived: `tools.freeze_evidence` refuses
# to hash them, and the test suite regenerates them and checks what they say.
# --- Futures Execution v1 (engineering, not a research checkpoint) ----------
# Dry-run only: no credential, no network, no live order. `futures-dry-run` runs
# the frozen operational protocol in `tools/futures_dry_run.py` and writes the
# evidence; `futures-dry-run-verify` rechecks a committed report against the
# protocol hash in this build, which is what stops an invariant being weakened
# after it failed. Neither target selects a model, a feature or a threshold, and
# the simulated PnL in the report describes the fill model, not a market.
futures-dry-run:  ## Run the frozen futures dry-run validation protocol
	$(PYTHON) -m tools.futures_dry_run --out $(FUTURES_DRY_RUN_DIR)

futures-dry-run-verify:  ## Recheck the committed futures dry-run report
	$(PYTHON) -m tools.futures_dry_run --verify $(FUTURES_DRY_RUN_DIR)

freeze-evidence:  ## Verify a frozen checksum manifest. Args: MANIFEST=artifacts/....txt
	$(PYTHON) -m tools.freeze_evidence --verify $(MANIFEST)

infer:  ## Serve the promoted model on port 3000
	CHIMERA_MODELS_DIR=$(MODELS) uvicorn nn.infer_service:app --host 127.0.0.1 --port 3000

dry-run:  ## Start Freqtrade in dry-run. Args: EXCHANGE STRATEGY
	$(PYTHON) -m tools.run_bot --exchange $(EXCHANGE) --mode $(MODE) --strategy $(STRATEGY)

docker-build:  ## Build all images
	docker compose build

docker-up:  ## Start the whole stack in the background
	docker compose up -d

docker-down:  ## Stop the stack
	docker compose down

docker-logs:  ## Follow logs from all services
	docker compose logs -f

clean:  ## Remove caches and build artifacts (keeps data/ and artifacts/)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.py[co]' -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage build dist *.egg-info ray_results
	@echo "Cleaned. data/ and artifacts/ were left alone."
