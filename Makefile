.DEFAULT_GOAL := help
.PHONY: help setup lint format test smoke sample backfill features train \
	research experiment walkforward wf-diagnostics \
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
SEQ_LEN  ?= 64
STRATEGY ?= NNPredictorStrategy
MODE     ?= test

help:  ## Show this help
	@echo "ProjectChimera — dry-run research platform"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
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
