.PHONY: help setup lint test clean train backfill docker-build docker-up docker-down docker-logs
DEFAULT_GOAL := help

help:
	@echo "Crypto Trading Bot Makefile"
	@echo "---------------------------"
	@echo "Available commands:"
	@echo "  setup          - Install development dependencies (from requirements-dev.txt) and pre-commit hooks."
	@echo "  lint           - Run linters and formatters (pre-commit run --all-files)."
	@echo "  test           - Run tests using pytest."
	@echo "  clean          - Remove temporary build, test, and Python files (e.g., __pycache__, .pytest_cache)."
	@echo "  train          - Run neural network training (nn/train.py). Args: FEATURES, EPOCHS. Example: make train FEATURES=data/features/my_data.parquet EPOCHS=30"
	@echo "  backfill       - Run data backfilling (tools/backfill.py). Args: EXCHANGE, SYMBOL, START_DATE, TF. Example: make backfill EXCHANGE=binance SYMBOL=BTC/USDT START_DATE=2023-01-01 TF=4h"
	@echo "  docker-build   - Build or rebuild services using docker-compose."
	@echo "  docker-up      - Start services using docker-compose in detached mode."
	@echo "  docker-down    - Stop services using docker-compose."
	@echo "  docker-logs    - Follow logs for docker-compose services."

setup:
	pip install -r requirements-dev.txt
	pre-commit install
	@echo ""
	@echo "Development environment setup complete."
	@echo "For NN training, remember to install dependencies specific to the 'nn' module:"
	@echo "  cd nn && pip install -r requirements.txt && cd .."

lint:
	pre-commit run --all-files

test:
	pytest

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf ray_results/
	@echo "Cleaned temporary files."

# Note: For FEATURES in train target, using an absolute path or path relative to Makefile root is recommended.
FEATURES ?= data/features/default_feature_set.parquet # Default placeholder, user should override
EPOCHS ?= 20
train:
	@echo "Starting training with FEATURES=$(FEATURES) and EPOCHS=$(EPOCHS)..."
	@if [ ! -f "$(FEATURES)" ]; then \
		echo "Warning: Features file $(FEATURES) not found. Please specify a valid FEATURES path."; \
		echo "Example: make train FEATURES=path/to/your/features.parquet"; \
	fi
	python nn/train.py --features $(FEATURES) --epochs $(EPOCHS)

EXCHANGE ?= binance
SYMBOL ?= BTC/USDT
START_DATE ?= 2023-01-01
TF ?= 1h
backfill:
	@echo "Starting backfill for $(EXCHANGE) $(SYMBOL) from $(START_DATE) using $(TF) timeframe..."
	python tools/backfill.py $(EXCHANGE) $(SYMBOL) $(START_DATE) $(TF)

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f
