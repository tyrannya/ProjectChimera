#!/bin/bash

set -e

if [ $# -ne 2 ]; then
  echo "Usage: $0 <exchange> <mode>"
  echo "  <exchange>: binance | bybit | okx"
  echo "  <mode>: live | test"
  exit 1
fi

EXCHANGE="$1"
MODE="$2"

CONFIG_DIR="$(dirname "$(realpath "$0")")/../conf"
BASE_CONFIG="$CONFIG_DIR/base.json"
OVERRIDE_CONFIG="$CONFIG_DIR/${EXCHANGE}.${MODE}.json"
MERGED_CONFIG="/tmp/${EXCHANGE}_${MODE}_config.json"

if [ ! -f "$BASE_CONFIG" ] || [ ! -f "$OVERRIDE_CONFIG" ]; then
  echo "Config files not found!"
  exit 2
fi

jq -s '.[0] * .[1]' "$BASE_CONFIG" "$OVERRIDE_CONFIG" > "$MERGED_CONFIG"

if [ "$MODE" = "test" ]; then
  EXTRA_ARGS="--dry-run"
else
  EXTRA_ARGS=""
fi

echo "Merged config: $MERGED_CONFIG"
echo "Extra args: $EXTRA_ARGS"

# Пример запуска freqtrade, адаптируйте под свой docker/cli запуск
docker run --rm \
  --env-file "$(dirname "$CONFIG_DIR")/.env" \
  -v "$(dirname "$CONFIG_DIR")/strategies:/freqtrade/user_data/strategies" \
  -v "$MERGED_CONFIG:/freqtrade/user_data/config/config.json" \
  freqtradeorg/freqtrade:stable trade $EXTRA_ARGS --config /freqtrade/user_data/config/config.json

