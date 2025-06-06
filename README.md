# Crypto Trading Bot

Минималистичный каркас для торгового бота, поддерживающего rule-based стратегии и нейросетевые модули. Инфраструктура включает Prometheus и Grafana для мониторинга, а также готова к быстрой разработке и тестированию.

## Структура каталогов

- `conf/` — конфигурационные файлы.
- `strategies/` — rule-based стратегии.
- `nn/` — модули нейросети.
- `tools/` — вспомогательные скрипты.

## Запуск через Docker Compose

```bash
docker compose up -d
```

Команда поднимет контейнеры бота, сервис нейросети, Prometheus и Grafana. При необходимости перед запуском можно выполнить `docker compose build`.

## Скрипт `tools/start.sh`

Скрипт собирает конфигурацию из `conf/base.json` и `<exchange>.<mode>.json`, после чего стартует Freqtrade. Примеры:

```bash
./tools/start.sh binance live  # запуск в боевом режиме
./tools/start.sh bybit test    # dry-run на тестовой бирже
```

## Обучение нейросети и сервис инференса

```bash
python nn/train.py --features path/to/features.parquet --epochs 20
```

После обучения модель сохраняется в каталоге `nn/`. Сервис инференса BentoML можно запустить отдельно:

```bash
docker compose up -d nn_infer
```
=======
## Running tests

Install development dependencies and execute pytest:

```bash
pytest
```
 main
