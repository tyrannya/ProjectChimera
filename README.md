# Crypto Trading Bot

Минималистичный каркас для торгового бота, поддерживающего rule-based стратегии и нейросетевые модули. Инфраструктура включает Prometheus и Grafana для мониторинга, а также готова к быстрой разработке и тестированию.

## Структура каталогов

- `conf/` — конфигурационные файлы.
- `strategies/` — rule-based стратегии.
- `nn/` — модули нейросети.
- `tools/` — вспомогательные скрипты.

## Настройка окружения

В корне проекта находится файл `.env.example` с перечнем переменных окружения.
Скопируйте его в `.env` и заполните требуемые значения:
- `docs/` — документация по архитектуре и модулям.

Дополнительные сведения о подключении риск‑менеджера см. в [docs/risk_manager.md](docs/risk_manager.md).

## Требования

- Docker и `docker-compose` для запуска инфраструктуры.
- Python 3.10+ для обучения и запуска сервиса инференса.

## Переменные окружения

Скопируйте `.env.example` в `.env` и заполните API‑ключи бирж:
 main

```bash
cp .env.example .env
```

### Переменные

- `BINANCE_KEY` и `BINANCE_SECRET` — ключи API для работы с Binance в реальном режиме.
- `BINANCE_TEST_KEY` и `BINANCE_TEST_SECRET` — ключи API Binance для тестовой сети.
- `BYBIT_KEY` и `BYBIT_SECRET` — ключи API Bybit для реальной торговли.
- `BYBIT_TEST_KEY` и `BYBIT_TEST_SECRET` — ключи API Bybit для тестового режима.
- `OKX_KEY` и `OKX_SECRET` — ключи API OKX для реальной торговли.
- `OKX_TEST_KEY` и `OKX_TEST_SECRET` — ключи API OKX для тестовой среды.

Указывайте только те ключи, которые соответствуют используемым биржам и режимам работы.
=======
Переменные вида `BINANCE_KEY`, `BYBIT_SECRET` и другие используются скриптами и
в Docker‑контейнерах.

## Запуск бота через `docker-compose`

Поднимите инфраструктуру:

```bash
docker-compose up -d
```

Далее запустите бота с нужными параметрами, например для Binance в тестовом
режиме:

```bash
./tools/start.sh binance test
```

Скрипт объединит базовую конфигурацию с файлом `conf/binance.test.json` и
запустит контейнер `freqtrade`.

## Обучение нейросети

```bash
cd nn
pip install -r requirements.txt
python train.py --features path/to/features.parquet --epochs 20
```

Модель сохраняется в каталог `nn/` и регистрируется в MLflow.

## Запуск сервиса инференса

После обучения модели можно запустить BentoML‑сервис:

```bash
docker-compose up -d nn_infer
```

Он будет доступен на порту `3000` и использует модель с алиасом `prod` из
реестра BentoML.
=======
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
