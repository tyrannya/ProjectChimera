# Риск-менеджер

`CommonRiskManager` из `strategies/common/risk_manager.py` следит за просадкой,
контролирует частые ошибки 429 от биржи и расходы на финансирование. При
превышении порогов он поднимает `TemporaryStopException`, что временно ставит
торговлю на паузу.

## Подключение через конфигурацию

В конфигурации Freqtrade пропишите путь к классу риск-менеджера:

```json
{
  "risk": {
    "max_drawdown": 0.08
  },
  "custom_risk_management": "strategies.common.risk_manager.CommonRiskManager"
}
```

## Использование в стратегии

В стратегии можно создать экземпляр и вызывать проверки при необходимости:

```python
from strategies.common.risk_manager import CommonRiskManager

class ExampleStrategy(IStrategy):
    risk = CommonRiskManager()

    def populate_entry_trend(self, dataframe, metadata):
        # Пример вызова одной из проверок
        self.risk.funding_guard()
        return dataframe
```
