import logging
import os
from datetime import datetime, timezone

# Attempt to load .env for local testing/development if this file is run directly or imported early
try:
    from dotenv import load_dotenv
    # Assuming .env is in the project root, which is two levels up from strategies/common/
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
        # print(f"NotifyingStrategy: Loaded .env from {dotenv_path}") # For debugging
    # else:
        # print(f"NotifyingStrategy: .env not found at {dotenv_path}") # For debugging
except ImportError:
    # print("NotifyingStrategy: python-dotenv not found, relying on env vars.") # For debugging
    pass

from freqtrade.strategy import IStrategy
from freqtrade.persistence import Trade

# Correctly import TelegramNotifier from the tools directory
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from tools.telegram_notifier import TelegramNotifier
except ImportError as e:
    # This is a critical import. If it fails, notifications won't work.
    # Log an error or print, and set notifier to None.
    # Strategies should still function without notifications.
    print(f"CRITICAL: Failed to import TelegramNotifier in NotifyingStrategy: {e}. Notifications will be disabled.")
    TelegramNotifier = None


logger = logging.getLogger(__name__)

class NotifyingStrategy(IStrategy):
    _notifier = None

    def __init__(self, config: dict):
        super().__init__(config)
        if TelegramNotifier:
            try:
                if NotifyingStrategy._notifier is None: # Initialize only once
                    NotifyingStrategy._notifier = TelegramNotifier()
                self.notifier = NotifyingStrategy._notifier
                # print("NotifyingStrategy: TelegramNotifier initialized.") # For debugging
            except ValueError as e:
                logger.warning(f"Failed to initialize TelegramNotifier in NotifyingStrategy: {e}. Trade notifications will be disabled.")
                self.notifier = None
            except Exception as e:
                logger.error(f"Unexpected error initializing TelegramNotifier in NotifyingStrategy: {e}")
                self.notifier = None
        else:
            # print("NotifyingStrategy: TelegramNotifier class not available.") # For debugging
            self.notifier = None

    # Freqtrade calls sell_filled_get_config to get sell reason and other info after a sell is filled.
    # This seems like a reasonable place to hook into, as the trade object is populated with exit details.
    # However, a more direct callback after a trade is fully processed by the bot might be ideal.
    # Let's try to use a method that is reliably called after a trade is considered closed by freqtrade.
    # The `custom_sell` method provides the trade object, but it's for custom sell signals.
    # `confirm_trade_exit` is called before the order is placed.

    # A common way to react to closed trades is to check `Trade.is_open == False`.
    # Freqtrade's architecture doesn't have a simple "trade_closed_callback" directly in IStrategy.
    # One approach is to augment a method that is called frequently, like `populate_indicators`,
    # and check for recently closed trades. This is inefficient.

    # Another option: Freqtrade's REST API or RPC can provide closed trades.
    # For now, let's try to augment a method that is called when a sell happens.
    # `order_filled` is a method in `IStrategy` that gets called when an order (buy or sell) is filled.
    # This could be a good place.

    def order_filled(self, pair: str, trade: 'Trade', order: dict, current_time: datetime) -> None:
        """
        Called when an order is filled (either buy or sell).
        We are interested in sell fills to notify about closed trades.
        """
        super().order_filled(pair, trade, order, current_time) # Call parent method

        if self.notifier and order['status'] == 'closed' and order['side'] == 'sell' and trade.close_profit_abs is not None:
            try:
                timestamp_str = trade.close_date.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z') if trade.close_date else current_time.strftime('%Y-%m-%d %H:%M:%S')

                trade_info = {
                    'symbol': trade.pair,
                    'side': trade.trade_direction.upper(), # 'long' or 'short' -> 'LONG' or 'SHORT'
                    'amount': f"{trade.amount:.8f}", # Format amount
                    'entry': f"{trade.open_rate:.{trade.pair_decimals}f}", # Format entry price
                    'exit': f"{trade.close_rate:.{trade.pair_decimals}f}" if trade.close_rate else 'N/A', # Format exit price
                    'pnl': f"{trade.close_profit_abs:.{trade.price_precision}f}", # Absolute profit/loss
                    'timestamp': timestamp_str
                }
                # The 'side' in send_trade expects BUY/SELL for the operation that opened the trade.
                # Freqtrade's trade.trade_direction is 'long' or 'short'.
                # We can map 'long' to 'BUY' (entry) and 'short' to 'SELL' (entry).
                # The send_trade method itself will add "Сделка закрыта с прибылью/лосс" based on PNL sign.
                # Let's assume the 'side' for the notification should represent the entry action.
                if trade.trade_direction == 'long':
                    trade_info['side'] = 'BUY' # Entry was a buy
                elif trade.trade_direction == 'short':
                    trade_info['side'] = 'SELL' # Entry was a sell (for futures)
                else:
                    trade_info['side'] = str(trade.trade_direction).upper()


                logger.info(f"Sending trade notification for {trade.pair}, PNL: {trade.close_profit_abs}")
                self.notifier.send_trade(trade_info)
            except Exception as e:
                logger.error(f"Error formatting or sending trade notification for trade {trade.id}: {e}")
        return None # Explicitly return None as per IStrategy docs for some callbacks


    # Optional: If you want to notify on every sell signal generated, not just filled orders.
    # def custom_sell(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
    #                 current_profit: float, **kwargs):
    #     sell_reason = super().custom_sell(pair, trade, current_time, current_rate, current_profit, **kwargs)
    #     if sell_reason and self.notifier:
    #         # This is more complex as the trade isn't fully closed and PNL might not be final.
    #         # For now, relying on order_filled is safer for actual PNL.
    #         pass
    #     return sell_reason
