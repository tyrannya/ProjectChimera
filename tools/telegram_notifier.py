import requests
import os
import logging

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class TelegramNotifier:
    def __init__(self, bot_token=None, chat_id=None):
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")

        if not self.bot_token:
            logger.error("Telegram Bot Token not found. Please set TELEGRAM_BOT_TOKEN environment variable.")
            raise ValueError("Telegram Bot Token not found.")
        if not self.chat_id:
            logger.error("Telegram Chat ID not found. Please set TELEGRAM_CHAT_ID environment variable.")
            raise ValueError("Telegram Chat ID not found.")

    def send(self, message: str):
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML"  # Using HTML for rich text formatting
        }
        try:
            response = requests.post(url, data=payload, timeout=10)
            response.raise_for_status()  # Raise an exception for HTTP errors
            logger.info("Telegram message sent successfully.")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Telegram notification error: {e}")
            # Fallback to console print if Telegram fails
            print(f"FALLBACK: Telegram notification error: {e}")
            print(f"FALLBACK_MESSAGE: {message}")
            return False

    def send_error(self, message: str):
        # Prepending emoji and using <pre> for monospaced font, good for tracebacks
        error_message = f"⚠️ <b>Ошибка:</b>
<pre>{message}</pre>"
        return self.send(error_message)

    def send_trade(self, trade_info: dict):
        """
        Sends a structured trade notification.
        Example trade_info:
        {
            'symbol': 'BTC/USDT',
            'side': 'BUY',
            'amount': '0.01',
            'entry': '43000',
            'exit': '43250',
            'pnl': '+25',
            'timestamp': '2025-06-06 15:15'
        }
        """
        if not all(key in trade_info for key in ['symbol', 'side', 'amount', 'entry', 'pnl', 'timestamp']):
            logger.error(f"Trade info is missing some keys. Received: {trade_info}")
            self.send_error(f"Получены неполные данные о сделке: {trade_info}")
            return False

        # Determine if it's a profit or loss for appropriate emoji and title
        try:
            pnl_value = float(str(trade_info['pnl']).replace(' USDT', '')) # Make sure PNL is a number
            if pnl_value >= 0:
                title = "💰 Сделка закрыта с прибылью"
                pnl_str = f"+{pnl_value}"
            else:
                title = "🔻 Лосс по сделке"
                pnl_str = str(pnl_value)
        except ValueError:
            logger.warning(f"Could not parse PNL value: {trade_info.get('pnl')}. Defaulting to neutral message.")
            title = "ℹ️ Информация о сделке" # Neutral title if PNL parsing fails
            pnl_str = str(trade_info.get('pnl','N/A'))


        msg_lines = [
            f"<b>{title}</b>",
            f"💹 <b>[TRADE]</b>",
            f"<b>Пара:</b> {trade_info.get('symbol', 'N/A')}",
            f"<b>Операция:</b> {trade_info.get('side', 'N/A')}",
            f"<b>Объём:</b> {trade_info.get('amount', 'N/A')}",
            f"<b>Цена входа:</b> {trade_info.get('entry', 'N/A')}",
        ]

        # Optional 'exit' price
        if 'exit' in trade_info and trade_info['exit'] is not None:
            msg_lines.append(f"<b>Цена выхода:</b> {trade_info['exit']}")

        msg_lines.append(f"<b>PNL:</b> {pnl_str} USDT")
        msg_lines.append(f"<b>Время:</b> {trade_info.get('timestamp', 'N/A')}")

        return self.send("
".join(msg_lines))

if __name__ == '__main__':
    # This is for basic testing directly running this file.
    # Ensure TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set as environment variables.
    print("Attempting to send a test message...")

    # Load .env file for local testing if available
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("Loaded .env file for local testing.")
    except ImportError:
        print("dotenv library not found, relying on environment variables being set manually.")

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("TELEGRAM_BOT_TOKEN and/or TELEGRAM_CHAT_ID environment variables are not set.")
        print("Please set them to run the test.")
    else:
        notifier = TelegramNotifier(bot_token=token, chat_id=chat_id)

        print(f"Notifier initialized with token: {token[:10]}... and chat_id: {chat_id}")

        test_send = notifier.send("Hello from TelegramNotifier! 👋
This is a <b>bold</b> and <i>italic</i> test message.")
        print(f"Test send successful: {test_send}")

        test_error_send = notifier.send_error("This is a test error message with a traceback line:
ValueError: Something went wrong.")
        print(f"Test error send successful: {test_error_send}")

        test_trade_profit = {
            'symbol': 'ETH/USDT',
            'side': 'SELL',
            'amount': '1.5',
            'entry': '3000',
            'exit': '2950',
            'pnl': '-75', # Loss example
            'timestamp': '2023-10-27 10:00'
        }
        test_trade_send_profit = notifier.send_trade(test_trade_profit)
        print(f"Test trade (loss) send successful: {test_trade_send_profit}")

        test_trade_loss = {
            'symbol': 'ADA/USDT',
            'side': 'BUY',
            'amount': '1000',
            'entry': '0.30',
            'exit': '0.33',
            'pnl': '+30', # Profit example
            'timestamp': '2023-10-27 11:00'
        }
        test_trade_send_loss = notifier.send_trade(test_trade_loss)
        print(f"Test trade (profit) send successful: {test_trade_send_loss}")

        test_incomplete_trade = {
            'symbol': 'SOL/USDT',
            'side': 'BUY',
            # amount is missing
            'entry': '40',
            'pnl': '+5',
            'timestamp': '2023-10-27 12:00'
        }
        test_trade_incomplete_send = notifier.send_trade(test_incomplete_trade)
        print(f"Test incomplete trade send successful (should send an error): {test_trade_incomplete_send}")

        print("Test messages attempt finished. Check your Telegram chat.")

# Ensure 'requests' and 'python-dotenv' (for local testing of this script) are in requirements.txt
# Add to requirements.txt:
# requests
# python-dotenv
