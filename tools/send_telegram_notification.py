import argparse
import os
import sys
import logging

# Add project root to sys.path to allow importing tools.telegram_notifier
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from tools.telegram_notifier import TelegramNotifier
except ImportError:
    # Fallback if run in an environment where tools.telegram_notifier is not directly importable
    # This might happen if PYTHONPATH is not set up, and the script is run from a different directory.
    # For simplicity, we'll try to make it work, but proper PYTHONPATH setup is better.
    logging.error("Failed to import TelegramNotifier. Ensure PYTHONPATH is set correctly or run from project root.")
    # As a last resort, print to stdout if notifier cannot be imported.
    # This part is tricky as we don't want to duplicate TelegramNotifier logic here.
    # The primary expectation is that this script is called from start.sh where context is root.
    # If this script needs to be super robust on its own, it would need more complex path handling.
    print(f"TELEGRAM_FALLBACK_NOTIFICATION: {sys.argv[1] if len(sys.argv) > 1 else 'No message provided'}")
    sys.exit(1)


# Configure basic logging for the script itself
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send a message via Telegram.")
    parser.add_argument("message", type=str, help="The message to send.")
    args = parser.parse_args()

    notifier = None
    try:
        # Attempt to load .env file if python-dotenv is available
        # This ensures that if the script is run directly in an environment
        # that uses .env files, it can pick up the necessary variables.
        from dotenv import load_dotenv
        dotenv_path = os.path.join(project_root, '.env')
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path)
            logger.info(f"Loaded .env file from {dotenv_path}")
        else:
            logger.info(f".env file not found at {dotenv_path}, relying on environment variables.")

        notifier = TelegramNotifier()
        logger.info("TelegramNotifier initialized successfully for send_telegram_notification.py.")
    except ImportError:
        logger.warning("python-dotenv library not found. Cannot load .env file. Relying on environment variables for TelegramNotifier.")
        # Attempt to initialize notifier anyway, it will fail if env vars not set
        try:
            notifier = TelegramNotifier()
        except ValueError as ve:
            logger.error(f"Failed to initialize TelegramNotifier (dotenv not found, and env vars likely missing): {ve}")
            print(f"TELEGRAM_FALLBACK_INIT_ERROR: {ve} MESSAGE: {args.message}") # Fallback
            sys.exit(1) # Critical if notifier cannot be initialized
    except ValueError as e:
        logger.error(f"Failed to initialize TelegramNotifier: {e}. Message will not be sent.")
        print(f"TELEGRAM_FALLBACK_INIT_ERROR: {e} MESSAGE: {args.message}") # Fallback
        sys.exit(1) # Critical if notifier cannot be initialized
    except Exception as e:
        logger.error(f"An unexpected error occurred during TelegramNotifier initialization: {e}")
        print(f"TELEGRAM_FALLBACK_UNEXPECTED_ERROR: {e} MESSAGE: {args.message}") # Fallback
        sys.exit(1)


    if notifier:
        if not notifier.send(args.message):
            logger.error("Failed to send Telegram message.")
            # The notifier.send() method already prints a fallback message.
            sys.exit(1) # Indicate failure
        else:
            logger.info(f"Successfully sent Telegram message: \"{args.message}\"")
    else:
        # This case should ideally not be reached if initialization errors sys.exit(1)
        logger.error("TelegramNotifier was not initialized. Cannot send message.")
        print(f"TELEGRAM_FALLBACK_NOTIFIER_NONE: MESSAGE: {args.message}")
        sys.exit(1)
