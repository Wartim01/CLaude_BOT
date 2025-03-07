# main.py
"""
Main entry point for the crypto trading bot
"""
import os
import sys
import argparse
import signal
import time
from datetime import datetime

from trading_bot import TradingBot
from config.config import CONFIG_PATH, load_config
from utils.logger import setup_logger

logger = setup_logger("main")

def signal_handler(sig, frame):
    """Handle program termination signals"""
    logger.info("Shutdown signal received. Stopping bot...")
    if 'bot' in globals() and bot is not None:
        bot.stop()
    sys.exit(0)

def start_bot(config_path=CONFIG_PATH, headless=False):
    """
    Start the trading bot
    
    Args:
        config_path: Path to the configuration file
        headless: Whether to run in headless mode
    """
    logger.info(f"Starting crypto trading bot with config: {config_path}")
    
    # Create and start the bot
    global bot
    bot = TradingBot(config_path=config_path)
    
    try:
        # Start the bot
        bot.start()
        
        if headless:
            logger.info("Bot started in headless mode. Press Ctrl+C to stop.")
            
            # Keep the main thread alive
            while True:
                time.sleep(1)
        else:
            # Import and start the CLI for interactive mode
            from bot_cli import BotCommander
            commander = BotCommander(bot=bot)
            commander.cmdloop()
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down...")
    except Exception as e:
        logger.error(f"Error running bot: {str(e)}")
    finally:
        if bot.running:
            bot.stop()

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Cryptocurrency Trading Bot")
    parser.add_argument("--config", type=str, help="Path to configuration file", default=CONFIG_PATH)
    parser.add_argument("--headless", action="store_true", help="Run in headless mode without interactive CLI")
    
    args = parser.parse_args()
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start the bot
    start_bot(config_path=args.config, headless=args.headless)