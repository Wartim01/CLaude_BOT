"""
Command-line interface for interacting with the trading bot
"""
import os
import sys
import cmd
import shlex
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
from tabulate import tabulate

from trading_bot import TradingBot
from utils.logger import setup_logger
from config.config import CONFIG_PATH
import argparse
import signal

logger = setup_logger("bot_cli")

class BotCommander(cmd.Cmd):
    """
    Interactive command-line interface for the trading bot
    """
    intro = "\nCryptoCurrency Trading Bot CLI. Type 'help' to see available commands.\n"
    prompt = "crypto-bot> "
    
    def __init__(self, bot: TradingBot = None, config_path: str = CONFIG_PATH):
        """
        Initialize the CLI
        
        Args:
            bot: TradingBot instance
            config_path: Path to config file
        """
        super().__init__()
        self.bot = bot
        self.config_path = config_path
        
        # If bot not provided, create one
        if self.bot is None:
            self.bot = TradingBot(config_path=config_path)
    
    def do_start(self, arg):
        """Start the trading bot"""
        if self.bot.running:
            print("Bot is already running!")
            return
            
        print("Starting bot...")
        result = self.bot.start()
        
        if result:
            print("Bot started successfully!")
        else:
            print("Failed to start bot. Check logs for details.")
    
    def do_stop(self, arg):
        """Stop the trading bot"""
        if not self.bot.running:
            print("Bot is not running!")
            return
            
        print("Stopping bot...")
        result = self.bot.stop()
        
        if result:
            print("Bot stopped successfully!")
        else:
            print("Failed to stop bot. Check logs for details.")
    
    def do_status(self, arg):
        """Show the current status of the bot"""
        status = self.bot.get_status()
        
        print("\n=== Bot Status ===")
        print(f"Running: {status['running']}")
        print(f"Paused: {status.get('paused', False)}")
        print(f"Mode: {status.get('mode', 'unknown')}")
        
        if status['running'] and status.get('start_time'):
            print(f"Uptime: {status['uptime']}")
            print(f"Active pairs: {', '.join(status['active_pairs'])}")
            print(f"Active positions: {status['active_positions']}")
            print(f"Trades executed: {status['trades_executed']}")
        
        # Show performance if available
        if 'performance' in status:
            perf = status['performance']
            print("\n=== Performance ===")
            
            if 'total_pnl' in perf:
                print(f"Total PnL: {perf['total_pnl']:.2f}")
                
            if 'win_rate' in perf:
                print(f"Win rate: {perf['win_rate']:.2f}%")
    
    def do_balance(self, arg):
        """Show the current account balance"""
        if not self.bot.running:
            print("Bot is not running! Start the bot first.")
            return
        
        balance = self.bot._update_account_balance()
        
        print("\n=== Account Balance ===")
        print(f"Total: {balance.get('total', 'N/A')}")
        print(f"Free: {balance.get('free', 'N/A')}")
        print(f"Used: {balance.get('used', 'N/A')}")
        
        # Show individual assets if available
        if 'balances' in balance:
            print("\n=== Assets ===")
            
            # Create a table for better formatting
            table_data = []
            for asset, data in balance['balances'].items():
                if isinstance(data, dict) and data.get('total', 0) > 0:
                    table_data.append([
                        asset,
                        data.get('free', 0),
                        data.get('locked', 0),
                        data.get('total', 0)
                    ])
            
            if table_data:
                print(tabulate(
                    table_data,
                    headers=["Asset", "Free", "Locked", "Total"],
                    floatfmt=".8f"
                ))
            else:
                print("No assets found.")
    
    def do_pairs(self, arg):
        """Show or modify the trading pairs"""
        if not self.bot.running:
            print("Bot is not running! Start the bot first.")
            return
        
        args = shlex.split(arg)
        
        if len(args) == 0:
            # Show current trading pairs
            pairs = self.bot.active_pairs
            
            print("\n=== Active Trading Pairs ===")
            for i, pair in enumerate(pairs):
                print(f"{i + 1}. {pair}")
        else:
            # Modify trading pairs
            action = args[0].lower()
            
            if action == "add" and len(args) > 1:
                new_pairs = args[1:]
                self.bot.add_pairs(new_pairs)
                print(f"Added pairs: {', '.join(new_pairs)}")
            elif action == "remove" and len(args) > 1:
                remove_pairs = args[1:]
                self.bot.remove_pairs(remove_pairs)
                print(f"Removed pairs: {', '.join(remove_pairs)}")
            else:
                print("Invalid command. Usage: pairs [add|remove] [pair1 pair2 ...]")
    
    def do_exit(self, arg):
        """Exit the CLI"""
        print("Exiting...")
        return True
    
    def do_quit(self, arg):
        """Exit the CLI"""
        return self.do_exit(arg)
    
    def do_EOF(self, arg):
        """Exit on EOF (Ctrl+D)"""
        print()
        return self.do_exit(arg)

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Command-line interface for crypto trading bot")
    parser.add_argument("--config", type=str, help="Path to configuration file", default=CONFIG_PATH)
    parser.add_argument("--headless", action="store_true", help="Run bot in headless mode without interactive CLI")
    
    args = parser.parse_args()
    
    if args.headless:
        # Run in headless mode
        bot = TradingBot(config_path=args.config)
        
        # Setup signal handler for graceful shutdown
        def signal_handler(sig, frame):
            print("\nShutdown signal received. Stopping bot...")
            bot.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            bot.start()
            print("Bot started in headless mode. Press Ctrl+C to stop.")
            
            # Keep main thread alive
            while True:
                time.sleep(1)
        except Exception as e:
            print(f"Error in headless mode: {e}")
            bot.stop()
    else:
        # Run interactive CLI
        commander = BotCommander(config_path=args.config)
        
        try:
            commander.cmdloop()
        except KeyboardInterrupt:
            print("\nReceived keyboard interrupt. Exiting...")
            if commander.bot and commander.bot.running:
                commander.bot.stop()

if __name__ == "__main__":
    main()