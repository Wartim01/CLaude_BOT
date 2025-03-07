"""
Main entry point for the cryptocurrency trading bot.
Coordinates all components including data collection, analysis,
decision making, and trade execution.
"""
import os
import json
import time
import logging
import traceback
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import threading

from config.config import load_config, CONFIG_PATH
from exchanges.exchange_client import ExchangeClient
from data.data_manager import DataManager
from strategies.strategy_manager import StrategyManager
from risk.risk_manager import RiskManager
from ai.models.model_workflow import ModelWorkflow
from ai.decision_engine import DecisionEngine
from utils.logger import setup_logger
from utils.notification_handler import NotificationHandler
from utils.performance_metrics import PerformanceTracker

class TradingBot:
    """
    Main trading bot class that orchestrates all components
    """
    def __init__(self, config_path: str = CONFIG_PATH):
        """
        Initialize the trading bot
        
        Args:
            config_path: Path to configuration file
        """
        self.start_time = datetime.now()
        self.logger = setup_logger("trading_bot")
        self.logger.info(f"Starting trading bot at {self.start_time}")
        
        # Load configuration
        self.config = load_config(config_path)
        self.bot_config = self.config.get("bot", {})
        self.trading_config = self.config.get("trading", {})
        
        # Initialize components
        self._initialize_components()
        
        # Runtime state
        self.running = False
        self.paused = False
        self.last_check_time = None
        self.last_trade_time = {}
        self.trading_pairs = self._load_trading_pairs()
        self.trading_timeframes = self._load_timeframes()
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        
        # Thread control
        self.main_thread = None
        self.monitoring_thread = None
        self.should_stop = threading.Event()
    
    def _initialize_components(self) -> None:
        """Initialize all bot components"""
        try:
            # Initialize exchange client
            self.logger.info("Initializing exchange client")
            self.exchange = ExchangeClient(
                exchange_id=self.config.get("exchange", {}).get("name", "binance"),
                api_key=self.config.get("exchange", {}).get("api_key"),
                api_secret=self.config.get("exchange", {}).get("api_secret")
            )
            
            # Initialize data manager
            self.logger.info("Initializing data manager")
            self.data_manager = DataManager(exchange_client=self.exchange)
            
            # Initialize risk manager
            self.logger.info("Initializing risk manager")
            self.risk_manager = RiskManager(
                default_risk_per_trade=self.trading_config.get("risk_per_trade", 0.01),
                account_risk_limit=self.trading_config.get("max_risk", 0.15)
            )
            
            # Initialize strategy manager
            self.logger.info("Initializing strategy manager")
            self.strategy_manager = StrategyManager()
            
            # Initialize AI components if enabled
            if self.bot_config.get("use_ai", False):
                self.logger.info("Initializing AI components")
                self.model_workflow = ModelWorkflow(
                    use_enhanced_model=self.bot_config.get("use_enhanced_model", True)
                )
                
                # Load the model
                if self.bot_config.get("load_model_on_start", True):
                    self.logger.info("Loading AI model")
                    self.model_workflow.load_model()
            else:
                self.model_workflow = None
            
            # Initialize decision engine
            self.logger.info("Initializing decision engine")
            self.decision_engine = DecisionEngine()
            
            # Initialize notification handler
            self.logger.info("Initializing notification handler")
            self.notification = NotificationHandler(
                telegram_token=self.config.get("notifications", {}).get("telegram_token"),
                telegram_chat_id=self.config.get("notifications", {}).get("telegram_chat_id")
            )
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    def _load_trading_pairs(self) -> List[str]:
        """Load the trading pairs from configuration"""
        pairs = self.trading_config.get("pairs", [])
        
        if not pairs and self.trading_config.get("auto_select_pairs", False):
            # Implement automatic pair selection based on volume, etc.
            self.logger.info("Auto-selecting trading pairs")
            pairs = self._auto_select_pairs()
        
        self.logger.info(f"Trading pairs: {pairs}")
        return pairs
    
    def _auto_select_pairs(self) -> List[str]:
        """Automatically select trading pairs based on volume, etc."""
        try:
            # Get market data
            markets = self.exchange.get_markets()
            
            # Filter for USDT pairs
            usdt_markets = [m for m in markets if m['symbol'].endswith('USDT')]
            
            # Sort by volume
            usdt_markets.sort(key=lambda x: x.get('quoteVolume', 0), reverse=True)
            
            # Take top N
            top_n = self.trading_config.get("auto_select_count", 5)
            selected_pairs = [m['symbol'] for m in usdt_markets[:top_n]]
            
            self.logger.info(f"Auto-selected pairs: {selected_pairs}")
            return selected_pairs
        except Exception as e:
            self.logger.error(f"Error auto-selecting pairs: {str(e)}")
            # Return default pairs as fallback
            return ["BTCUSDT", "ETHUSDT"]
    
    def _load_timeframes(self) -> List[str]:
        """Load the trading timeframes from configuration"""
        default_timeframes = ["1h", "4h"]
        timeframes = self.trading_config.get("timeframes", default_timeframes)
        
        self.logger.info(f"Trading timeframes: {timeframes}")
        return timeframes
    
    def start(self) -> None:
        """Start the trading bot"""
        if self.running:
            self.logger.warning("Bot is already running")
            return
        
        self.logger.info("Starting trading bot")
        self.running = True
        self.paused = False
        
        # Initialize account data
        self._update_account_info()
        
        # Send startup notification
        self._send_startup_notification()
        
        # Start main loop in a separate thread
        self.should_stop.clear()
        self.main_thread = threading.Thread(target=self._main_loop)
        self.main_thread.daemon = True
        self.main_thread.start()
        
        # Start monitoring thread if enabled
        if self.bot_config.get("enable_monitoring", True):
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
        
        self.logger.info("Trading bot started")
    
    def stop(self) -> None:
        """Stop the trading bot"""
        if not self.running:
            return
        
        self.logger.info("Stopping trading bot")
        self.should_stop.set()
        
        # Wait for threads to finish
        if self.main_thread:
            self.main_thread.join(timeout=10)
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        
        self.running = False
        
        # Send shutdown notification
        self.notification.send_message(
            "📴 Trading bot shutdown",
            f"Bot has been shut down at {datetime.now()}"
        )
        
        self.logger.info("Trading bot stopped")
    
    def pause(self) -> None:
        """Pause the trading bot (stop opening new trades)"""
        self.paused = True
        self.logger.info("Trading bot paused")
        
        self.notification.send_message(
            "⏸️ Trading bot paused",
            "Bot has been paused. No new trades will be opened, but existing positions will be managed."
        )
    
    def resume(self) -> None:
        """Resume the trading bot"""
        self.paused = False
        self.logger.info("Trading bot resumed")
        
        self.notification.send_message(
            "▶️ Trading bot resumed",
            "Bot has been resumed and will continue normal operation."
        )
    
    def _main_loop(self) -> None:
        """Main trading loop"""
        self.logger.info("Main trading loop started")
        
        check_interval = self.bot_config.get("check_interval_seconds", 60)
        
        while not self.should_stop.is_set():
            try:
                current_time = datetime.now()
                
                # Update last check time
                self.last_check_time = current_time
                
                # Check market conditions and make trading decisions
                if not self.paused:
                    self._check_and_trade()
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Sleep until next check
                time_to_sleep = max(0, check_interval - (datetime.now() - current_time).total_seconds())
                if time_to_sleep > 0:
                    # Use wait instead of sleep to allow for early termination
                    self.should_stop.wait(time_to_sleep)
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {str(e)}")
                self.logger.error(traceback.format_exc())
                
                # Notify about the error
                self.notification.send_message(
                    "❌ Trading bot error",
                    f"Error in main loop: {str(e)}\nBot will continue running."
                )
                
                # Sleep for a while before continuing
                time.sleep(30)
    
    def _monitoring_loop(self) -> None:
        """Separate thread for monitoring system health and performance"""
        self.logger.info("Monitoring loop started")
        
        monitoring_interval = self.bot_config.get("monitoring_interval_seconds", 300)  # 5 minutes
        
        while not self.should_stop.is_set():
            try:
                # Check system health
                self._check_system_health()
                
                # Check open positions and manage them
                self._manage_open_positions()
                
                # Wait until next check
                self.should_stop.wait(monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(60)  # Sleep for a minute before continuing
    
    def _check_and_trade(self) -> None:
        """
        Check market conditions for all pairs and timeframes
        and execute trades if conditions are met
        """
        for pair in self.trading_pairs:
            for timeframe in self.trading_timeframes:
                try:
                    # Check if we should skip this pair/timeframe based on timing rules
                    if not self._should_check_now(pair, timeframe):
                        continue
                    
                    self.logger.info(f"Checking {pair} on {timeframe} timeframe")
                    
                    # Get market data
                    data = self.data_manager.get_market_data(pair, timeframe)
                    
                    if data is None or data.empty:
                        self.logger.warning(f"No data available for {pair} on {timeframe} timeframe")
                        continue
                    
                    # Check for trade opportunities
                    self._process_trading_opportunity(pair, timeframe, data)
                    
                    # Update last check time for this pair and timeframe
                    if pair not in self.last_trade_time:
                        self.last_trade_time[pair] = {}
                    
                    self.last_trade_time[pair][timeframe] = datetime.now()
                    
                except Exception as e:
                    self.logger.error(f"Error checking {pair} on {timeframe}: {str(e)}")
                    self.logger.error(traceback.format_exc())
        
    def _should_check_now(self, pair: str, timeframe: str) -> bool:
        """
        Determine if this pair/timeframe should be checked now
        based on time since last check and timeframe duration
        """
        # If this pair/timeframe has not been checked yet, do it now
        if pair not in self.last_trade_time or timeframe not in self.last_trade_time[pair]:
            return True
        
        # Get time since last check
        time_since_last_check = datetime.now() - self.last_trade_time[pair][timeframe]
        
        # Get timeframe duration in seconds
        tf_seconds = self._timeframe_to_seconds(timeframe)
        
        # Define minimum check intervals based on timeframe
        # For shorter timeframes, check more frequently as a percentage of the timeframe
        if tf_seconds <= 3600:  # 1h or less
            min_interval = tf_seconds * 0.25  # 25% of timeframe
        elif tf_seconds <= 14400:  # 4h or less
            min_interval = tf_seconds * 0.15  # 15% of timeframe
        else:  # Daily or longer
            min_interval = tf_seconds * 0.1  # 10% of timeframe
        
        # Ensure a reasonable minimum (e.g., 30 seconds)
        min_interval = max(30, min_interval)
        
        # Check if enough time has passed
        return time_since_last_check.total_seconds() >= min_interval
    
    def _timeframe_to_seconds(self, timeframe: str) -> int:
        """Convert timeframe string to seconds"""
        value = int(timeframe[:-1])
        unit = timeframe[-1].lower()
        
        if unit == 'm':
            return value * 60
        elif unit == 'h':
            return value * 3600
        elif unit == 'd':
            return value * 86400
        elif unit == 'w':
            return value * 604800
        else:
            return 3600  # Default to 1h if unknown
    
    def _process_trading_opportunity(self, pair: str, timeframe: str, data: pd.DataFrame) -> None:
        """
        Process a potential trading opportunity for a specific pair and timeframe
        
        Args:
            pair: Trading pair (e.g., 'BTCUSDT')
            timeframe: Timeframe (e.g., '1h')
            data: OHLCV data
        """
        self.logger.debug(f"Processing potential trading opportunity for {pair} on {timeframe}")
        
        # Get account info for position sizing
        account_info = self._get_account_info()
        balance = account_info.get("balance", 0)
        
        # 1. Get AI model predictions if enabled
        prediction = None
        if self.model_workflow:
            prediction = self._get_model_prediction(pair, data)
        
        # 2. Get strategy signals
        strategy_signals = self._get_strategy_signals(pair, timeframe, data)
        
        # 3. Get market risk assessment
        market_data = {pair: data}
        market_risk = self.risk_manager.market_risk_analyzer.calculate_market_risk(market_data)
        
        # 4. Use decision engine to evaluate the opportunity
        decision = self.decision_engine.evaluate_trading_opportunity(
            symbol=pair,
            data=data,
            timeframe=timeframe,
            execute=False  # Don't auto-execute, we'll handle it here
        )
        
        # 5. Determine if we should trade
        if decision["should_trade"]:
            self.logger.info(f"Trading opportunity detected for {pair} on {timeframe}: {decision['signal_type']}")
            
            # Get entry price (current price)
            entry_price = data["close"].iloc[-1]
            
            # Calculate stop loss based on strategy or risk parameters
            if "stop_loss" in decision:
                stop_loss = decision["stop_loss"]
            else:
                # Default stop loss based on ATR or fixed percentage
                atr = data["atr"].iloc[-1] if "atr" in data.columns else entry_price * 0.03
                if decision["direction"] == "BUY":
                    stop_loss = entry_price - atr * 2
                else:  # SELL
                    stop_loss = entry_price + atr * 2
            
            # Evaluate trade risk
            risk_evaluation = self.risk_manager.evaluate_trade_risk(
                symbol=pair,
                direction=decision["direction"],
                entry_price=entry_price,
                stop_loss=stop_loss,
                account_balance=balance,
                market_data=market_data
            )
            
            # If risk is acceptable, execute the trade
            if risk_evaluation.get("approve_trade", False):
                self._execute_trade(
                    pair=pair,
                    direction=decision["direction"],
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    position_size=risk_evaluation["position_sizing"]["position_size"],
                    risk_percentage=risk_evaluation["position_sizing"]["risk_percentage"] / 100,  # Convert to decimal
                    timeframe=timeframe
                )
            else:
                risk_reason = "Unknown risk issue"
                if "position_sizing" in risk_evaluation and "error" in risk_evaluation["position_sizing"]:
                    risk_reason = risk_evaluation["position_sizing"]["error"]
                elif risk_evaluation.get("market_risk", {}).get("extreme_risk", False):
                    risk_reason = "Extreme market risk"
                elif risk_evaluation.get("correlation_risk", {}).get("high_correlation", False):
                    risk_reason = "High correlation risk"
                    
                self.logger.info(f"Trade opportunity for {pair} rejected due to risk assessment: {risk_reason}")
        else:
            self.logger.debug(f"No trading opportunity for {pair} on {timeframe}")
    
    def _get_model_prediction(self, pair: str, data: pd.DataFrame) -> Dict:
        """Get prediction from AI model"""
        if not self.model_workflow:
            return {"available": False, "reason": "AI model not enabled"}
        
        try:
            prediction = self.model_workflow.predict(data)
            
            if prediction.get("success", False):
                return {
                    "available": True,
                    "predictions": prediction["predictions"],
                    "raw": prediction
                }
            else:
                return {
                    "available": False,
                    "reason": prediction.get("error", "Unknown prediction error")
                }
        except Exception as e:
            self.logger.error(f"Error getting model prediction for {pair}: {str(e)}")
            return {"available": False, "reason": f"Error: {str(e)}"}
    
    def _get_strategy_signals(self, pair: str, timeframe: str, data: pd.DataFrame) -> Dict:
        """Get signals from strategy manager"""
        try:
            signals = self.strategy_manager.run_strategies(data, pair, timeframe)
            return signals
        except Exception as e:
            self.logger.error(f"Error getting strategy signals for {pair} on {timeframe}: {str(e)}")
            return {"error": str(e)}
    
    def _execute_trade(self, pair: str, direction: str, entry_price: float,
                     stop_loss: float, position_size: float, risk_percentage: float,
                     timeframe: str) -> Dict:
        """
        Execute a trade
        
        Args:
            pair: Trading pair (e.g., 'BTCUSDT')
            direction: Trade direction ('BUY' or 'SELL')
            entry_price: Entry price
            stop_loss: Stop loss price
            position_size: Position size in base currency
            risk_percentage: Risk as percentage of account (decimal)
            timeframe: Timeframe that generated the signal
            
        Returns:
            Trade execution result
        """
        self.logger.info(f"Executing {direction} trade for {pair} with position size {position_size}")
        
        try:
            # Execute the trade on the exchange
            order_result = self.exchange.create_order(
                symbol=pair,
                order_type="market",
                side=direction.lower(),
                amount=position_size
            )
            
            if order_result.get("id"):
                # Trade executed successfully
                trade_id = order_result["id"]
                
                # Register the trade with risk manager
                self.risk_manager.register_new_trade(
                    trade_id=trade_id,
                    symbol=pair,
                    direction=direction,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    position_size=position_size,
                    risk_percentage=risk_percentage,
                    account_balance=self._get_account_info()["balance"]
                )
                
                # Set a stop loss order
                self._set_stop_loss(pair, direction, stop_loss, position_size)
                
                # Send notification
                self._send_trade_notification(
                    trade_id=trade_id,
                    pair=pair,
                    direction=direction,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    position_size=position_size,
                    timeframe=timeframe
                )
                
                return {
                    "success": True,
                    "trade_id": trade_id,
                    "order": order_result
                }
            else:
                self.logger.error(f"Failed to execute trade for {pair}: {order_result}")
                return {
                    "success": False,
                    "error": "Failed to execute trade",
                    "order_result": order_result
                }
        
        except Exception as e:
            self.logger.error(f"Error executing trade for {pair}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _set_stop_loss(self, pair: str, direction: str, stop_price: float, amount: float) -> Dict:
        """Set a stop loss order"""
        try:
            # Determine stop loss order side (opposite of entry direction)
            stop_side = "sell" if direction.lower() == "buy" else "buy"
            
            stop_order = self.exchange.create_order(
                symbol=pair,
                order_type="stop",
                side=stop_side,
                amount=amount,
                price=stop_price
            )
            
            self.logger.info(f"Stop loss set for {pair} at {stop_price}")
            return stop_order
        except Exception as e:
            self.logger.error(f"Error setting stop loss for {pair}: {str(e)}")
            return {"error": str(e)}
    
    def _manage_open_positions(self) -> None:
        """Check and manage open positions"""
        try:
            # Get open positions
            positions = self.exchange.get_open_positions()
            
            # Update account info with current positions
            self._update_account_info()
            
            # Get current account balance
            account_info = self._get_account_info()
            
            # For each position, update the risk manager
            for position in positions:
                trade_id = position.get("id")
                symbol = position.get("symbol")
                current_price = position.get("market_price")
                
                if trade_id and symbol and current_price:
                    self.risk_manager.update_trade_status(
                        trade_id=trade_id,
                        current_price=current_price,
                        account_balance=account_info.get("balance", 0)
                    )
        except Exception as e:
            self.logger.error(f"Error managing open positions: {str(e)}")
    
    def _check_system_health(self) -> Dict:
        """Check the health of the system"""
        health_metrics = {
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "last_check_time": self.last_check_time.isoformat() if self.last_check_time else None,
            "connection_status": self.exchange.check_connection(),
            "active_positions": len(self.exchange.get_open_positions()),
            "account_balance": self._get_account_info().get("balance", 0),
            "cpu_usage": None,  # Could be implemented with psutil
            "memory_usage": None,  # Could be implemented with psutil
            "api_rate_limit": None,  # Could be provided by exchange client
            "timestamp": datetime.now().isoformat()
        }
        
        # Log issues if any
        if not health_metrics["connection_status"]:
            self.logger.warning("Connection to exchange lost")
            
            # Attempt to reconnect
            self._attempt_reconnect()
        
        return health_metrics
    
    def _attempt_reconnect(self) -> bool:
        """Attempt to reconnect to the exchange"""
        try:
            self.logger.info("Attempting to reconnect to exchange")
            connection = self.exchange.reconnect()
            
            if connection:
                self.logger.info("Successfully reconnected to exchange")
                return True
            else:
                self.logger.error("Failed to reconnect to exchange")
                return False
        except Exception as e:
            self.logger.error(f"Error reconnecting to exchange: {str(e)}")
            return False
    
    def _update_performance_metrics(self) -> None:
        """Update performance metrics"""
        try:
            # Get performance data
            account_info = self._get_account_info()
            current_balance = account_info.get("balance", 0)
            
            # Update performance metrics
            self.performance_tracker.update(
                current_balance=current_balance,
                open_positions=self.exchange.get_open_positions(),
                closed_trades=self.risk_manager.historical_positions
            )
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {str(e)}")
    
    def _update_account_info(self) -> None:
        """Update account information"""
        try:
            account_info = self.exchange.get_account_info()
            
            # Store account info for later use
            self._account_info = {
                "balance": account_info.get("total", {}).get("USDT", 0),
                "free_balance": account_info.get("free", {}).get("USDT", 0),
                "used_balance": account_info.get("used", {}).get("USDT", 0),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error updating account info: {str(e)}")
    
    def _get_account_info(self) -> Dict:
        """Get the latest account information"""
        if not hasattr(self, '_account_info'):
            self._update_account_info()
        
        return self._account_info
    
    def _send_startup_notification(self) -> None:
        """Send a notification about bot startup"""
        # Calculate initial account value
        account_info = self._get_account_info()
        
        message = (
            f"🤖 Trading Bot Started\n\n"
            f"Time: {datetime.now()}\n"
            f"Balance: {account_info.get('balance', 0):.2f} USDT\n"
            f"Trading pairs: {', '.join(self.trading_pairs)}\n"
            f"Timeframes: {', '.join(self.trading_timeframes)}\n"
            f"AI enabled: {'Yes' if self.model_workflow else 'No'}"
        )
        
        self.notification.send_message("🚀 Bot Startup", message)
    
    def _send_trade_notification(self, trade_id: str, pair: str, direction: str,
                              entry_price: float, stop_loss: float, position_size: float,
                              timeframe: str) -> None:
        """Send a notification about a new trade"""
        risk_amount = abs(entry_price - stop_loss) * position_size
        account_info = self._get_account_info()
        risk_percentage = (risk_amount / account_info.get("balance", 1)) * 100
        
        message = (
            f"🔄 New Trade Executed\n\n"
            f"Pair: {pair}\n"
            f"Direction: {'🟢 BUY' if direction == 'BUY' else '🔴 SELL'}\n"
            f"Timeframe: {timeframe}\n"
            f"Entry price: {entry_price}\n"
            f"Stop loss: {stop_loss}\n"
            f"Position size: {position_size:.4f}\n"
            f"Risk: {risk_percentage:.2f}%\n"
            f"Trade ID: {trade_id}"
        )
        
        self.notification.send_message(
            "🔄 New Trade",
            message
        )
    
    def get_status(self) -> Dict:
        """Get current bot status"""
        account_info = self._get_account_info()
        
        return {
            "running": self.running,
            "paused": self.paused,
            "uptime": str(datetime.now() - self.start_time),
            "start_time": self.start_time.isoformat(),
            "last_check_time": self.last_check_time.isoformat() if self.last_check_time else None,
            "account_balance": account_info.get("balance", 0),
            "free_balance": account_info.get("free_balance", 0),
            "active_pairs": self.trading_pairs,
            "active_positions": len(self.exchange.get_open_positions()),
            "trades_executed": len(self.risk_manager.historical_positions),
            "performance": self.performance_tracker.get_summary(),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    # Create and start the trading bot
    bot = TradingBot()
    
    try:
        bot.start()
        
        # Keep main thread alive
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Shutdown signal received")
        bot.stop()
    except Exception as e:
        print(f"Unhandled exception: {str(e)}")
        bot.stop()
