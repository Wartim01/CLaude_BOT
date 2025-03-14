"""
Performance tracking and analytics for the trading bot
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta

from config.config import DATA_DIR
from utils.logger import setup_logger

logger = setup_logger("performance_metrics")

class PerformanceTracker:
    """
    Tracks and analyzes trading performance metrics
    """
    def __init__(self, exchange, trade_history_file: str = None, performance_file: str = None):
        """Initialize the performance tracker"""
        # Performance history
        self.balance_history = []
        self.trade_history = []
        
        # Overall metrics
        self.initial_balance = 0.0
        self.current_balance = 0.0
        self.peak_balance = 0.0
        self.total_pnl = 0.0
        self.total_pnl_percent = 0.0
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        
        # Trade metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.win_rate = 0.0
        self.profit_factor = 0.0
        self.avg_win = 0.0
        self.avg_loss = 0.0
        self.largest_win = 0.0
        self.largest_loss = 0.0
        
        # Time-based metrics
        self.start_time = datetime.now()
        self.trading_days = 0
        self.profitable_days = 0
        self.daily_returns = []
        
        # Data storage
        self.metrics_dir = os.path.join(DATA_DIR, "performance")
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Load previous data if available
        self._load_history()
        
        self.exchange = exchange
    
    def update(self, current_balance: float, 
             open_positions: List[Dict] = None, 
             closed_trades: List[Dict] = None) -> None:
        """
        Update performance metrics with latest data
        
        Args:
            current_balance: Current account balance
            open_positions: List of currently open positions
            closed_trades: List of recently closed trades
        """
        try:
            # Use get_open_positions if available; otherwise fallback to empty list.
            if hasattr(self.exchange, "get_open_positions"):
                open_positions = self.exchange.get_open_positions()
            else:
                open_positions = []
            
            # Initialize balance on first update
            if not self.balance_history:
                self.initial_balance = current_balance
                self.peak_balance = current_balance
            
            # Update current balance
            self.current_balance = current_balance
            
            # Update balance history
            timestamp = datetime.now().isoformat()
            self.balance_history.append({
                "timestamp": timestamp,
                "balance": current_balance,
                "open_positions": len(open_positions) if open_positions else 0
            })
            
            # Limit history size
            max_history = 10000
            if len(self.balance_history) > max_history:
                self.balance_history = self.balance_history[-max_history:]
            
            # Update peak balance
            if current_balance > self.peak_balance:
                self.peak_balance = current_balance
            
            # Update drawdown
            if self.peak_balance > 0:
                self.current_drawdown = (self.peak_balance - current_balance) / self.peak_balance * 100
                self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
            
            # Process closed trades
            if closed_trades:
                # Find new trades (not already in our history)
                known_trade_ids = {t.get("id") for t in self.trade_history if "id" in t}
                
                new_trades = [
                    trade for trade in closed_trades 
                    if trade.get("id") not in known_trade_ids and "exit_time" in trade
                ]
                
                # Add new trades to history
                for trade in new_trades:
                    self.trade_history.append(trade)
                    self._process_trade(trade)
            
            # Calculate total metrics
            self._calculate_total_metrics()
            
            # Calculate time-based metrics
            self._calculate_time_metrics()
            
            # Save periodically (e.g., every 100 updates)
            if len(self.balance_history) % 100 == 0:
                self._save_history()
        
        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")
    
    def _process_trade(self, trade: Dict) -> None:
        """
        Process a single trade for metrics
        
        Args:
            trade: Trade data dictionary
        """
        # Increment total trades
        self.total_trades += 1
        
        # Extract PnL
        pnl = trade.get("pnl_amount", 0)
        pnl_pct = trade.get("pnl_percentage", 0)
        
        # Update win/loss counters
        if pnl > 0:
            self.winning_trades += 1
            self.avg_win = ((self.avg_win * (self.winning_trades - 1)) + pnl) / self.winning_trades
            self.largest_win = max(self.largest_win, pnl)
        else:
            self.losing_trades += 1
            # Track losses as positive numbers for averaging
            self.avg_loss = ((self.avg_loss * (self.losing_trades - 1)) + abs(pnl)) / self.losing_trades
            self.largest_loss = max(self.largest_loss, abs(pnl))
    
    def _calculate_total_metrics(self) -> None:
        """Calculate overall performance metrics"""
        # Calculate total PnL
        if self.initial_balance > 0:
            self.total_pnl = self.current_balance - self.initial_balance
            self.total_pnl_percent = (self.total_pnl / self.initial_balance) * 100
        
        # Calculate win rate
        if self.total_trades > 0:
            self.win_rate = (self.winning_trades / self.total_trades) * 100
        
        # Calculate profit factor
        total_profit = self.avg_win * self.winning_trades
        total_loss = self.avg_loss * self.losing_trades
        
        if total_loss > 0:
            self.profit_factor = total_profit / total_loss
        elif total_profit > 0:
            self.profit_factor = float('inf')  # No losses, all profit
        else:
            self.profit_factor = 0.0  # No profits
    
    def _calculate_time_metrics(self) -> None:
        """Calculate time-based performance metrics"""
        if not self.balance_history:
            return
        
        # Calculate trading days
        if len(self.balance_history) >= 2:
            start_time = datetime.fromisoformat(self.balance_history[0]["timestamp"])
            self.trading_days = (datetime.now() - start_time).days + 1
            
            # Pre-process data for daily metrics
            df = pd.DataFrame(self.balance_history)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
            
            # Resample to daily and calculate returns
            daily = df.resample('D').last()
            daily['balance'] = daily['balance'].ffill()
            
            if len(daily) >= 2:
                daily['return'] = daily['balance'].pct_change()
                
                # Count profitable days
                self.profitable_days = sum(daily['return'] > 0)
                
                # Store daily returns
                self.daily_returns = daily['return'].dropna().tolist()
    
    def get_summary(self) -> Dict:
        """
        Get a summary of current performance metrics
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            "current_balance": self.current_balance,
            "initial_balance": self.initial_balance,
            "total_pnl": self.total_pnl,
            "total_pnl_percent": self.total_pnl_percent,
            "peak_balance": self.peak_balance,
            "current_drawdown": self.current_drawdown,
            "max_drawdown": self.max_drawdown,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "trading_days": self.trading_days,
            "profitable_days": self.profitable_days,
            "sharpe_ratio": self._calculate_sharpe_ratio(),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_detailed_report(self) -> Dict:
        """
        Get a detailed performance report with additional metrics
        
        Returns:
            Dictionary with detailed performance metrics
        """
        basic_metrics = self.get_summary()
        
        # Add advanced metrics
        advanced_metrics = {
            "sortino_ratio": self._calculate_sortino_ratio(),
            "calmar_ratio": self._calculate_calmar_ratio(),
            "volatility": self._calculate_volatility(),
            "max_consecutive_wins": self._calculate_consecutive(True),
            "max_consecutive_losses": self._calculate_consecutive(False),
            "avg_trade_duration": self._calculate_avg_trade_duration(),
            "best_pair": self._get_best_pair(),
            "worst_pair": self._get_worst_pair(),
            "monthly_performance": self._get_monthly_performance()
        }
        
        return {**basic_metrics, **advanced_metrics}
    
    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio based on daily returns
        
        Args:
            risk_free_rate: Annual risk-free rate as decimal
            
        Returns:
            Sharpe ratio or 0 if not enough data
        """
        if not self.daily_returns or len(self.daily_returns) < 5:
            return 0.0
        
        # Convert annual risk-free rate to daily
        daily_risk_free = (1 + risk_free_rate) ** (1/365) - 1
        
        # Calculate excess returns
        excess_returns = np.array(self.daily_returns) - daily_risk_free
        
        # Calculate Sharpe ratio (annualized)
        sharpe = np.mean(excess_returns) / np.std(excess_returns, ddof=1) if np.std(excess_returns, ddof=1) > 0 else 0
        sharpe_annualized = sharpe * np.sqrt(365)  # Annualize
        
        return float(sharpe_annualized)
    
    def _calculate_sortino_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sortino ratio based on daily returns
        
        Args:
            risk_free_rate: Annual risk-free rate as decimal
            
        Returns:
            Sortino ratio or 0 if not enough data
        """
        if not self.daily_returns or len(self.daily_returns) < 5:
            return 0.0
        
        # Convert annual risk-free rate to daily
        daily_risk_free = (1 + risk_free_rate) ** (1/365) - 1
        
        # Calculate excess returns
        excess_returns = np.array(self.daily_returns) - daily_risk_free
        
        # Calculate downside deviation (standard deviation of negative returns)
        negative_returns = excess_returns[excess_returns < 0]
        
        if len(negative_returns) == 0 or np.std(negative_returns, ddof=1) == 0:
            return 0.0 if np.mean(excess_returns) <= 0 else float('inf')
        
        # Calculate Sortino ratio (annualized)
        sortino = np.mean(excess_returns) / np.std(negative_returns, ddof=1)
        sortino_annualized = sortino * np.sqrt(365)  # Annualize
        
        return float(sortino_annualized)
    
    def _calculate_calmar_ratio(self) -> float:
        """
        Calculate Calmar ratio (annualized return / max drawdown)
        
        Returns:
            Calmar ratio or 0 if not enough data
        """
        if self.trading_days < 30 or self.max_drawdown == 0:
            return 0.0
        
        # Calculate annualized return
        if self.initial_balance > 0:
            total_return = self.total_pnl / self.initial_balance
            years = self.trading_days / 365
            annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
            
            # Calmar ratio is annualized return divided by max drawdown (as a decimal)
            return float(annualized_return / (self.max_drawdown / 100))
        return 0.0
    
    def _calculate_volatility(self) -> float:
        """
        Calculate annualized volatility based on daily returns
        
        Returns:
            Annualized volatility as percentage
        """
        if not self.daily_returns or len(self.daily_returns) < 5:
            return 0.0
        
        daily_volatility = np.std(self.daily_returns, ddof=1)
        annualized_volatility = daily_volatility * np.sqrt(365)
        
        return float(annualized_volatility * 100)  # Convert to percentage
    
    def _calculate_consecutive(self, wins: bool) -> int:
        """
        Calculate maximum consecutive wins or losses
        
        Args:
            wins: If True, calculate consecutive wins; otherwise, consecutive losses
            
        Returns:
            Maximum consecutive wins or losses
        """
        if not self.trade_history:
            return 0
        
        # Extract trade results (win/loss)
        results = []
        for trade in self.trade_history:
            pnl = trade.get("pnl_amount", 0)
            results.append(pnl > 0)
        
        # Find maximum consecutive occurrence
        max_consecutive = 0
        current_consecutive = 0
        target = wins  # True for wins, False for losses
        
        for result in results:
            if result == target:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_avg_trade_duration(self) -> float:
        """
        Calculate average trade duration in hours
        
        Returns:
            Average trade duration in hours
        """
        if not self.trade_history:
            return 0.0
        
        durations = []
        for trade in self.trade_history:
            if "entry_time" in trade and "exit_time" in trade:
                try:
                    entry_time = datetime.fromisoformat(trade["entry_time"])
                    exit_time = datetime.fromisoformat(trade["exit_time"])
                    
                    duration = (exit_time - entry_time).total_seconds() / 3600  # Convert to hours
                    durations.append(duration)
                except:
                    pass
        
        return float(np.mean(durations)) if durations else 0.0
    
    def _get_best_pair(self) -> Dict:
        """
        Get the best performing trading pair
        
        Returns:
            Info about the best performing pair
        """
        if not self.trade_history:
            return {"symbol": None, "pnl": 0.0, "win_rate": 0.0, "trades": 0}
        
        # Group trades by symbol
        symbol_performance = {}
        
        for trade in self.trade_history:
            symbol = trade.get("symbol")
            pnl = trade.get("pnl_amount", 0.0)
            
            if symbol not in symbol_performance:
                symbol_performance[symbol] = {
                    "total_pnl": 0.0,
                    "wins": 0,
                    "losses": 0,
                    "trades": 0
                }
            
            symbol_performance[symbol]["total_pnl"] += pnl
            symbol_performance[symbol]["trades"] += 1
            
            if pnl > 0:
                symbol_performance[symbol]["wins"] += 1
            else:
                symbol_performance[symbol]["losses"] += 1
        
        # Find best symbol by PnL
        best_symbol = None
        best_pnl = float('-inf')
        
        for symbol, perf in symbol_performance.items():
            if perf["trades"] >= 3 and perf["total_pnl"] > best_pnl:  # At least 3 trades
                best_symbol = symbol
                best_pnl = perf["total_pnl"]
        
        if best_symbol:
            perf = symbol_performance[best_symbol]
            return {
                "symbol": best_symbol,
                "pnl": float(perf["total_pnl"]),
                "win_rate": float(perf["wins"] / perf["trades"] * 100) if perf["trades"] > 0 else 0.0,
                "trades": perf["trades"]
            }
        
        return {"symbol": None, "pnl": 0.0, "win_rate": 0.0, "trades": 0}
    
    def _get_worst_pair(self) -> Dict:
        """
        Get the worst performing trading pair
        
        Returns:
            Info about the worst performing pair
        """
        if not self.trade_history:
            return {"symbol": None, "pnl": 0.0, "win_rate": 0.0, "trades": 0}
        
        # Group trades by symbol
        symbol_performance = {}
        
        for trade in self.trade_history:
            symbol = trade.get("symbol")
            pnl = trade.get("pnl_amount", 0.0)
            
            if symbol not in symbol_performance:
                symbol_performance[symbol] = {
                    "total_pnl": 0.0,
                    "wins": 0,
                    "losses": 0,
                    "trades": 0
                }
            
            symbol_performance[symbol]["total_pnl"] += pnl
            symbol_performance[symbol]["trades"] += 1
            
            if pnl > 0:
                symbol_performance[symbol]["wins"] += 1
            else:
                symbol_performance[symbol]["losses"] += 1
        
        # Find worst symbol by PnL
        worst_symbol = None
        worst_pnl = float('inf')
        
        for symbol, perf in symbol_performance.items():
            if perf["trades"] >= 3 and perf["total_pnl"] < worst_pnl:  # At least 3 trades
                worst_symbol = symbol
                worst_pnl = perf["total_pnl"]
        
        if worst_symbol:
            perf = symbol_performance[worst_symbol]
            return {
                "symbol": worst_symbol,
                "pnl": float(perf["total_pnl"]),
                "win_rate": float(perf["wins"] / perf["trades"] * 100) if perf["trades"] > 0 else 0.0,
                "trades": perf["trades"]
            }
        
        return {"symbol": None, "pnl": 0.0, "win_rate": 0.0, "trades": 0}
    
    def _get_monthly_performance(self) -> Dict:
        """
        Get performance metrics by month
        
        Returns:
            Dictionary with monthly performance data
        """
        if not self.balance_history or len(self.balance_history) < 2:
            return {}
        
        try:
            # Convert balance history to DataFrame
            df = pd.DataFrame(self.balance_history)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
            
            # Resample to monthly
            monthly = df.resample('M').last()
            monthly['balance'] = monthly['balance'].ffill()
            
            # Calculate monthly returns
            monthly['return'] = monthly['balance'].pct_change() * 100
            
            # Create result dictionary
            result = {}
            
            for date, row in monthly.iterrows():
                month_key = date.strftime('%Y-%m')
                result[month_key] = {
                    "balance": float(row['balance']) if not pd.isna(row['balance']) else None,
                    "return": float(row['return']) if not pd.isna(row['return']) else None
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating monthly performance: {str(e)}")
            return {}
    
    def _load_history(self) -> None:
        """Load performance history from disk"""
        balance_path = os.path.join(self.metrics_dir, "balance_history.json")
        trades_path = os.path.join(self.metrics_dir, "trade_history.json")
        metrics_path = os.path.join(self.metrics_dir, "metrics.json")
        
        # Load balance history
        if os.path.exists(balance_path):
            try:
                with open(balance_path, 'r') as f:
                    self.balance_history = json.load(f)
                logger.info(f"Loaded balance history: {len(self.balance_history)} entries")
            except Exception as e:
                logger.error(f"Error loading balance history: {str(e)}")
        
        # Load trade history
        if os.path.exists(trades_path):
            try:
                with open(trades_path, 'r') as f:
                    self.trade_history = json.load(f)
                logger.info(f"Loaded trade history: {len(self.trade_history)} entries")
            except Exception as e:
                logger.error(f"Error loading trade history: {str(e)}")
        
        # Load metrics
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                    
                # Update instance attributes
                for key, value in metrics.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
                
                logger.info(f"Loaded performance metrics")
            except Exception as e:
                logger.error(f"Error loading metrics: {str(e)}")
    
    def _save_history(self) -> None:
        """Save performance history to disk"""
        balance_path = os.path.join(self.metrics_dir, "balance_history.json")
        trades_path = os.path.join(self.metrics_dir, "trade_history.json")
        metrics_path = os.path.join(self.metrics_dir, "metrics.json")
        
        # Save balance history
        try:
            with open(balance_path, 'w') as f:
                json.dump(self.balance_history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving balance history: {str(e)}")
        
        # Save trade history
        try:
            with open(trades_path, 'w') as f:
                json.dump(self.trade_history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving trade history: {str(e)}")
        
        # Save metrics
        try:
            metrics = {
                "initial_balance": self.initial_balance,
                "current_balance": self.current_balance,
                "peak_balance": self.peak_balance,
                "total_pnl": self.total_pnl,
                "total_pnl_percent": self.total_pnl_percent,
                "current_drawdown": self.current_drawdown,
                "max_drawdown": self.max_drawdown,
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "losing_trades": self.losing_trades,
                "win_rate": self.win_rate,
                "profit_factor": self.profit_factor,
                "avg_win": self.avg_win,
                "avg_loss": self.avg_loss,
                "largest_win": self.largest_win,
                "largest_loss": self.largest_loss,
                "trading_days": self.trading_days,
                "profitable_days": self.profitable_days,
                "daily_returns": self.daily_returns
            }
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")