"""
Backtesting engine for cryptocurrency trading strategies
Simulates trading on historical data with realistic execution, fees, and risk management
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import logging

from config.config import DATA_DIR
from utils.logger import setup_logger

# Set up logging
logger = setup_logger("backtest_engine")

@dataclass
class Position:
    """Data class for tracking trading positions"""
    entry_date: datetime
    entry_price: float
    size: float
    direction: str  # 'long' or 'short'
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    leverage: float = 1.0
    metadata: Dict[str, Any] = None
    
    def calculate_pnl(self, exit_price: float) -> Tuple[float, float]:
        """Calculate PnL for the position"""
        if self.direction == 'long':
            if self.entry_price == 0:
                return 0, 0
            pnl = self.size * (exit_price / self.entry_price - 1) * self.leverage
            pnl_percent = (exit_price / self.entry_price - 1) * 100 * self.leverage
        else:  # 'short'
            if self.entry_price == 0:
                return 0, 0
            pnl = self.size * (1 - exit_price / self.entry_price) * self.leverage
            pnl_percent = (1 - exit_price / self.entry_price) * 100 * self.leverage
            
        return pnl, pnl_percent
    
    def is_stop_loss_triggered(self, current_price: float) -> bool:
        """Check if stop loss is triggered"""
        if self.stop_loss is None:
            return False
            
        if self.direction == 'long':
            return current_price <= self.stop_loss
        else:  # 'short'
            return current_price >= self.stop_loss
    
    def is_take_profit_triggered(self, current_price: float) -> bool:
        """Check if take profit is triggered"""
        if self.take_profit is None:
            return False
            
        if self.direction == 'long':
            return current_price >= self.take_profit
        else:  # 'short'
            return current_price <= self.take_profit


@dataclass
class Trade:
    """Data class for completed trades"""
    entry_date: datetime
    entry_price: float
    exit_date: datetime
    exit_price: float
    size: float
    direction: str
    pnl: float
    pnl_percent: float
    exit_reason: str
    trade_duration: timedelta
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    leverage: float = 1.0
    metadata: Dict[str, Any] = None


class BacktestEngine:
    """
    Backtesting engine for cryptocurrency trading strategies
    """
    
    def __init__(self, 
                 initial_capital: float = 10000.0,
                 fee_rate: float = 0.1,  # 0.1% per trade
                 slippage_pct: float = 0.05,  # 0.05% slippage
                 enable_fractional: bool = True,
                 enable_shorting: bool = False,
                 max_leverage: float = 1.0):
        """
        Initialize the backtesting engine
        
        Args:
            initial_capital: Starting capital for backtests
            fee_rate: Trading fee percentage
            slippage_pct: Slippage percentage
            enable_fractional: Allow fractional sizing of positions
            enable_shorting: Allow short positions
            max_leverage: Maximum leverage allowed
        """
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate / 100.0  # Convert percentage to decimal
        self.slippage_pct = slippage_pct / 100.0  # Convert percentage to decimal
        self.enable_fractional = enable_fractional
        self.enable_shorting = enable_shorting
        self.max_leverage = max_leverage
        
        # Output directory for results
        self.output_dir = os.path.join(DATA_DIR, "backtest_results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Store most recent backtest results
        self.last_results = None
        
        # Strategy settings
        self.strategy_name = "generic_strategy"
        self.strategy_settings = {}
        
        # Performance metrics cache
        self._metrics_cache = {}
    
    def backtest(self,
               market_data: pd.DataFrame,
               strategy_func: Callable,
               strategy_name: str = "custom_strategy",
               strategy_params: Dict[str, Any] = None,
               initial_capital: Optional[float] = None,
               position_size_pct: float = 100.0,
               stop_loss_pct: Optional[float] = None,
               take_profit_pct: Optional[float] = None,
               trailing_stop_pct: Optional[float] = None,
               max_open_positions: int = 1,
               leverage: float = 1.0,
               fee_override: Optional[float] = None,
               slippage_override: Optional[float] = None,
               start_date: Optional[str] = None,
               end_date: Optional[str] = None) -> Dict:
        """
        Run a backtest using a provided strategy function
        
        Args:
            market_data: DataFrame with OHLCV data
            strategy_func: Function that implements the trading strategy
            strategy_name: Name of the strategy for reporting
            strategy_params: Parameters to pass to the strategy function
            initial_capital: Starting capital (overrides instance setting)
            position_size_pct: Position size as percentage of capital
            stop_loss_pct: Stop loss percentage (None for no SL)
            take_profit_pct: Take profit percentage (None for no TP)
            trailing_stop_pct: Trailing stop percentage (None for no trailing stop)
            max_open_positions: Maximum number of open positions
            leverage: Leverage multiplier for position sizing
            fee_override: Override the default fee rate
            slippage_override: Override the default slippage percentage
            start_date: Date to start backtest (format: 'YYYY-MM-DD')
            end_date: Date to end backtest (format: 'YYYY-MM-DD')
            
        Returns:
            Dict with backtest results
        """
        logger.info(f"Starting backtest for strategy: {strategy_name}")
        
        # Store strategy info
        self.strategy_name = strategy_name
        self.strategy_settings = {
            "position_size_pct": position_size_pct,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "trailing_stop_pct": trailing_stop_pct,
            "max_open_positions": max_open_positions,
            "leverage": leverage
        }
        if strategy_params:
            self.strategy_settings.update(strategy_params)
        
        # Use override values if provided
        capital = initial_capital or self.initial_capital
        fee_rate = fee_override / 100.0 if fee_override is not None else self.fee_rate
        slippage = slippage_override / 100.0 if slippage_override is not None else self.slippage_pct
        
        # Convert percentage values to decimals
        position_size_pct /= 100.0
        if stop_loss_pct is not None:
            stop_loss_pct /= 100.0
        if take_profit_pct is not None:
            take_profit_pct /= 100.0
        if trailing_stop_pct is not None:
            trailing_stop_pct /= 100.0
        
        # Validate leverage
        if leverage > self.max_leverage:
            logger.warning(f"Leverage {leverage}x exceeds maximum allowed {self.max_leverage}x. Using {self.max_leverage}x.")
            leverage = self.max_leverage
        
        # Clean and prepare market data
        data = market_data.copy()
        
        # Ensure data is sorted chronologically
        data = data.sort_index()
        
        # Filter by date range if provided
        if start_date:
            start_dt = pd.to_datetime(start_date)
            data = data[data.index >= start_dt]
        if end_date:
            end_dt = pd.to_datetime(end_date)
            data = data[data.index <= end_dt]
        
        if len(data) == 0:
            logger.error("No data available for backtest after date filtering")
            return {"error": "No data available for backtest"}
        
        # Extract dates for reference
        dates = data.index.tolist()
        
        # Initialize tracking variables
        positions = []  # Current open positions
        trades = []  # Completed trades
        cash = capital  # Available cash
        equity_curve = [capital]  # Track equity over time
        dates_equity = [dates[0]]  # Dates for equity curve
        
        # Performance tracking
        peak_equity = capital
        max_drawdown = 0
        drawdown_start = None
        drawdown_end = None
        
        # Track trailing stop levels
        trailing_stops = {}  # {position_idx: highest_price}
        
        # Loop through each day/candle
        for i in range(1, len(data)):
            current_date = dates[i]
            current_candle = data.iloc[i]
            
            # Calculate current equity (cash + position values)
            current_equity = cash
            
            for idx, position in enumerate(positions):
                position_value = self._calculate_position_value(position, current_candle['close'])
                current_equity += position_value
            
            # Update equity curve
            equity_curve.append(current_equity)
            dates_equity.append(current_date)
            
            # Update drawdown metrics
            if current_equity > peak_equity:
                peak_equity = current_equity
                drawdown_start = None
            
            current_drawdown = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0
            if current_drawdown > 0:
                if drawdown_start is None:
                    drawdown_start = current_date
                
                if current_drawdown > max_drawdown:
                    max_drawdown = current_drawdown
                    drawdown_end = current_date
            
            # Check for stop-loss, take-profit, and trailing stops
            positions_to_close = []
            
            for idx, position in enumerate(positions):
                close_position = False
                exit_reason = ""
                exit_price = current_candle['close']
                
                # Check stop loss (using low price for long positions)
                if position.direction == 'long' and position.stop_loss is not None:
                    if current_candle['low'] <= position.stop_loss:
                        close_position = True
                        exit_reason = "stop_loss"
                        exit_price = position.stop_loss
                
                # Check stop loss (using high price for short positions)
                elif position.direction == 'short' and position.stop_loss is not None:
                    if current_candle['high'] >= position.stop_loss:
                        close_position = True
                        exit_reason = "stop_loss"
                        exit_price = position.stop_loss
                
                # Check take profit (using high price for long positions)
                if not close_position and position.direction == 'long' and position.take_profit is not None:
                    if current_candle['high'] >= position.take_profit:
                        close_position = True
                        exit_reason = "take_profit"
                        exit_price = position.take_profit
                
                # Check take profit (using low price for short positions)
                elif not close_position and position.direction == 'short' and position.take_profit is not None:
                    if current_candle['low'] <= position.take_profit:
                        close_position = True
                        exit_reason = "take_profit"
                        exit_price = position.take_profit
                
                # Update trailing stop if enabled
                if not close_position and trailing_stop_pct is not None:
                    if idx not in trailing_stops:
                        # Initialize trailing stop at entry
                        trailing_stops[idx] = position.entry_price
                    
                    if position.direction == 'long':
                        # Update trailing stop if price moves higher
                        if current_candle['high'] > trailing_stops[idx]:
                            trailing_stops[idx] = current_candle['high']
                            # Update stop loss based on new trailing stop level
                            position.stop_loss = trailing_stops[idx] * (1 - trailing_stop_pct)
                    else:  # Short position
                        # Update trailing stop if price moves lower
                        if current_candle['low'] < trailing_stops[idx]:
                            trailing_stops[idx] = current_candle['low']
                            # Update stop loss based on new trailing stop level
                            position.stop_loss = trailing_stops[idx] * (1 + trailing_stop_pct)
                
                if close_position:
                    positions_to_close.append((idx, exit_reason, exit_price))
            
            # Close positions identified for closing (in reverse order)
            for idx, reason, price in sorted(positions_to_close, reverse=True):
                position = positions[idx]
                result = self._close_position(position, current_date, price, reason, fee_rate, slippage)
                cash += result['cash_return']
                trades.append(result['trade'])
                # Remove position and trailing stop
                del positions[idx]
                if idx in trailing_stops:
                    del trailing_stops[idx]
            
            # Call the strategy function to get new signals
            try:
                signals = strategy_func(
                    data.iloc[:i+1],
                    positions=positions,
                    current_equity=current_equity,
                    **(strategy_params or {})
                )
            except Exception as e:
                logger.error(f"Strategy function error at {current_date}: {str(e)}")
                signals = []
            
            # Process signals to open new positions
            if signals and len(positions) < max_open_positions:
                for signal in signals:
                    if len(positions) >= max_open_positions:
                        break
                    
                    # Extract signal parameters
                    direction = signal.get('direction', 'long')
                    
                    # Skip short signals if shorting is disabled
                    if direction == 'short' and not self.enable_shorting:
                        logger.warning(f"Short signal ignored - shorting disabled")
                        continue
                    
                    # Calculate position size
                    if 'size' in signal:
                        # Fixed position size
                        size = min(signal['size'], cash * leverage)
                    else:
                        # Percentage of capital
                        size_pct = signal.get('size_pct', position_size_pct)
                        size = cash * size_pct * leverage
                    
                    # Skip if insufficient funds
                    if size <= 0:
                        logger.warning(f"Insufficient funds for {direction} position")
                        continue
                    
                    # Apply entry fee
                    entry_fee = size * fee_rate / leverage
                    adjusted_size = size - entry_fee
                    
                    # Calculate entry price with slippage
                    entry_price = current_candle['close']
                    if direction == 'long':
                        entry_price *= (1 + slippage)
                    else:
                        entry_price *= (1 - slippage)
                    
                    # Calculate stop loss and take profit levels
                    sl_price = None
                    tp_price = None
                    
                    if 'stop_loss_pct' in signal:
                        sl_pct = signal['stop_loss_pct'] / 100.0
                        if direction == 'long':
                            sl_price = entry_price * (1 - sl_pct)
                        else:
                            sl_price = entry_price * (1 + sl_pct)
                    elif stop_loss_pct is not None:
                        if direction == 'long':
                            sl_price = entry_price * (1 - stop_loss_pct)
                        else:
                            sl_price = entry_price * (1 + stop_loss_pct)
                    
                    if 'take_profit_pct' in signal:
                        tp_pct = signal['take_profit_pct'] / 100.0
                        if direction == 'long':
                            tp_price = entry_price * (1 + tp_pct)
                        else:
                            tp_price = entry_price * (1 - tp_pct)
                    elif take_profit_pct is not None:
                        if direction == 'long':
                            tp_price = entry_price * (1 + take_profit_pct)
                        else:
                            tp_price = entry_price * (1 - take_profit_pct)
                    
                    # Create new position
                    position = Position(
                        entry_date=current_date,
                        entry_price=entry_price,
                        size=adjusted_size,
                        direction=direction,
                        stop_loss=sl_price,
                        take_profit=tp_price,
                        leverage=leverage,
                        metadata=signal.get('metadata', {})
                    )
                    
                    # Add position and deduct capital
                    positions.append(position)
                    cash -= adjusted_size / leverage
                    
                    # Initialize trailing stop
                    if trailing_stop_pct is not None:
                        trailing_stops[len(positions) - 1] = entry_price
        
        # Close any open positions at the end of the backtest
        for position in positions.copy():
            result = self._close_position(
                position, 
                dates[-1], 
                data.iloc[-1]['close'], 
                "end_of_backtest", 
                fee_rate, 
                slippage
            )
            cash += result['cash_return']
            trades.append(result['trade'])
        
        # Final equity
        final_equity = cash
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(equity_curve, trades, max_drawdown, capital, dates_equity)
        
        # Build results
        results = {
            'initial_capital': capital,
            'final_equity': final_equity,
            'cash': cash,
            'start_date': dates[0].strftime('%Y-%m-%d %H:%M:%S') if hasattr(dates[0], 'strftime') else str(dates[0]),
            'end_date': dates[-1].strftime('%Y-%m-%d %H:%M:%S') if hasattr(dates[-1], 'strftime') else str(dates[-1]),
            'total_bars': len(data),
            'metrics': metrics,
            'trades': [self._trade_to_dict(t) for t in trades],
            'equity_curve': {
                'dates': [d.strftime('%Y-%m-%d %H:%M:%S') if hasattr(d, 'strftime') else str(d) for d in dates_equity],
                'equity': equity_curve
            },
            'settings': {
                'strategy': strategy_name,
                **self.strategy_settings
            }
        }
        
        # Save as last results
        self.last_results = results
        
        # Log summary
        logger.info(f"Backtest completed: {round(metrics['total_return_pct'], 2)}% return, {metrics['total_trades']} trades")
        
        return results
    
    def _close_position(self, 
                      position: Position, 
                      exit_date: datetime, 
                      exit_price: float, 
                      reason: str, 
                      fee_rate: float, 
                      slippage: float) -> Dict:
        """
        Close a position and calculate results
        
        Args:
            position: Position to close
            exit_date: Date of closing
            exit_price: Price at closing
            reason: Reason for closing
            fee_rate: Trading fee rate
            slippage: Slippage percentage
        """
        # Apply slippage to exit price
        slippage_adjusted_price = exit_price
        if position.direction == 'long':
            slippage_adjusted_price *= (1 - slippage)
        else:  # 'short'
            slippage_adjusted_price *= (1 + slippage)
        
        # Calculate position value
        position_value = self._calculate_position_value(position, slippage_adjusted_price)
        
        # Apply trading fee
        fee = position_value * fee_rate
        position_value -= fee
        
        # Calculate PnL
        pnl, pnl_percent = position.calculate_pnl(slippage_adjusted_price)
        
        # Adjust for fees
        pnl -= fee
        
        # Calculate trade duration
        if isinstance(position.entry_date, str):
            entry_date = pd.to_datetime(position.entry_date)
        else:
            entry_date = position.entry_date
            
        if isinstance(exit_date, str):
            exit_date = pd.to_datetime(exit_date)
            
        trade_duration = exit_date - entry_date
        
        # Create trade record
        trade = Trade(
            entry_date=position.entry_date,
            entry_price=position.entry_price,
            exit_date=exit_date,
            exit_price=slippage_adjusted_price,
            size=position.size,
            direction=position.direction,
            pnl=pnl,
            pnl_percent=pnl_percent,
            exit_reason=reason,
            trade_duration=trade_duration,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
            leverage=position.leverage,
            metadata=position.metadata
        )
        
        return {
            "trade": trade,
            "cash_return": position_value
        }
    
    def _calculate_position_value(self, position: Position, current_price: float) -> float:
        """Calculate current value of a position"""
        if position.direction == 'long':
            return position.size * (current_price / position.entry_price)
        else:  # 'short'
            return position.size * (2 - current_price / position.entry_price)
    
    def _calculate_performance_metrics(self, 
                                     equity_curve: List[float], 
                                     trades: List[Trade],
                                     max_drawdown: float,
                                     initial_capital: float,
                                     dates_equity: List) -> Dict:
        """
        Calculate performance metrics from backtest results
        
        Args:
            equity_curve: List of equity values
            trades: List of completed trades
            max_drawdown: Maximum drawdown as decimal
            initial_capital: Initial capital
            dates_equity: List of dates for the equity curve
        """
        if not trades:
            return {
                "total_return": 0.0,
                "total_return_pct": 0.0,
                "annual_return_pct": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "calmar_ratio": 0.0,
                "avg_trade_duration_days": 0.0
            }
        
        # Calculate return metrics
        final_equity = equity_curve[-1]
        total_return = final_equity - initial_capital
        total_return_pct = (total_return / initial_capital) * 100
        
        # Calculate annualized return
        if len(dates_equity) > 1:
            if isinstance(dates_equity[0], str):
                start_date = pd.to_datetime(dates_equity[0])
                end_date = pd.to_datetime(dates_equity[-1])
            else:
                start_date = dates_equity[0]
                end_date = dates_equity[-1]
                
            days = (end_date - start_date).days
            if days > 0:
                annual_return_pct = ((1 + total_return_pct/100) ** (365/days) - 1) * 100
            else:
                annual_return_pct = 0.0
        else:
            annual_return_pct = 0.0
        
        # Trade statistics
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        avg_win = sum(t.pnl_percent for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t.pnl_percent for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate daily returns
        daily_returns = []
        for i in range(1, len(equity_curve)):
            daily_return = (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
            daily_returns.append(daily_return)
        
        # Risk metrics
        sharpe_ratio = 0
        sortino_ratio = 0
        calmar_ratio = 0
        
        if daily_returns:
            avg_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            
            # Sharpe ratio
            risk_free_rate = 0.02 / 252  # Daily risk-free rate (assuming 2% annually)
            if std_return > 0:
                sharpe_ratio = ((avg_return - risk_free_rate) / std_return) * np.sqrt(252)
            
            # Sortino ratio (downside deviation)
            negative_returns = [r for r in daily_returns if r < 0]
            if negative_returns:
                downside_deviation = np.std(negative_returns)
                if downside_deviation > 0:
                    sortino_ratio = ((avg_return - risk_free_rate) / downside_deviation) * np.sqrt(252)
            
            # Calmar ratio
            if max_drawdown > 0:
                calmar_ratio = annual_return_pct / 100 / max_drawdown
        
        # Average trade duration
        durations = [(t.exit_date - t.entry_date).days for t in trades]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        # Compile metrics
        metrics = {
            "total_return": float(total_return),
            "total_return_pct": float(total_return_pct),
            "annual_return_pct": float(annual_return_pct),
            "total_trades": len(trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": float(win_rate * 100),  # Convert to percentage
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "profit_factor": float(profit_factor),
            "max_drawdown_pct": float(max_drawdown * 100),  # Convert to percentage
            "sharpe_ratio": float(sharpe_ratio),
            "sortino_ratio": float(sortino_ratio),
            "calmar_ratio": float(calmar_ratio),
            "avg_trade_duration_days": float(avg_duration)
        }
        
        # Cache the metrics
        self._metrics_cache = metrics
        
        return metrics
    
    def _trade_to_dict(self, trade: Trade) -> Dict:
        """Convert Trade object to dictionary"""
        return {
            'entry_date': trade.entry_date.strftime('%Y-%m-%d %H:%M:%S') if hasattr(trade.entry_date, 'strftime') else str(trade.entry_date),
            'entry_price': float(trade.entry_price),
            'exit_date': trade.exit_date.strftime('%Y-%m-%d %H:%M:%S') if hasattr(trade.exit_date, 'strftime') else str(trade.exit_date),
            'exit_price': float(trade.exit_price),
            'size': float(trade.size),
            'direction': trade.direction,
            'pnl': float(trade.pnl),
            'pnl_percent': float(trade.pnl_percent),
            'exit_reason': trade.exit_reason,
            'duration_days': trade.trade_duration.days,
            'stop_loss': float(trade.stop_loss) if trade.stop_loss is not None else None,
            'take_profit': float(trade.take_profit) if trade.take_profit is not None else None,
            'leverage': float(trade.leverage),
            'metadata': trade.metadata
        }
    
    def visualize_results(self, results: Dict = None, save_path: Optional[str] = None) -> None:
        """
        Visualize backtest results with comprehensive charts
        
        Args:
            results: Results from backtest (uses last_results if None)
            save_path: Path to save the visualization (displays if None)
        """
        if results is None:
            if self.last_results is None:
                logger.error("No backtest results available to visualize")
                return
            results = self.last_results
        
        # Create plotting figure and subplots
        fig = plt.figure(figsize=(15, 18))
        grid = GridSpec(5, 2, figure=fig)
        
        # Extract data from results
        equity_curve = results['equity_curve']['equity']
        dates = [pd.to_datetime(d) for d in results['equity_curve']['dates']]
        trades = results['trades']
        metrics = results['metrics']
        
        # 1. Equity Curve (top plot spanning both columns)
        ax_equity = fig.add_subplot(grid[0, :])
        ax_equity.plot(dates, equity_curve, color='#1f77b4', linewidth=2)
        ax_equity.set_title('Equity Curve', fontsize=14)
        ax_equity.set_ylabel('Capital')
        ax_equity.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax_equity.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax_equity.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add initial and final equity annotations
        ax_equity.annotate(f"Initial: ${results['initial_capital']:.2f}", 
                          xy=(dates[0], results['initial_capital']),
                          xytext=(dates[0], results['initial_capital']*1.05),
                          arrowprops=dict(facecolor='black', width=0.5, headwidth=4, alpha=0.5))
        
        ax_equity.annotate(f"Final: ${results['final_equity']:.2f}", 
                          xy=(dates[-1], results['final_equity']),
                          xytext=(dates[-1], results['final_equity']*1.05),
                          arrowprops=dict(facecolor='black', width=0.5, headwidth=4, alpha=0.5))
        
        # Mark winning and losing trades on the equity curve
        for trade in trades:
            try:
                entry_date = pd.to_datetime(trade['entry_date'])
                exit_date = pd.to_datetime(trade['exit_date'])
                
                # Find nearest indices in the date list
                entry_idx = min(range(len(dates)), key=lambda i: abs(dates[i] - entry_date))
                exit_idx = min(range(len(dates)), key=lambda i: abs(dates[i] - exit_date))
                
                if entry_idx < len(equity_curve) and exit_idx < len(equity_curve):
                    # Draw markers for entry and exit
                    color = 'green' if trade['pnl'] > 0 else 'red'
                    ax_equity.plot([dates[entry_idx], dates[exit_idx]], 
                                  [equity_curve[entry_idx], equity_curve[exit_idx]], 
                                  color=color, alpha=0.5, linewidth=1.5)
            except Exception as e:
                continue
        
        # 2. Drawdown Chart
        ax_dd = fig.add_subplot(grid[1, :])
        
        # Calculate drawdown from equity curve
        peak = np.maximum.accumulate(equity_curve)
        drawdown = np.zeros_like(equity_curve, dtype=float)
        for i in range(len(equity_curve)):
            if peak[i] > 0:  # Avoid division by zero
                drawdown[i] = (peak[i] - equity_curve[i]) / peak[i] * 100
        
        ax_dd.fill_between(dates, drawdown, 0, color='red', alpha=0.3)
        ax_dd.set_title('Drawdown (%)', fontsize=14)
        ax_dd.set_ylabel('Drawdown %')
        ax_dd.grid(True, alpha=0.3)
        ax_dd.set_ylim(bottom=0, top=max(drawdown)*1.1 if max(drawdown) > 0 else 5)
        
        # Format x-axis dates
        ax_dd.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax_dd.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add annotation for max drawdown
        max_dd_idx = np.argmax(drawdown)
        if max_dd_idx > 0:
            ax_dd.annotate(f"Max DD: {drawdown[max_dd_idx]:.2f}%", 
                         xy=(dates[max_dd_idx], drawdown[max_dd_idx]),
                         xytext=(dates[max_dd_idx], drawdown[max_dd_idx]*1.1),
                         arrowprops=dict(facecolor='black', width=0.5, headwidth=4, alpha=0.5))
        
        # 3. Return Distribution
        ax_ret_dist = fig.add_subplot(grid[2, 0])
        
        # Extract trade returns
        trade_returns = [t['pnl_percent'] for t in trades]
        
        if trade_returns:
            # Filter out extreme outliers for better visualization
            filtered_returns = [r for r in trade_returns if -100 <= r <= 100]
            
            # Create histogram with KDE
            sns.histplot(filtered_returns, kde=True, bins=20, ax=ax_ret_dist, color='skyblue')
            ax_ret_dist.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            ax_ret_dist.axvline(x=np.mean(filtered_returns), color='red', linestyle='-', alpha=0.7)
            
            ax_ret_dist.set_title('Trade Return Distribution', fontsize=14)
            ax_ret_dist.set_xlabel('Return %')
            ax_ret_dist.set_ylabel('Frequency')
        else:
            ax_ret_dist.text(0.5, 0.5, "No trades to display", 
                           horizontalalignment='center', verticalalignment='center')
        
        # 4. Cumulative Returns vs Benchmark (if available)
        ax_cum_ret = fig.add_subplot(grid[2, 1])
        
        # Calculate cumulative returns
        cum_returns = [(eq / results['initial_capital'] - 1) * 100 for eq in equity_curve]
        
        # Plot cumulative returns
        ax_cum_ret.plot(dates, cum_returns, color='blue', linewidth=2, label='Strategy')
        
        # If we have buy-and-hold benchmark info, plot it
        if 'comparison' in results and 'buy_and_hold_return_pct' in results['comparison']:
            # Create a simple linear interpolation for the benchmark return
            benchmark_return = results['comparison']['buy_and_hold_return_pct']
            benchmark_curve = np.linspace(0, benchmark_return, len(dates))
            ax_cum_ret.plot(dates, benchmark_curve, color='gray', linestyle='--', 
                           linewidth=1.5, label='Buy & Hold')
            
            # Add annotation for outperformance
            outperf = results['comparison']['outperformance_pct']
            ax_cum_ret.text(0.02, 0.95, f"Outperformance: {outperf:.2f}%", 
                           transform=ax_cum_ret.transAxes, fontsize=10,
                           bbox=dict(facecolor='white', alpha=0.7))
        
        ax_cum_ret.set_title('Cumulative Returns (%)', fontsize=14)
        ax_cum_ret.set_ylabel('Return %')
        ax_cum_ret.legend()
        ax_cum_ret.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax_cum_ret.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax_cum_ret.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 5. Monthly Returns Heatmap
        ax_monthly = fig.add_subplot(grid[3, :])
        
        # Calculate monthly returns
        if len(dates) > 30:  # Only if we have enough data
            try:
                # Convert equity curve to daily returns
                daily_returns = pd.Series([0] + [(equity_curve[i] / equity_curve[i-1] - 1) * 100 
                                               for i in range(1, len(equity_curve))], 
                                         index=dates)
                
                # Resample to monthly returns
                monthly_returns = daily_returns.resample('M').apply(
                    lambda x: ((1 + x/100).prod() - 1) * 100
                )
                
                # Create a pivot table for the heatmap
                monthly_pivot = monthly_returns.to_frame('return')
                monthly_pivot['year'] = monthly_pivot.index.year
                monthly_pivot['month'] = monthly_pivot.index.month
                monthly_pivot = monthly_pivot.pivot('year', 'month', 'return')
                
                # Plot heatmap
                sns.heatmap(monthly_pivot, annot=True, fmt=".1f", cmap="RdYlGn", 
                           center=0, ax=ax_monthly, cbar_kws={'label': 'Return %'})
                
                # Adjust labels
                month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                ax_monthly.set_xticklabels(month_labels, rotation=0)
                ax_monthly.set_title('Monthly Returns (%)', fontsize=14)
            except Exception as e:
                ax_monthly.text(0.5, 0.5, f"Could not create monthly heatmap: {str(e)}", 
                              horizontalalignment='center', verticalalignment='center')
        else:
            ax_monthly.text(0.5, 0.5, "Not enough data for monthly analysis", 
                          horizontalalignment='center', verticalalignment='center')
        
        # 6. Performance Metrics Table
        ax_metrics = fig.add_subplot(grid[4, 0])
        ax_metrics.axis('off')
        
        # Create a table with the metrics
        metrics_data = []
        metrics_to_display = [
            ('Total Return', f"{metrics['total_return_pct']:.2f}%"),
            ('Annual Return', f"{metrics['annual_return_pct']:.2f}%"),
            ('Total Trades', f"{metrics['total_trades']}"),
            ('Win Rate', f"{metrics['win_rate']:.2f}%"),
            ('Profit Factor', f"{metrics['profit_factor']:.2f}"),
            ('Sharpe Ratio', f"{metrics['sharpe_ratio']:.2f}"),
            ('Max Drawdown', f"{metrics['max_drawdown_pct']:.2f}%"),
            ('Avg Win', f"{metrics['avg_win']:.2f}%"),
            ('Avg Loss', f"{metrics['avg_loss']:.2f}%"),
            ('Avg Duration', f"{metrics['avg_trade_duration_days']:.1f} days")
        ]
        
        for metric, value in metrics_to_display:
            metrics_data.append([metric, value])
        
        metrics_table = ax_metrics.table(
            cellText=metrics_data,
            colLabels=['Metric', 'Value'],
            cellLoc='center',
            loc='center',
            colWidths=[0.6, 0.4]
        )
        
        # Adjust table style
        metrics_table.auto_set_font_size(False)
        metrics_table.set_fontsize(10)
        metrics_table.scale(1, 1.5)
        
        ax_metrics.set_title('Performance Metrics', fontsize=14)
        
        # 7. Trade Analysis
        ax_trade = fig.add_subplot(grid[4, 1])
        
        # Extract trade types and their counts
        exit_reasons = {}
        for trade in trades:
            reason = trade['exit_reason']
            if reason not in exit_reasons:
                exit_reasons[reason] = {'count': 0, 'win': 0, 'loss': 0}
            
            exit_reasons[reason]['count'] += 1
            if trade['pnl'] > 0:
                exit_reasons[reason]['win'] += 1
            else:
                exit_reasons[reason]['loss'] += 1
        
        # Prepare data for bar chart
        reasons = list(exit_reasons.keys())
        win_counts = [exit_reasons[r]['win'] for r in reasons]
        loss_counts = [exit_reasons[r]['loss'] for r in reasons]
        
        if reasons:
            # Create stacked bar chart
            bar_width = 0.8
            ax_trade.bar(reasons, win_counts, bar_width, label='Win', color='green', alpha=0.7)
            ax_trade.bar(reasons, loss_counts, bar_width, bottom=win_counts, 
                        label='Loss', color='red', alpha=0.7)
            
            # Add annotations
            for i, reason in enumerate(reasons):
                total = exit_reasons[reason]['count']
                win_rate = exit_reasons[reason]['win'] / total * 100 if total > 0 else 0
                ax_trade.text(i, exit_reasons[reason]['count'] + 0.5, 
                             f"{win_rate:.1f}%", ha='center')
            
            ax_trade.set_title('Trade Exit Analysis', fontsize=14)
            ax_trade.set_ylabel('Number of Trades')
            ax_trade.legend()
        else:
            ax_trade.text(0.5, 0.5, "No trades to analyze", 
                        horizontalalignment='center', verticalalignment='center')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or display the figure
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Backtest visualization saved to {save_path}")
        else:
            plt.show()
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """
        Save backtest results to a JSON file
        
        Args:
            filename: Custom filename (will use strategy name and timestamp if None)
            
        Returns:
            Path to the saved file
        """
        if self.last_results is None:
            logger.error("No backtest results to save")
            return ""
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            strategy_name = self.strategy_name.lower().replace(" ", "_")
            filename = f"{strategy_name}_backtest_{timestamp}.json"
        
        # Ensure file has .json extension
        if not filename.endswith('.json'):
            filename += '.json'
        
        # Create full path
        file_path = os.path.join(self.output_dir, filename)
        
        try:
            # Save results as JSON
            with open(file_path, 'w') as f:
                json.dump(self.last_results, f, indent=2, default=str)
            
            logger.info(f"Backtest results saved to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving backtest results: {str(e)}")
            return ""
    
    def compare_strategies(self, results_list: List[Dict], labels: List[str] = None) -> Dict:
        """
        Compare multiple backtest results
        
        Args:
            results_list: List of backtest result dictionaries
            labels: List of labels for each strategy (uses strategy names if None)
            
        Returns:
            Dictionary with comparison results
        """
        if not results_list:
            logger.error("No results provided for comparison")
            return {}
        
        # Use strategy names as labels if not provided
        if labels is None:
            labels = [r.get('settings', {}).get('strategy', f"Strategy {i+1}") 
                    for i, r in enumerate(results_list)]
        
        # Ensure we have enough labels
        if len(labels) < len(results_list):
            labels.extend([f"Strategy {i+1}" for i in range(len(labels), len(results_list))])
        
        # Extract key metrics for comparison
        comparison = {
            'strategies': labels,
            'metrics': {},
            'returns': {},
            'drawdowns': {},
            'trades': {}
        }
        
        # Collect metrics
        metrics_to_compare = [
            'total_return_pct', 'annual_return_pct', 'max_drawdown_pct', 
            'sharpe_ratio', 'win_rate', 'profit_factor'
        ]
        
        for metric in metrics_to_compare:
            comparison['metrics'][metric] = [
                r.get('metrics', {}).get(metric, 0) for r in results_list
            ]
        
        # Collect trade counts
        comparison['trades']['total_trades'] = [
            r.get('metrics', {}).get('total_trades', 0) for r in results_list
        ]
        
        comparison['trades']['winning_trades'] = [
            r.get('metrics', {}).get('winning_trades', 0) for r in results_list
        ]
        
        comparison['trades']['losing_trades'] = [
            r.get('metrics', {}).get('losing_trades', 0) for r in results_list
        ]
        
        # Return the comparison
        return comparison
    
    def visualize_comparison(self, comparison: Dict, save_path: Optional[str] = None) -> None:
        """
        Visualize strategy comparison
        
        Args:
            comparison: Results from compare_strategies method
            save_path: Path to save visualization (None to display)
        """
        if not comparison:
            logger.error("No comparison data provided")
            return
        
        strategies = comparison.get('strategies', [])
        if not strategies:
            logger.error("No strategies found in comparison data")
            return
        
        n_strategies = len(strategies)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(14, 10))
        grid = GridSpec(2, 2)
        
        # 1. Return and Drawdown Comparison
        ax_returns = fig.add_subplot(grid[0, 0])
        
        # Collect return and drawdown data
        total_returns = comparison['metrics'].get('total_return_pct', [0] * n_strategies)
        annual_returns = comparison['metrics'].get('annual_return_pct', [0] * n_strategies)
        drawdowns = comparison['metrics'].get('max_drawdown_pct', [0] * n_strategies)
        
        # Create bar chart
        x = np.arange(n_strategies)
        width = 0.35
        
        ax_returns.bar(x - width/2, total_returns, width, label='Total Return', color='blue')
        ax_returns.bar(x + width/2, annual_returns, width, label='Annual Return', color='green')
        
        ax_returns.set_title('Return Comparison', fontsize=14)
        ax_returns.set_ylabel('Return %')
        ax_returns.set_xticks(x)
        ax_returns.set_xticklabels(strategies, rotation=45, ha='right')
        ax_returns.legend()
        
        # Add drawdown data on top
        for i, dd in enumerate(drawdowns):
            ax_returns.annotate(f"DD: {dd:.1f}%", 
                              xy=(i, max(total_returns[i], annual_returns[i]) + 2), 
                              ha='center', color='red', fontweight='bold')
        
        # 2. Trade Statistics
        ax_trades = fig.add_subplot(grid[0, 1])
        
        # Collect trade data
        total_trades = comparison['trades'].get('total_trades', [0] * n_strategies)
        winning_trades = comparison['trades'].get('winning_trades', [0] * n_strategies)
        losing_trades = comparison['trades'].get('losing_trades', [0] * n_strategies)
        
        # Create stacked bar chart
        ax_trades.bar(strategies, winning_trades, label='Winning Trades', color='green', alpha=0.7)
        ax_trades.bar(strategies, losing_trades, bottom=winning_trades, 
                    label='Losing Trades', color='red', alpha=0.7)
        
        # Add win rate annotations
        for i, (win, total) in enumerate(zip(winning_trades, total_trades)):
            if total > 0:
                win_rate = win / total * 100
                ax_trades.annotate(f"{win_rate:.1f}%", 
                                xy=(i, total + 1), 
                                ha='center')
        
        ax_trades.set_title('Trade Analysis', fontsize=14)
        ax_trades.set_ylabel('Number of Trades')
        ax_trades.set_xticklabels(strategies, rotation=45, ha='right')
        ax_trades.legend()
        
        # 3. Performance Metrics Radar Chart
        ax_radar = fig.add_subplot(grid[1, :], polar=True)
        
        # Metrics to include in the radar chart
        radar_metrics = ['total_return_pct', 'annual_return_pct', 'sharpe_ratio', 
                        'win_rate', 'profit_factor']
        radar_labels = ['Total Return', 'Annual Return', 'Sharpe Ratio', 
                       'Win Rate', 'Profit Factor']
        
        # Normalize metric values to 0-1 scale for radar chart
        normalized_metrics = []
        for metric in radar_metrics:
            values = comparison['metrics'].get(metric, [0] * n_strategies)
            
            # Handle special cases
            if metric == 'max_drawdown_pct':
                # Invert drawdown (lower is better)
                values = [1 - v/100 if v < 100 else 0 for v in values]
            elif metric in ['profit_factor', 'sharpe_ratio']:
                # Cap extreme values
                values = [min(v, 5) for v in values]
            
            # Normalize
            max_val = max(values) if max(values) > 0 else 1
            normalized_metrics.append([v / max_val for v in values])
        
        # Set up the radar chart
        angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Plot each strategy
        for i, strategy in enumerate(strategies):
            values = [metrics[i] for metrics in normalized_metrics]
            values += values[:1]  # Close the loop
            
            ax_radar.plot(angles, values, linewidth=2, label=strategy)
            ax_radar.fill(angles, values, alpha=0.1)
        
        # Set up the radar chart labels and styling
        ax_radar.set_thetagrids(np.degrees(angles[:-1]), radar_labels)
        ax_radar.set_ylim(0, 1.1)
        ax_radar.set_title('Strategy Performance Comparison', fontsize=14, pad=20)
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or display the figure
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Strategy comparison visualization saved to {save_path}")
        else:
            plt.show()
    
    def backtest_model_strategy(self, market_data: pd.DataFrame, model, feature_engineering,
                              threshold: float = 0.6, exit_threshold: float = 0.4,
                              **kwargs) -> Dict:
        """
        Backtest a model-based trading strategy
        
        Args:
            market_data: Market data for backtesting
            model: Prediction model (e.g., LSTM, Transformer)
            feature_engineering: Feature engineering pipeline
            threshold: Threshold for buy signals (0.0-1.0)
            exit_threshold: Threshold for sell signals (0.0-1.0)
            **kwargs: Additional parameters for backtest method
            
        Returns:
            Backtest results
        """
        # Extract model parameters
        sequence_length = getattr(model, 'input_length', 60)
        
        if hasattr(model, 'horizon_periods'):
            horizons = model.horizon_periods
        else:
            horizons = getattr(model, 'prediction_horizons', [12, 24, 96])
        
        # Define the strategy function
        def model_strategy(data, positions=None, current_equity=None):
            signals = []
            
            if positions and positions[0].direction == 'long':
                # Skip if we already have a position
                return signals
            
            if len(data) < sequence_length + 20:  # Need enough data for features
                return signals
            
            try:
                # Prepare features
                featured_data = feature_engineering.create_features(
                    data,
                    include_time_features=True,
                    include_price_patterns=True
                )
                
                normalized_data = feature_engineering.scale_features(
                    featured_data,
                    is_training=False,
                    method='standard',
                    feature_group='lstm'
                )
                
                # Create sequence for prediction
                X, _ = feature_engineering.create_multi_horizon_data(
                    normalized_data,
                    sequence_length=sequence_length,
                    horizons=horizons,
                    is_training=False
                )
                
                if len(X) > 0:
                    # Get prediction for latest data
                    prediction = model.model.predict(X[-1:], verbose=0)
                    
                    # Ensure predictions is a list
                    if not isinstance(prediction, list):
                        prediction = [prediction]
                    
                    # Generate signals based on prediction threshold
                    if prediction[0][0][0] > threshold:  # Strong bullish signal
                        signals.append({
                            'direction': 'long',
                            'metadata': {
                                'prediction': float(prediction[0][0][0]),
                                'confidence': float(abs(prediction[0][0][0] - 0.5) * 2)
                            }
                        })
            except Exception as e:
                logger.error(f"Error in model strategy: {str(e)}")
            
            return signals
        
        # Run the backtest with the model strategy
        return self.backtest(
            market_data=market_data,
            strategy_func=model_strategy,
            strategy_name=f"Model_{model.__class__.__name__}",
            **kwargs
        )