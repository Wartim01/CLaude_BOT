import os
import sys
# Update sys.path to insert the project root at the beginning so that "config" can be imported
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # Ajout de l'import manquant
from matplotlib.gridspec import GridSpec  # Ajout de l'import manquant
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import tensorflow as tf
from datetime import datetime
import json

from config.config import DATA_DIR
from utils.logger import setup_logger
from ai.models.feature_engineering import FeatureEngineering

logger = setup_logger("model_validator")

class ModelValidator:
    """
    Validator class for model evaluation and backtesting
    Provides methods to evaluate model performance and visualization tools
    """
    
    def __init__(self, model=None, feature_engineering=None):
        """
        Initialize the model validator
        
        Args:
            model: Pre-trained model (LSTM, Transformer, etc.)
            feature_engineering: Feature engineering pipeline
        """
        self.model = model
        self.feature_engineering = feature_engineering or FeatureEngineering()
        
        # Directories for saving results
        self.output_dir = os.path.join(DATA_DIR, "evaluation")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def evaluate_on_test_set(self, test_data: pd.DataFrame, horizon_idx: int = 0) -> Dict:
        """
        Evaluate model on a test set
        
        Args:
            test_data: DataFrame with test data
            horizon_idx: Index of prediction horizon to evaluate (0=short-term, 1=mid-term, 2=long-term)
            
        Returns:
            Dict with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not set. Please provide a model in constructor or set it with set_model()")
        
        # Preprocess the test data
        featured_data = self.feature_engineering.create_features(
            test_data,
            include_time_features=True,
            include_price_patterns=True
        )
        
        # Normalize the data
        normalized_data = self.feature_engineering.scale_features(
            featured_data,
            is_training=False,
            method='standard',
            feature_group='lstm'
        )
        
        # Create sequences
        sequence_length = getattr(self.model, 'input_length', 60)
        
        if hasattr(self.model, 'horizon_periods'):
            # For multi-horizon models
            horizons = self.model.horizon_periods
        else:
            # Default horizons if not specified
            horizons = [12, 24, 96]
            
        # Prepare sequences
        X_test, y_test = self.feature_engineering.create_multi_horizon_data(
            normalized_data,
            sequence_length=sequence_length,
            horizons=horizons,
            is_training=False
        )
        
        # Get predictions
        predictions = self.model.model.predict(X_test)
        
        # Ensure predictions is a list for multi-output models
        if not isinstance(predictions, list):
            predictions = [predictions]
        
        # Get the target output for the specified horizon
        if horizon_idx >= len(predictions):
            horizon_idx = 0
            
        y_true = y_test[horizon_idx]
        y_pred_proba = predictions[horizon_idx]
        
        # Convert probabilities to binary predictions
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate evaluation metrics
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred)),
            "recall": float(recall_score(y_true, y_pred)),
            "f1_score": float(f1_score(y_true, y_pred)),
        }
        
        # Calculate ROC and AUC
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        metrics["auc"] = float(auc(fpr, tpr))
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate balanced class accuracy
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_acc = (sensitivity + specificity) / 2
        metrics["balanced_accuracy"] = float(balanced_acc)
        
        # Group all results
        results = {
            "metrics": metrics,
            "predictions": {
                "y_true": y_true.tolist(),
                "y_pred_proba": y_pred_proba.flatten().tolist(),
                "y_pred": y_pred.flatten().tolist()
            },
            "confusion_matrix": cm.tolist(),
            "roc_curve": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist()
            },
            "horizon_idx": horizon_idx,
            "horizon_periods": horizons[horizon_idx] if horizon_idx < len(horizons) else None
        }
        
        return results
    
    def backtest_model(self, market_data: pd.DataFrame, initial_capital: float = 1000.0, 
                      position_size_pct: float = 100.0, use_stop_loss: bool = True,
                      stop_loss_pct: float = 2.0, take_profit_pct: float = 4.0,
                      fee_rate: float = 0.1, slippage_pct: float = 0.05) -> Dict:
        """
        Backtest the model on historical market data
        
        Args:
            market_data: DataFrame with OHLCV data
            initial_capital: Initial capital for backtesting
            position_size_pct: Position size as percentage of capital
            use_stop_loss: Whether to use stop loss
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            fee_rate: Trading fee percentage
            slippage_pct: Slippage percentage
            
        Returns:
            Dict with backtesting results
        """
        if self.model is None:
            raise ValueError("Model not set")
        
        # Create a copy of market data to avoid modifying the original
        data = market_data.copy()
        
        # Ensure data is chronologically sorted
        data = data.sort_index()
        
        # Extract dates for reference
        dates = data.index.tolist()
        
        # Initialize parameters
        capital = initial_capital
        position = None
        trades = []
        equity_curve = [initial_capital]
        dates_equity = [dates[0]]
        
        # Performance tracking
        peak_equity = initial_capital
        max_drawdown = 0
        drawdown_start = None
        drawdown_end = None
        
        # Convert percentages to decimals
        position_size_pct = position_size_pct / 100
        stop_loss_pct = stop_loss_pct / 100
        take_profit_pct = take_profit_pct / 100
        fee_rate = fee_rate / 100
        slippage_pct = slippage_pct / 100
        
        # Create features for the entire dataset
        featured_data = self.feature_engineering.create_features(
            data,
            include_time_features=True,
            include_price_patterns=True
        )
        
        normalized_data = self.feature_engineering.scale_features(
            featured_data,
            is_training=False,
            method='standard',
            feature_group='lstm'
        )
        
        # Get model parameters
        sequence_length = getattr(self.model, 'input_length', 60)
        
        if hasattr(self.model, 'horizon_periods'):
            horizons = self.model.horizon_periods
        else:
            horizons = [12, 24, 96]
        
        # Minimum data needed for features and prediction
        min_data_points = sequence_length + max(horizons)
        
        # Iterate through each day (starting from the minimum required data points)
        for i in range(min_data_points, len(data)):
            current_date = dates[i]
            current_price = data.iloc[i]['close']
            
            # Update equity curve
            if position is None:
                # No open position, equity is just capital
                current_equity = capital
            else:
                # Calculate current equity with open position
                position_value = position['size'] * (current_price / position['entry_price'])
                current_equity = capital - position['size'] + position_value
            
            equity_curve.append(current_equity)
            dates_equity.append(current_date)
            
            # Update max drawdown
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
            
            # Check if we need to close an open position
            if position is not None:
                # Check stop loss - corrected calculation
                if use_stop_loss and current_price <= position['stop_loss']:
                    # Close position at stop loss price (with slippage)
                    exit_price = max(position['stop_loss'] * (1 - slippage_pct), 0.01)  # Prevent negative prices
                    
                    # Calculate profit/loss (corrected calculation)
                    position_value = position['size'] * (exit_price / position['entry_price'])
                    pnl = position_value - position['size']
                    
                    # Apply trading fee
                    fee = position_value * fee_rate
                    pnl -= fee
                    
                    # Update capital
                    capital += position_value - fee
                    
                    # Record trade
                    trade = {
                        'entry_date': position['entry_date'],
                        'entry_price': position['entry_price'],
                        'exit_date': current_date,
                        'exit_price': exit_price,
                        'size': position['size'],
                        'pnl': pnl,
                        'pnl_percent': (pnl / position['size']) * 100,
                        'exit_reason': 'stop_loss'
                    }
                    trades.append(trade)
                    
                    # Reset position
                    position = None
                
                # Check take profit - corrected calculation
                elif current_price >= position['take_profit']:
                    # Close position at take profit (with slippage)
                    exit_price = position['take_profit'] * (1 - slippage_pct)
                    
                    # Calculate profit/loss (corrected calculation)
                    position_value = position['size'] * (exit_price / position['entry_price'])
                    pnl = position_value - position['size']
                    
                    # Apply trading fee
                    fee = position_value * fee_rate
                    pnl -= fee
                    
                    # Update capital
                    capital += position_value - fee
                    
                    # Record trade
                    trade = {
                        'entry_date': position['entry_date'],
                        'entry_price': position['entry_price'],
                        'exit_date': current_date,
                        'exit_price': exit_price,
                        'size': position['size'],
                        'pnl': pnl,
                        'pnl_percent': (pnl / position['size']) * 100,
                        'exit_reason': 'take_profit'
                    }
                    trades.append(trade)
                    
                    # Reset position
                    position = None
                
                # Get model prediction for early exit - check less frequently to improve performance
                elif i % 5 == 0:  # Check every 5 candles
                    try:
                        # Get a window of data up to current point
                        data_window = data.iloc[:i+1]
                        normalized_window = normalized_data.iloc[:i+1]
                        
                        # Create sequence for prediction
                        X, _ = self.feature_engineering.create_multi_horizon_data(
                            normalized_window,
                            sequence_length=sequence_length,
                            horizons=horizons,
                            is_training=False
                        )
                        
                        # Make prediction (take only the most recent sample)
                        X_latest = X[-1:] if len(X) > 0 else X
                        if len(X_latest) > 0:
                            predictions = self.model.model.predict(X_latest, verbose=0)  # Suppressing verbose output
                            
                            # Ensure predictions is a list
                            if not isinstance(predictions, list):
                                predictions = [predictions]
                            
                            # Check if the prediction suggests closing the position
                            # For long positions, close if prediction is bearish (< 0.5)
                            if len(predictions) > 0 and len(predictions[0]) > 0 and predictions[0][0][0] < 0.4:  # Strong bearish signal
                                # Close position at current price (with slippage)
                                exit_price = current_price * (1 - slippage_pct)
                                
                                # Calculate profit/loss (corrected calculation)
                                position_value = position['size'] * (exit_price / position['entry_price'])
                                pnl = position_value - position['size']
                                
                                # Apply trading fee
                                fee = position_value * fee_rate
                                pnl -= fee
                                
                                # Update capital
                                capital += position_value - fee
                                
                                # Record trade
                                trade = {
                                    'entry_date': position['entry_date'],
                                    'entry_price': position['entry_price'],
                                    'exit_date': current_date,
                                    'exit_price': exit_price,
                                    'size': position['size'],
                                    'pnl': pnl,
                                    'pnl_percent': (pnl / position['size']) * 100,
                                    'exit_reason': 'model_signal'
                                }
                                trades.append(trade)
                                
                                # Reset position
                                position = None
                    except Exception as e:
                        logger.warning(f"Error predicting for position exit at index {i}: {str(e)}")
            
            # Check if we should open a new position
            if position is None:
                try:
                    # Get a window of data up to current point
                    normalized_window = normalized_data.iloc[:i+1]
                    
                    # Create sequence for prediction
                    X, _ = self.feature_engineering.create_multi_horizon_data(
                        normalized_window,
                        sequence_length=sequence_length,
                        horizons=horizons,
                        is_training=False
                    )
                    
                    # Make prediction (take only the most recent sample)
                    X_latest = X[-1:] if len(X) > 0 else X
                    if len(X_latest) > 0:
                        predictions = self.model.model.predict(X_latest, verbose=0)  # Suppressing verbose output
                        
                        # Ensure predictions is a list
                        if not isinstance(predictions, list):
                            predictions = [predictions]
                        
                        # Check if prediction suggests opening a position
                        # Only consider the first horizon (short-term) for trading decisions
                        if len(predictions) > 0 and len(predictions[0]) > 0 and predictions[0][0][0] > 0.6:  # Strong bullish signal
                            # Calculate position size
                            position_size = capital * position_size_pct
                            
                            # Only trade if we have enough capital
                            if position_size >= 10:  # Minimum position size
                                # Apply entry fee
                                entry_fee = position_size * fee_rate
                                adjusted_position_size = position_size - entry_fee
                                
                                # Calculate entry price (with slippage)
                                entry_price = current_price * (1 + slippage_pct)
                                
                                # Open a long position
                                position = {
                                    'entry_date': current_date,
                                    'entry_price': entry_price,
                                    'size': adjusted_position_size,
                                    'stop_loss': entry_price * (1 - stop_loss_pct),
                                    'take_profit': entry_price * (1 + take_profit_pct),
                                }
                                
                                # Deduct position size from available capital
                                capital -= position_size
                except Exception as e:
                    logger.warning(f"Error predicting for position entry at index {i}: {str(e)}")
        
        # Close any open position at the end of the period
        if position is not None:
            # Close position at final price
            exit_price = data.iloc[-1]['close'] * (1 - slippage_pct)
            
            # Calculate profit/loss (corrected calculation)
            position_value = position['size'] * (exit_price / position['entry_price'])
            pnl = position_value - position['size']
            
            # Apply trading fee
            fee = position_value * fee_rate
            pnl -= fee
            
            # Update capital
            final_capital = capital + position_value - fee
            
            # Record trade
            trade = {
                'entry_date': position['entry_date'],
                'entry_price': position['entry_price'],
                'exit_date': dates[-1],
                'exit_price': exit_price,
                'size': position['size'],
                'pnl': pnl,
                'pnl_percent': (pnl / position['size']) * 100,
                'exit_reason': 'end_of_period'
            }
            trades.append(trade)
        else:
            final_capital = capital
        
        # Calculate performance metrics
        total_return = (final_capital - initial_capital) / initial_capital * 100
        
        # Fix potential NaN or infinite values in trades
        for t in trades:
            for key, value in t.items():
                if isinstance(value, (int, float)) and (np.isnan(value) or np.isinf(value)):
                    t[key] = 0
        
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        avg_win = sum(t['pnl_percent'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t['pnl_percent'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        profit_factor = sum(t['pnl'] for t in winning_trades) / abs(sum(t['pnl'] for t in losing_trades)) if losing_trades and sum(t['pnl'] for t in losing_trades) < 0 else float('inf')
        
        # Calculate daily returns for Sharpe ratio - ensure no NaN or infinite values
        daily_returns = []
        for i in range(1, len(equity_curve)):
            if equity_curve[i-1] > 0:  # Prevent division by zero
                ret = (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
                if not np.isnan(ret) and not np.isinf(ret):
                    daily_returns.append(ret)
        
        sharpe_ratio = 0
        if daily_returns:
            avg_daily_return = sum(daily_returns) / len(daily_returns)
            std_daily_return = np.std(daily_returns) if len(daily_returns) > 1 else 0
            if std_daily_return > 0:
                sharpe_ratio = (avg_daily_return / std_daily_return) * np.sqrt(252)  # Annualized
        
        # Compile results
        results = {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return_pct': total_return,
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown_pct': max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio,
            'trades': trades,
            'equity_curve': {
                'dates': [d.strftime('%Y-%m-%d %H:%M:%S') if isinstance(d, pd.Timestamp) else str(d) for d in dates_equity],
                'equity': equity_curve
            },
            'model_performance': {
                'return_pct': total_return,
                'win_rate_pct': win_rate * 100,
                'profit_factor': profit_factor,
                'max_drawdown_pct': max_drawdown * 100,
                'sharpe_ratio': sharpe_ratio
            }
        }
        
        return results
    
    def compare_with_baseline(self, market_data: pd.DataFrame, initial_capital: float = 1000.0) -> Dict:
        """
        Compare model performance with baseline buy-and-hold strategy
        
        Args:
            market_data: DataFrame with OHLCV data
            initial_capital: Initial capital for backtesting
        """
        # Run model backtest
        model_results = self.backtest_model(market_data, initial_capital=initial_capital)
        
        # Calculate buy-and-hold returns
        start_price = market_data['close'].iloc[0]
        end_price = market_data['close'].iloc[-1]
        
        buy_and_hold_return = (end_price - start_price) / start_price * 100
        buy_and_hold_final_capital = initial_capital * (1 + buy_and_hold_return/100)
        
        # Calculate outperformance
        outperformance = model_results['total_return_pct'] - buy_and_hold_return
        
        comparison = {
            'model_return_pct': model_results['total_return_pct'],
            'buy_and_hold_return_pct': buy_and_hold_return,
            'outperformance_pct': outperformance,
            'model_final_capital': model_results['final_capital'],
            'buy_and_hold_final_capital': buy_and_hold_final_capital
        }
        
        return comparison
    
    def visualize_predictions(self, market_data: pd.DataFrame, horizon_idx: int = 0, 
                            start_idx: Optional[int] = None, end_idx: Optional[int] = None,
                            save_path: Optional[str] = None) -> None:
        """
        Visualize model predictions vs actual price movements
        
        Args:
            market_data: DataFrame with OHLCV data
            horizon_idx: Index of prediction horizon to visualize
            start_idx: Start index for visualization (None for all)
            end_idx: End index for visualization (None for all)
            save_path: Path to save the visualization (None for display)
        """
        if self.model is None:
            raise ValueError("Model not set")
        
        # Create features
        featured_data = self.feature_engineering.create_features(
            market_data,
            include_time_features=True,
            include_price_patterns=True
        )
        
        normalized_data = self.feature_engineering.scale_features(
            featured_data,
            is_training=False,
            method='standard',
            feature_group='lstm'
        )
        
        # Get model parameters
        sequence_length = getattr(self.model, 'input_length', 60)
        
        if hasattr(self.model, 'horizon_periods'):
            horizons = self.model.horizon_periods
        else:
            horizons = [12, 24, 96]
        
        # Create sequences
        X, y = self.feature_engineering.create_multi_horizon_data(
            normalized_data,
            sequence_length=sequence_length,
            horizons=horizons,
            is_training=False
        )
        
        # Limit data for visualization
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(X)
        
        X_vis = X[start_idx:end_idx]
        y_vis = [y_h[start_idx:end_idx] for y_h in y]
        
        # Get predictions
        if len(X_vis) > 0:
            predictions = self.model.model.predict(X_vis)
            
            # Ensure predictions is a list
            if not isinstance(predictions, list):
                predictions = [predictions]
            
            # Get dates for the X sequences (end date of each sequence)
            dates = market_data.index[sequence_length:][start_idx:end_idx]
            
            # For directional predictions
            direction_pred = predictions[horizon_idx]
            direction_true = y_vis[horizon_idx]
            
            # Create figure with subplots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14), gridspec_kw={'height_ratios': [3, 1, 1]})
            
            # Plot 1: Price with buy/sell signals
            ax1.plot(market_data['close'][sequence_length:].iloc[start_idx:end_idx], color='blue', label='Price')
            
            # Add buy/sell signals based on model predictions
            buy_signals = []
            sell_signals = []
            
            for i, pred in enumerate(direction_pred):
                if pred > 0.7:  # Strong buy signal
                    buy_signals.append((dates[i], market_data['close'].loc[dates[i]]))
                elif pred < 0.3:  # Strong sell signal
                    sell_signals.append((dates[i], market_data['close'].loc[dates[i]]))
            
            if buy_signals:
                buy_x, buy_y = zip(*buy_signals)
                ax1.scatter(buy_x, buy_y, color='green', s=70, marker='^', label='Buy Signal')
            
            if sell_signals:
                sell_x, sell_y = zip(*sell_signals)
                ax1.scatter(sell_x, sell_y, color='red', s=70, marker='v', label='Sell Signal')
            
            ax1.set_title(f'Price Chart with Model Signals (Horizon: {horizons[horizon_idx]} periods)')
            ax1.set_ylabel('Price')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Model Prediction Probability
            ax2.plot(dates, direction_pred, color='purple', label='Direction Probability')
            ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.6)
            ax2.fill_between(dates, 0.5, direction_pred, where=(direction_pred > 0.5), 
                             color='green', alpha=0.3, label='Bullish')
            ax2.fill_between(dates, direction_pred, 0.5, where=(direction_pred < 0.5), 
                             color='red', alpha=0.3, label='Bearish')
            ax2.set_ylabel('Probability')
            ax2.set_ylim(0, 1)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Prediction Accuracy
            correct_preds = (direction_pred > 0.5).astype(int) == direction_true.astype(int)
            colors = ['green' if correct else 'red' for correct in correct_preds]
            ax3.bar(dates, [1] * len(dates), color=colors, width=0.8, 
                   label='Prediction Accuracy (Green = Correct)')
            
            # Calculate and show accuracy metrics
            accuracy = np.mean(correct_preds)
            ax3.text(dates[0], 0.5, f'Accuracy: {accuracy:.2f}', 
                    fontsize=12, ha='left', va='center')
            
            ax3.set_ylabel('Correct/Wrong')
            ax3.set_yticks([])
            ax3.legend()
            
            # Format x-axis dates
            for ax in [ax1, ax2, ax3]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Adjust layout and save/show
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
        else:
            logger.warning("No data to visualize")

    def visualize_performance(self, backtest_results: Dict, save_path: Optional[str] = None) -> None:
        """
        Visualize backtest performance metrics
        
        Args:
            backtest_results: Results from the backtest_model method
            save_path: Path to save the visualization (None for display)
        """
        # Extract data from results
        equity_curve = backtest_results['equity_curve']['equity']
        dates = [pd.to_datetime(d) for d in backtest_results['equity_curve']['dates']]
        trades = backtest_results['trades']
        
        # Create figure with subplots
        fig = plt.figure(figsize=(14, 14))
        gs = GridSpec(3, 2, figure=fig)
        
        # Plot 1: Equity Curve
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(dates, equity_curve, color='blue', linewidth=2)
        ax1.set_title('Equity Curve', fontsize=14)
        ax1.set_ylabel('Capital')
        ax1.grid(True, alpha=0.3)
        
        # Mark winning and losing trades on the equity curve
        for trade in trades:
            try:
                entry_date = pd.to_datetime(trade['entry_date'])
                exit_date = pd.to_datetime(trade['exit_date'])
                
                # Find nearest points in our dates list
                entry_idx = min(range(len(dates)), key=lambda i: abs(dates[i] - entry_date))
                exit_idx = min(range(len(dates)), key=lambda i: abs(dates[i] - exit_date))
                
                if 0 <= entry_idx < len(equity_curve) and 0 <= exit_idx < len(equity_curve):
                    # Draw markers for entry and exit, color based on profit/loss
                    color = 'green' if trade['pnl'] > 0 else 'red'
                    ax1.plot([dates[entry_idx], dates[exit_idx]], 
                            [equity_curve[entry_idx], equity_curve[exit_idx]], 
                            color=color, alpha=0.7, linewidth=1)
            except Exception as e:
                logger.warning(f"Error plotting trade: {str(e)}")
                continue
            
        # Plot 2: Drawdown
        ax2 = fig.add_subplot(gs[1, :])
        
        # Calculate drawdown from equity curve
        peak = np.maximum.accumulate(equity_curve)
        drawdown = np.zeros_like(equity_curve)
        for i in range(len(equity_curve)):
            if peak[i] > 0:  # Prevent division by zero
                drawdown[i] = (peak[i] - equity_curve[i]) / peak[i] * 100
        
        ax2.fill_between(dates, drawdown, 0, color='red', alpha=0.3)
        ax2.set_title('Drawdown (%)', fontsize=14)
        ax2.set_ylabel('Drawdown %')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Trade Distribution
        ax3 = fig.add_subplot(gs[2, 0])
        
        # Extract trade returns
        trade_returns = [t['pnl_percent'] for t in trades]
        
        # Create histogram
        if trade_returns:
            # Remove any extreme outliers for better visualization
            filtered_returns = [r for r in trade_returns if -100 <= r <= 100]
            sns.histplot(filtered_returns, ax=ax3, kde=True, bins=20)
            ax3.axvline(x=0, color='black', linestyle='--')
        
        ax3.set_title('Trade Return Distribution (%)', fontsize=14)
        ax3.set_xlabel('Return %')
        
        # Plot 4: Performance Metrics Table
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.axis('off')
        
        # Extract key metrics
        metrics = [
            ('Total Return', f"{backtest_results['total_return_pct']:.2f}%"),
            ('Number of Trades', str(backtest_results['total_trades'])),
            ('Win Rate', f"{backtest_results['win_rate']:.2f}%"),
            ('Profit Factor', f"{backtest_results['profit_factor']:.2f}"),
            ('Max Drawdown', f"{backtest_results['max_drawdown_pct']:.2f}%"),
            ('Sharpe Ratio', f"{backtest_results['sharpe_ratio']:.2f}"),
            ('Avg Win', f"{backtest_results['avg_win']:.2f}%"),
            ('Avg Loss', f"{backtest_results['avg_loss']:.2f}%")
        ]
        
        # Create table
        table_data = []
        for metric, value in metrics:
            table_data.append([metric, value])
        
        ax4.table(
            cellText=table_data,
            colLabels=['Metric', 'Value'],
            loc='center',
            cellLoc='center',
            colWidths=[0.6, 0.4]
        )
        
        ax4.set_title('Performance Metrics', fontsize=14)
        
        # Format x-axis dates
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Adjust layout and save/show
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def save_results(self, results: Dict, name: str) -> str:
        """
        Save evaluation or backtest results to a file
        
        Args:
            results: Results dictionary
            name: Base name for the file
            
        Returns:
            Path to the saved file
        """
        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{name}_{timestamp}.json"
        
        # Create full path
        filepath = os.path.join(self.output_dir, filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save results as JSON
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")
        
        return filepath