# Uncomment the following line to disable oneDNN custom operations and turn off numerical round-off differences:
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
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
        try:
            featured_data = self.feature_engineering.create_features(
                test_data,
                include_time_features=True,
                include_price_patterns=True
            )
        except Exception as e:
            logger.error(f"Error creating features: {str(e)}")
            # Return a consistent structure with empty metrics
            return {
                "input_shape": None,
                "metrics": {},
                "predictions": {"y_pred": []},
                "confusion_matrix": None,
                "roc_curve": None,
                "horizon_idx": horizon_idx,
                "horizon_periods": None,
                "error": str(e)
            }
        
        # Normalize the data
        try:
            normalized_data = self.feature_engineering.scale_features(
                featured_data,
                is_training=False,
                method='standard',
                feature_group='lstm'
            )
        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
            # Return a consistent structure with empty metrics
            return {
                "input_shape": None,
                "metrics": {},
                "predictions": {"y_pred": []},
                "confusion_matrix": None,
                "roc_curve": None,
                "horizon_idx": horizon_idx,
                "horizon_periods": None,
                "error": str(e)
            }
        
        # Create sequences
        sequence_length = getattr(self.model, 'input_length', 60)
        
        # Get expected feature dimension from model if available
        expected_feature_dim = getattr(self.model, 'feature_dim', None)
        if expected_feature_dim:
            logger.info(f"Model expects feature dimension: {expected_feature_dim}")
        
        if hasattr(self.model, 'horizon_periods'):
            # For multi-horizon models
            horizons = self.model.horizon_periods
        else:
            # Default horizons if not specified
            horizons = [12, 24, 96]
            
        # Check if we need to compute or infer labels
        logger.info(f"Attempting to create sequences with horizons={horizons}")
        target_col = None
        
        # Look for potential target columns
        potential_target_cols = ['target', 'label', 'direction']
        for col in potential_target_cols:
            if col in normalized_data.columns:
                target_col = col
                logger.info(f"Found target column: {target_col}")
                break
        
        # If no target column found, we'll try to compute future returns
        if target_col is None and 'close' in normalized_data.columns:
            logger.info("No target column found, computing future returns as potential labels")
            # Compute future returns
            for h in horizons:
                normalized_data[f'future_return_{h}'] = normalized_data['close'].pct_change(h).shift(-h)
                normalized_data[f'target_{h}'] = (normalized_data[f'future_return_{h}'] > 0).astype(int)
            
            target_col = f'target_{horizons[0]}'
            logger.info(f"Created synthetic target column: {target_col}")
        
        # Prepare sequences
        try:
            logger.info("Creating sequences for evaluation...")
            result = self.feature_engineering.create_multi_horizon_data(
                normalized_data,
                sequence_length=sequence_length,
                horizons=horizons,
                is_training=False
            )
            
            # Handle return value properly - could be tuple of (X, y) or just X
            if isinstance(result, tuple) and len(result) == 2:
                X_test, y_test = result
                logger.info(f"Successfully created sequences with labels: X_shape={X_test.shape}, y_length={len(y_test) if isinstance(y_test, list) else 'N/A'}")
            else:
                X_test = result
                y_test = None  # No labels available
                logger.warning("No labels available in test data for evaluation")
        except ValueError as e:
            logger.error(f"Error creating sequences: {str(e)}")
            # Try with is_training=True as fallback
            try:
                logger.info("Trying alternative approach with is_training=True...")
                result = self.feature_engineering.create_multi_horizon_data(
                    normalized_data,
                    sequence_length=sequence_length,
                    horizons=horizons,
                    is_training=True
                )
                X_test, y_test = result
                logger.info(f"Successfully created sequences with fallback approach: X_shape={X_test.shape}")
            except Exception as fallback_e:
                logger.error(f"Fallback approach also failed: {str(fallback_e)}")
                # Return partial result with empty metrics but consistent structure
                return {
                    "input_shape": None,
                    "metrics": {},
                    "predictions": {"y_pred": []},
                    "confusion_matrix": None,
                    "roc_curve": None,
                    "horizon_idx": horizon_idx,
                    "horizon_periods": horizons[horizon_idx] if horizon_idx < len(horizons) else None,
                    "error": str(fallback_e)
                }

        # After creating sequences, verify feature dimensions match model expectations
        if isinstance(X_test, np.ndarray) and expected_feature_dim is not None:
            actual_feature_dim = X_test.shape[2]
            logger.info(f"Actual feature dimension from data: {actual_feature_dim}")
            
            if actual_feature_dim != expected_feature_dim:
                logger.warning(f"Truncating feature dimension from {actual_feature_dim} to {expected_feature_dim} to match model")
                X_test = X_test[:, :, :expected_feature_dim]

        expected_shape = (None, self.model.input_length, self.model.feature_dim)
        if X_test is not None and len(X_test.shape) == 3:
            if X_test.shape[1:] != expected_shape[1:]:
                logger.warning(f"Shape mismatch: expected {expected_shape[1:]}, got {X_test.shape[1:]}")

        # Verify we have valid y_test data
        if y_test is None or (isinstance(y_test, list) and (len(y_test) == 0 or horizon_idx >= len(y_test) or y_test[horizon_idx] is None or len(y_test[horizon_idx]) == 0)):
            logger.warning("No valid labels available in test data for evaluation")
            
            # Try to generate predictions even without labels
            try:
                if X_test is not None and len(X_test) > 0:
                    predictions = self.model.model.predict(X_test)
                    if not isinstance(predictions, list):
                        predictions = [predictions]
                    
                    # Create a result with predictions but no metrics
                    return {
                        "input_shape": X_test.shape if X_test is not None else None,
                        "metrics": {
                            "accuracy": None,
                            "precision": None,
                            "recall": None,
                            "f1_score": None,
                            "auc": None,
                            "balanced_accuracy": None,
                            "no_labels": True  # Flag to indicate no labels were available
                        },
                        "predictions": {
                            "y_pred": [p.flatten().tolist() for p in predictions] if predictions else []
                        },
                        "confusion_matrix": None,
                        "roc_curve": None,
                        "horizon_idx": horizon_idx,
                        "horizon_periods": horizons[horizon_idx] if horizon_idx < len(horizons) else None
                    }
                else:
                    logger.error("No valid input data (X_test) for prediction")
                    return {
                        "input_shape": None,
                        "metrics": {},
                        "predictions": {"y_pred": []},
                        "confusion_matrix": None,
                        "roc_curve": None,
                        "horizon_idx": horizon_idx,
                        "horizon_periods": None,
                        "error": "No valid input data"
                    }
            except Exception as pred_error:
                logger.error(f"Error generating predictions without labels: {str(pred_error)}")
                return {
                    "input_shape": X_test.shape if X_test is not None else None,
                    "metrics": {},
                    "predictions": {"y_pred": []},
                    "confusion_matrix": None,
                    "roc_curve": None,
                    "horizon_idx": horizon_idx,
                    "horizon_periods": horizons[horizon_idx] if horizon_idx < len(horizons) else None,
                    "error": str(pred_error)
                }
        
        # Get predictions
        try:
            predictions = self.model.model.predict(X_test)
            
            # Ensure predictions is a list for multi-output models
            if not isinstance(predictions, list):
                predictions = [predictions]
        except Exception as pred_error:
            logger.error(f"Error generating predictions: {str(pred_error)}")
            return {
                "input_shape": X_test.shape if X_test is not None else None,
                "metrics": {},
                "predictions": {"y_pred": []},
                "confusion_matrix": None,
                "roc_curve": None,
                "horizon_idx": horizon_idx,
                "horizon_periods": horizons[horizon_idx] if horizon_idx < len(horizons) else None,
                "error": str(pred_error)
            }
        
        # Get the target output for the specified horizon
        try:
            if horizon_idx >= len(y_test):
                logger.warning(f"horizon_idx {horizon_idx} exceeds available horizons {len(y_test)}. Using horizon 0 instead.")
                horizon_idx = 0
                
            y_true = y_test[horizon_idx]
            y_pred_proba = predictions[horizon_idx]
            
            # Verify shapes match
            if y_true.shape[0] != y_pred_proba.shape[0]:
                logger.warning(f"Shape mismatch: y_true {y_true.shape}, y_pred_proba {y_pred_proba.shape}")
                # Try to match lengths
                min_len = min(y_true.shape[0], y_pred_proba.shape[0])
                y_true = y_true[:min_len]
                y_pred_proba = y_pred_proba[:min_len]
                
            # Convert probabilities to binary predictions
            y_pred = (y_pred_proba > 0.5).astype(int)
        except Exception as e:
            logger.error(f"Error preparing data for metrics calculation: {str(e)}")
            return {
                "input_shape": X_test.shape,
                "metrics": {
                    "accuracy": None,
                    "precision": None,
                    "recall": None,
                    "f1_score": None,
                    "auc": None,
                    "balanced_accuracy": None,
                    "error": str(e)
                },
                "predictions": {
                    "y_pred": [p.flatten().tolist() for p in predictions] if predictions else []
                },
                "confusion_matrix": None,
                "roc_curve": None,
                "horizon_idx": horizon_idx,
                "horizon_periods": horizons[horizon_idx] if horizon_idx < len(horizons) else None
            }
        
        # Calculate evaluation metrics with error handling
        metrics = {}
        try:
            # Initialize with None values in case calculation fails
            metrics = {
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1_score": None,
                "auc": None,
                "balanced_accuracy": None
            }
            
            # Calculate basic metrics
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
            metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
            metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
            metrics["f1_score"] = float(f1_score(y_true, y_pred, zero_division=0))
        except Exception as metric_error:
            logger.error(f"Error calculating basic metrics: {str(metric_error)}")
            # Keep the None values in the metrics dictionary
        
        # Calculate ROC and AUC with separate error handling
        cm = None
        fpr, tpr = [], []
        try:
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
        except Exception as roc_error:
            logger.error(f"Error calculating ROC/AUC or confusion matrix: {str(roc_error)}")
            # ROC/AUC metrics will remain None
        
        # Ensure all metrics are JSON serializable
        for key, value in metrics.items():
            if value is not None and not isinstance(value, (int, float, str, bool, list, dict)):
                try:
                    metrics[key] = float(value)
                except:
                    metrics[key] = None
                    logger.warning(f"Could not convert metric {key} to float, setting to None")
        
        # Group all results - ensure consistent structure regardless of errors
        results = {
            "metrics": metrics,
            "predictions": {
                "y_true": y_true.tolist() if y_true is not None else [],
                "y_pred_proba": y_pred_proba.flatten().tolist() if y_pred_proba is not None else [],
                "y_pred": y_pred.flatten().tolist() if y_pred is not None else []
            },
            "confusion_matrix": cm.tolist() if cm is not None else None,
            "roc_curve": {
                "fpr": fpr.tolist() if len(fpr) > 0 else [],
                "tpr": tpr.tolist() if len(tpr) > 0 else []
            },
            "horizon_idx": horizon_idx,
            "horizon_periods": horizons[horizon_idx] if horizon_idx < len(horizons) else None,
            "input_shape": X_test.shape
        }
        
        # Log a summary of available metrics
        available_metrics = [k for k, v in metrics.items() if v is not None]
        missing_metrics = [k for k, v in metrics.items() if v is None]
        
        if available_metrics:
            logger.info(f"Successfully calculated metrics: {', '.join(available_metrics)}")
        if missing_metrics:
            logger.warning(f"Failed to calculate metrics: {', '.join(missing_metrics)}")
        
        return results
    
    def backtest_model(self, market_data: pd.DataFrame, initial_capital: float = 1000.0, 
                      position_size_pct: float = 100.0, use_stop_loss: bool = True,
                      stop_loss_pct: float = 2.0, take_profit_pct: float = 4.0,
                      use_trailing_stop: bool = True, trailing_activation_pct: float = 5.0,
                      trailing_step_pct: float = 2.0, fee_rate: float = 0.1, slippage_pct: float = 0.05) -> Dict:
        """
        Backtest the model on historical market data
        
        Args:
            market_data: DataFrame with OHLCV data
            initial_capital: Initial capital for backtesting
            position_size_pct: Position size as percentage of capital
            use_stop_loss: Whether to use stop loss
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            use_trailing_stop: Whether to use trailing stop
            trailing_activation_pct: Trailing stop activation percentage
            trailing_step_pct: Trailing stop step percentage
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
        trailing_activation_pct = trailing_activation_pct / 100
        trailing_step_pct = trailing_step_pct / 100
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
                
                # Check take profit only if trailing stop is not enabled
                elif not use_trailing_stop and current_price >= position['take_profit']:
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
                
                # Handle trailing stop logic
                elif use_trailing_stop and position is not None:
                    # Initialize trailing stop parameters if not set
                    if 'trailing_activated' not in position:
                        position['trailing_activated'] = False
                        position['highest_price'] = position['entry_price']
                    
                    # Calculate current profit percentage
                    current_profit_pct = (current_price - position['entry_price']) / position['entry_price'] * 100
                    
                    # Check if we should activate trailing stop
                    if not position['trailing_activated'] and current_profit_pct >= trailing_activation_pct:
                        # Activate trailing stop and set initial stop level
                        position['trailing_activated'] = True
                        # Lock in some profit - stop at half way between entry and current price
                        initial_stop_price = position['entry_price'] + (current_price - position['entry_price']) * 0.5
                        position['stop_loss'] = max(position['stop_loss'], initial_stop_price)
                        position['highest_price'] = current_price
                    
                    # Update trailing stop if activated and price moves higher
                    elif position['trailing_activated'] and current_price > position['highest_price']:
                        position['highest_price'] = current_price
                        # Move stop loss up based on trailing step percentage
                        new_stop_price = current_price * (1 - trailing_step_pct / 100)
                        position['stop_loss'] = max(position['stop_loss'], new_stop_price)
                
                # Get model prediction for early exit - check less frequently to improve performance
                elif i % 5 == 0:  # Check every 5 candles
                    try:
                        # Get a window of data up to current point
                        data_window = data.iloc[:i+1]
                        normalized_window = normalized_data.iloc[:i+1]
                        
                        # Create sequence for prediction
                        # FIX: Handle return value properly without unpacking
                        result = self.feature_engineering.create_multi_horizon_data(
                            normalized_window,
                            sequence_length=sequence_length,
                            horizons=horizons,
                            is_training=False
                        )
                        
                        # Check if result is a tuple (X, y) or just X
                        if isinstance(result, tuple) and len(result) == 2:
                            X = result[0]  # Extract X from tuple
                        else:
                            X = result  # Result is already X
                        
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
                    # FIX: Handle return value properly without unpacking
                    result = self.feature_engineering.create_multi_horizon_data(
                        normalized_window,
                        sequence_length=sequence_length,
                        horizons=horizons,
                        is_training=False
                    )
                    
                    # Check if result is a tuple (X, y) or just X
                    if isinstance(result, tuple) and len(result) == 2:
                        X = result[0]  # Extract X from tuple
                    else:
                        X = result  # Result is already X
                    
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
        
        # Create sequences - handle the consistent return value
        result = self.feature_engineering.create_multi_horizon_data(
            normalized_data,
            sequence_length=sequence_length,
            horizons=horizons,
            is_training=False
        )
        
        # Since feature_engineering now always returns a tuple
        X, y = result
        
        # If y is None, create dummy values for visualization
        if y is None:
            # Create dummy labels for visualization purposes only
            y = []
            for _ in horizons:
                # Create an array of zeros with the same length as X
                y.append(np.zeros(len(X)))
        
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

    def set_model(self, model):
        """Set the model to use for validation"""
        self.model = model
        logger.info(f"Model set: {type(model).__name__}")

# Add this new main function and execution code at the end of the file
def main():
    """
    Main function for command-line execution
    Allows direct validation of a model from the command line
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate a trained model on test data")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the market data CSV file")
    parser.add_argument("--report_path", type=str, default="reports", help="Directory to save reports")
    parser.add_argument("--model_type", type=str, default="lstm", choices=["lstm", "enhanced_lstm", "transformer"], 
                        help="Type of model to load")
    parser.add_argument("--horizon_idx", type=int, default=0, help="Prediction horizon index to evaluate")
    parser.add_argument("--backtest", action="store_true", help="Run backtesting simulation")
    parser.add_argument("--initial_capital", type=float, default=1000.0, help="Initial capital for backtesting")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.report_path, exist_ok=True)
    
    logger.info(f"Loading model from {args.model_path}")
    
    # Load the model based on model_type
    try:
        if args.model_type == "lstm":
            from ai.models.lstm_model import LSTMModel
            model = LSTMModel()
            model.load(args.model_path)
        elif args.model_type == "enhanced_lstm":
            from ai.models.lstm_model import EnhancedLSTMModel
            model = EnhancedLSTMModel()
            model.load(args.model_path)
        elif args.model_type == "transformer":
            from ai.models.transformer_model import TransformerModel
            model = TransformerModel()
            model.load(args.model_path)
        else:
            logger.error(f"Unsupported model type: {args.model_type}")
            return
            
        logger.info(f"Model loaded successfully: {args.model_type}")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return
    
    # Load market data
    logger.info(f"Loading data from {args.data_path}")
    try:
        # Check if the file exists
        if not os.path.exists(args.data_path):
            logger.warning(f"Data file not found at specified path: {args.data_path}")
            
            # Try alternative directories
            alt_paths = [
                os.path.join(DATA_DIR, "market_data", os.path.basename(args.data_path)),
                os.path.join(DATA_DIR, "market_data", f"{args.model_path.split('_')[-2]}_{args.model_path.split('_')[-1].split('.')[0]}.csv")
            ]
            
            logger.info(f"Looking for data file in alternative locations...")
            found = False
            
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    logger.info(f"Found data file at alternative location: {alt_path}")
                    args.data_path = alt_path
                    found = True
                    break
            
            # If not found in specific locations, look for any matching file by pattern
            if not found:
                symbol_timeframe = os.path.basename(args.data_path).split('.')[0]  # Get BTCUSDT_15m part
                market_data_dir = os.path.join(DATA_DIR, "market_data")
                
                if os.path.exists(market_data_dir):
                    matching_files = [f for f in os.listdir(market_data_dir) 
                                     if f.endswith('.csv') and symbol_timeframe in f]
                    
                    if matching_files:
                        args.data_path = os.path.join(market_data_dir, matching_files[0])
                        logger.info(f"Found matching data file: {args.data_path}")
                        found = True
                    else:
                        # Show all available files to help the user
                        available_files = [f for f in os.listdir(market_data_dir) if f.endswith('.csv')]
                        logger.error(f"No matching data files found. Available files in {market_data_dir}:")
                        for file in available_files:
                            logger.error(f"  - {file}")
            
            if not found:
                logger.error(f"Could not find a suitable data file")
                return
        
        # Load the CSV data
        data = pd.read_csv(args.data_path)
        
        # If 'timestamp' column exists, convert to datetime and set as index
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data.set_index('timestamp', inplace=True)
        
        logger.info(f"Data loaded successfully: {len(data)} rows")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return
    
    # Create validator
    feature_engineering = FeatureEngineering()
    validator = ModelValidator(model, feature_engineering)
    
    # Set output directory
    validator.output_dir = args.report_path
    
    # Run evaluation
    logger.info(f"Evaluating model on test data (horizon_idx={args.horizon_idx})...")
    eval_results = validator.evaluate_on_test_set(data, horizon_idx=args.horizon_idx)
    
    # Print key metrics
    if eval_results["metrics"]:
        logger.info("----- Model Evaluation Results -----")
        for metric, value in eval_results["metrics"].items():
            logger.info(f"{metric}: {value:.4f}")
    else:
        logger.warning("No metrics available in evaluation results")
    
    # Run backtest if requested
    if args.backtest:
        logger.info(f"Running backtest with initial capital: ${args.initial_capital}...")
        backtest_results = validator.backtest_model(
            data,
            initial_capital=args.initial_capital
        )
        
        logger.info("----- Backtest Results -----")
        logger.info(f"Total Return: {backtest_results['total_return_pct']:.2f}%")
        logger.info(f"Win Rate: {backtest_results['win_rate']:.2f}%")
        logger.info(f"Total Trades: {backtest_results['total_trades']}")
        logger.info(f"Max Drawdown: {backtest_results['max_drawdown_pct']:.2f}%")
        
        # Compare with buy & hold
        comparison = validator.compare_with_baseline(data, initial_capital=args.initial_capital)
        
        logger.info("----- Comparison with Buy & Hold -----")
        logger.info(f"Model Return: {comparison['model_return_pct']:.2f}%")
        logger.info(f"Buy & Hold Return: {comparison['buy_and_hold_return_pct']:.2f}%")
        logger.info(f"Outperformance: {comparison['outperformance_pct']:.2f}%")
    
    # Generate visualizations if requested
    if args.visualize:
        logger.info("Generating visualizations...")
        
        # Visualize predictions
        pred_vis_path = os.path.join(args.report_path, f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        validator.visualize_predictions(data, horizon_idx=args.horizon_idx, save_path=pred_vis_path)
        logger.info(f"Prediction visualization saved to {pred_vis_path}")
        
        # Visualize backtest performance if backtest was run
        if args.backtest:
            perf_vis_path = os.path.join(args.report_path, f"backtest_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            validator.visualize_performance(backtest_results, perf_vis_path)
            logger.info(f"Performance visualization saved to {perf_vis_path}")
    
    # Save results to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = {
        "evaluation": eval_results,
        "backtest": backtest_results if args.backtest else None,
        "comparison": comparison if args.backtest else None,
        "timestamp": timestamp
    }
    
    result_path = os.path.join(args.report_path, f"validation_results_{timestamp}.json")
    
    # Save to file
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {result_path}")
    
    logger.info("Model validation completed successfully")

if __name__ == "__main__":
    main()