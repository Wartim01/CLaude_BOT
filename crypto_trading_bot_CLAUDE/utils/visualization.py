"""
Visualization utilities for model evaluation and trading signals
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

from config.config import VISUALIZATION_DIR

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("talk")

def plot_model_performance(
    history: Dict, 
    metrics: List[str] = None, 
    save_path: Optional[str] = None,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None
) -> None:
    """
    Plot training history and key metrics from model training
    
    Args:
        history: Dictionary of training history (from model.fit())
        metrics: List of metrics to plot (defaults to loss and accuracy)
        save_path: Path to save the plot (if None, displays the plot)
        symbol: Trading pair symbol for the title
        timeframe: Timeframe for the title
    """
    if metrics is None:
        metrics = ['loss', 'accuracy']
    
    # Create a figure with subplots for each metric
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics), sharex=True)
    
    if n_metrics == 1:
        axes = [axes]  # Make axes iterable when there's only one subplot
    
    title_parts = []
    if symbol:
        title_parts.append(symbol)
    if timeframe:
        title_parts.append(timeframe)
    
    title = "Model Performance" if not title_parts else f"Model Performance - {' '.join(title_parts)}"
    fig.suptitle(title, fontsize=16)
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Plot training metric
        if metric in history:
            ax.plot(history[metric], label=f'Training {metric}', linewidth=2)
        
        # Plot validation metric if available
        val_metric = f'val_{metric}'
        if val_metric in history:
            ax.plot(history[val_metric], label=f'Validation {metric}', linewidth=2, linestyle='--')
        
        ax.set_title(f'{metric.capitalize()} over Epochs')
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        
        # Only add xlabel to the bottom subplot
        if i == n_metrics - 1:
            ax.set_xlabel('Epoch')
    
    # Tight layout for better spacing
    plt.tight_layout()
    fig.subplots_adjust(top=0.94)  # Adjust for the suptitle
    
    # Save or show the plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_trading_signals(
    df: pd.DataFrame,
    price_col: str = 'close',
    prediction_col: str = 'prediction',
    threshold: float = 0.5,
    save_path: Optional[str] = None,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    window_size: int = 200
) -> None:
    """
    Plot price chart with buy/sell signals and predictions
    
    Args:
        df: DataFrame with price and prediction data
        price_col: Column name for price data
        prediction_col: Column name for model predictions
        threshold: Threshold for buy signals (e.g., prediction > 0.5)
        save_path: Path to save the plot (if None, displays the plot)
        symbol: Trading pair symbol for the title
        timeframe: Timeframe for the title
        window_size: Number of candles to display (prevents overcrowding)
    """
    # Make a copy to avoid modifying the original dataframe
    plot_df = df.copy()
    
    # Make sure DataFrame has a datetime index
    if not isinstance(plot_df.index, pd.DatetimeIndex):
        if 'timestamp' in plot_df.columns:
            plot_df['timestamp'] = pd.to_datetime(plot_df['timestamp'])
            plot_df.set_index('timestamp', inplace=True)
        else:
            raise ValueError("DataFrame must have a datetime index or 'timestamp' column")
    
    # Calculate buy/sell signals based on predictions
    if prediction_col in plot_df.columns:
        plot_df['signal'] = 0
        plot_df.loc[plot_df[prediction_col] > threshold, 'signal'] = 1  # Buy signal
        plot_df.loc[plot_df[prediction_col] < threshold, 'signal'] = -1  # Sell signal
        
        # Generate signal change for plotting entry/exit points
        plot_df['signal_change'] = plot_df['signal'].diff().fillna(0)
        
    # Take the last window_size rows if df is too large
    if len(plot_df) > window_size:
        plot_df = plot_df.iloc[-window_size:]
    
    # Plot price chart with signals
    fig, ax1 = plt.subplots(figsize=(15, 8))
    
    # Plot price
    ax1.plot(plot_df.index, plot_df[price_col], label=price_col.capitalize(), color='royalblue', linewidth=1.5)
    ax1.set_ylabel('Price', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    
    # Format x-axis with dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    if len(plot_df) > 30:  # Rotate date labels for readability when many points
        plt.xticks(rotation=45)
    
    # Plot buy/sell signals if prediction column exists
    if prediction_col in plot_df.columns:
        # Plot entry points (signal changes from 0 or -1 to 1)
        buy_points = plot_df[plot_df['signal_change'] == 1]
        ax1.scatter(buy_points.index, buy_points[price_col], 
                    color='green', s=100, marker='^', label='Buy Signal')
        
        # Plot exit points (signal changes from 0 or 1 to -2)
        sell_points = plot_df[plot_df['signal_change'] == -2]
        ax1.scatter(sell_points.index, sell_points[price_col], 
                    color='red', s=100, marker='v', label='Sell Signal')
        
        # Plot prediction probability on secondary y-axis
        ax2 = ax1.twinx()
        ax2.plot(plot_df.index, plot_df[prediction_col], color='purple', 
                 alpha=0.5, linewidth=1, label='Prediction Probability')
        ax2.set_ylabel('Prediction Probability', color='purple')
        ax2.tick_params(axis='y', labelcolor='purple')
        ax2.axhline(y=threshold, color='gray', linestyle='--', alpha=0.7, 
                    label=f'Threshold ({threshold})')
        
        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    else:
        ax1.legend(loc='upper left')
    
    # Set title
    title_parts = []
    if symbol:
        title_parts.append(symbol)
    if timeframe:
        title_parts.append(timeframe)
    
    title = "Price Chart with Signals" if not title_parts else f"Price Chart with Signals - {' '.join(title_parts)}"
    plt.title(title)
    
    # Add grid, tight layout
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save or display
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_backtest_results(
    backtest_df: pd.DataFrame,
    save_path: Optional[str] = None,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None
) -> None:
    """
    Plot backtest results including equity curve, drawdowns, and trade metrics
    
    Args:
        backtest_df: DataFrame with backtest results (should have 'equity', 'returns', etc.)
        save_path: Path to save the plot (if None, displays the plot)
        symbol: Trading pair symbol for the title
        timeframe: Timeframe for the title
    """
    # Check if we have the necessary columns
    required_columns = ['equity', 'returns']
    missing_columns = [col for col in required_columns if col not in backtest_df.columns]
    if missing_columns:
        raise ValueError(f"Backtest DataFrame is missing required columns: {missing_columns}")
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(3, 1, figsize=(15, 15), gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # Plot equity curve
    backtest_df['equity'].plot(ax=axes[0], color='blue', linewidth=2)
    backtest_df['equity'].rolling(window=20).mean().plot(ax=axes[0], color='red', 
                                                        linestyle='--', linewidth=1)
    
    # If benchmark exists, plot it too
    if 'benchmark_equity' in backtest_df.columns:
        backtest_df['benchmark_equity'].plot(ax=axes[0], color='gray', 
                                            linestyle='--', linewidth=1.5, label='Benchmark')
    
    axes[0].set_title('Equity Curve')
    axes[0].set_ylabel('Portfolio Value')
    axes[0].legend(['Equity', '20-period MA' + (', Benchmark' if 'benchmark_equity' in backtest_df.columns else '')])
    axes[0].grid(True)
    
    # Plot returns
    backtest_df['returns'].plot(ax=axes[1], kind='bar', color='green', alpha=0.5)
    axes[1].set_title('Trade Returns')
    axes[1].set_ylabel('Return %')
    axes[1].grid(True)
    
    # Calculate and plot drawdowns
    if 'drawdown' not in backtest_df.columns:
        backtest_df['drawdown'] = backtest_df['equity'] / backtest_df['equity'].cummax() - 1
    
    backtest_df['drawdown'].plot(ax=axes[2], color='red', linewidth=1.5)
    axes[2].fill_between(backtest_df.index, 0, backtest_df['drawdown'], color='red', alpha=0.3)
    axes[2].set_title('Drawdowns')
    axes[2].set_ylabel('Drawdown %')
    axes[2].grid(True)
    
    # Set main title
    title_parts = []
    if symbol:
        title_parts.append(symbol)
    if timeframe:
        title_parts.append(timeframe)
    
    title = "Backtest Results" if not title_parts else f"Backtest Results - {' '.join(title_parts)}"
    fig.suptitle(title, fontsize=16)
    
    # Tight layout
    plt.tight_layout()
    fig.subplots_adjust(top=0.95)  # Adjust for the suptitle
    
    # Save or show
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    normalize: bool = False,
    save_path: Optional[str] = None,
    title: Optional[str] = None
) -> None:
    """
    Plot confusion matrix for binary classification
    
    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        normalize: Whether to normalize the confusion matrix
        save_path: Path to save the plot (if None, displays the plot)
        title: Title for the plot
    """
    from sklearn.metrics import confusion_matrix
    
    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm, 2)
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=['Down', 'Up'],
                yticklabels=['Down', 'Up'])
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title or 'Confusion Matrix')
    
    # Save or show
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_feature_importance(
    feature_names: List[str],
    importances: List[float],
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    top_n: int = 20
) -> None:
    """
    Plot feature importance
    
    Args:
        feature_names: List of feature names
        importances: List of importance scores
        save_path: Path to save the plot (if None, displays the plot)
        title: Title for the plot
        top_n: Number of top features to show
    """
    # Create DataFrame for sorting
    feature_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance and take top N
    feature_df = feature_df.sort_values('Importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_df, palette='viridis')
    
    plt.title(title or f'Top {top_n} Feature Importances')
    plt.tight_layout()
    
    # Save or show
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_prediction_vs_actual(
    df: pd.DataFrame,
    price_col: str = 'close',
    prediction_col: str = 'prediction',
    actual_direction_col: str = 'next_direction',
    horizon: int = 1,
    save_path: Optional[str] = None,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    window_size: int = 200
) -> None:
    """
    Plot predicted vs actual price direction
    
    Args:
        df: DataFrame with price, prediction, and actual direction data
        price_col: Column name for price data
        prediction_col: Column name for model predictions
        actual_direction_col: Column name for actual price direction
        horizon: Prediction horizon in periods
        save_path: Path to save the plot (if None, displays the plot)
        symbol: Trading pair symbol for the title
        timeframe: Timeframe for the title
        window_size: Number of candles to display
    """
    # Make a copy to avoid modifying the original dataframe
    plot_df = df.copy()
    
    # Make sure DataFrame has a datetime index
    if not isinstance(plot_df.index, pd.DatetimeIndex):
        if 'timestamp' in plot_df.columns:
            plot_df['timestamp'] = pd.to_datetime(plot_df['timestamp'])
            plot_df.set_index('timestamp', inplace=True)
    
    # Take the last window_size rows if df is too large
    if len(plot_df) > window_size:
        plot_df = plot_df.iloc[-window_size:]
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(15, 8))
    
    # Plot price
    ax1.plot(plot_df.index, plot_df[price_col], color='black', linewidth=1, label='Price')
    ax1.set_ylabel('Price', color='black')
    
    # Add secondary axis for predictions
    ax2 = ax1.twinx()
    
    # Plot predicted direction
    if prediction_col in plot_df.columns:
        # Convert to binary for clearer visualization
        binary_pred = (plot_df[prediction_col] > 0.5).astype(int)
        ax2.scatter(plot_df.index, binary_pred + 1.1, color='blue', 
                   marker='o', s=30, alpha=0.6, label='Predicted Direction')
    
    # Plot actual direction (shifted to avoid overlap)
    if actual_direction_col in plot_df.columns:
        ax2.scatter(plot_df.index, plot_df[actual_direction_col] + 1, color='green',
                   marker='x', s=30, alpha=0.6, label='Actual Direction')
    
    # Set up the secondary y-axis
    ax2.set_ylim(0.5, 2.5)
    ax2.set_yticks([1, 2])
    ax2.set_yticklabels(['Down', 'Up'])
    ax2.set_ylabel('Price Direction')
    
    # Format x-axis with dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    if len(plot_df) > 30:
        plt.xticks(rotation=45)
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Set title
    title_parts = []
    if symbol:
        title_parts.append(symbol)
    if timeframe:
        title_parts.append(timeframe)
    
    horizon_text = f"{horizon}-period" if horizon > 1 else "Next period"
    base_title = f"{horizon_text} Direction Prediction vs Actual"
    title = base_title if not title_parts else f"{base_title} - {' '.join(title_parts)}"
    plt.title(title)
    
    # Add grid, tight layout
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save or display
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_roc_curve(
    y_true: Union[List, np.ndarray],
    y_score: Union[List, np.ndarray],
    save_path: Optional[str] = None,
    title: Optional[str] = None
) -> Tuple[float, float]:
    """
    Plot ROC curve and calculate AUC
    
    Args:
        y_true: True binary labels
        y_score: Target scores (probabilities)
        save_path: Path to save the plot (if None, displays the plot)
        title: Title for the plot
        
    Returns:
        Tuple of (AUC score, optimal threshold)
    """
    from sklearn.metrics import roc_curve, auc
    
    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Calculate Youden's J statistic to find optimal threshold
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Mark the optimal threshold
    plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', 
             markersize=10, label=f'Optimal threshold = {optimal_threshold:.3f}')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title or 'Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    # Save or show
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    
    return roc_auc, optimal_threshold

def create_evaluation_plots(model, X_val, y_val, predictions, symbol, timeframe, model_path):
    """
    Crée des visualisations des performances du modèle
    
    Args:
        model: Modèle LSTM entraîné
        X_val: Données de validation (features)
        y_val: Données de validation (cibles)
        predictions: Prédictions du modèle
        symbol: Symbole de la paire de trading
        timeframe: Intervalle de temps
        model_path: Chemin du modèle sauvegardé
    """
    # Créer le répertoire pour les visualisations
    viz_dir = os.path.join(os.path.dirname(model_path), "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    
    # 1. Direction prediction accuracy - Simplified for single horizon
    plt.figure(figsize=(10, 6))
    
    # Pour un seul horizon, pas besoin de boucle
    y_true = y_val.flatten() if isinstance(y_val, np.ndarray) else y_val[0].flatten()
    y_pred = (predictions.flatten() > 0.5).astype(int) if isinstance(predictions, np.ndarray) else (predictions[0].flatten() > 0.5).astype(int)
    
    accuracy = np.mean(y_true == y_pred) * 100
    
    # Plot simple bar chart with the 1h accuracy
    horizons = ["1h"]
    accuracies = [accuracy]
    
    plt.bar(horizons, accuracies, color='skyblue')
    plt.axhline(y=50, color='r', linestyle='--', alpha=0.5)  # 50% line (random)
    plt.ylim([0, 100])
    plt.title(f'Précision de Prédiction 1h - {symbol} {timeframe}')
    plt.ylabel('Précision (%)')
    plt.xlabel('Horizon')
    
    for i, v in enumerate(accuracies):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center')
    
    plt.savefig(os.path.join(viz_dir, f"{symbol}_{timeframe}_direction_accuracy.png"))
    plt.close()
    
    # 2. Confusion matrix for the single horizon
    # ...existing code for confusion matrix...
