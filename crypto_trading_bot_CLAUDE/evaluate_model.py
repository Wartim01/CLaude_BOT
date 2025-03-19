#!/usr/bin/env python
"""
Script for evaluating trained models on historical or new market data
Provides performance metrics, visualization, and backtesting capabilities
"""
import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from ai.models.lstm_model import LSTMModel, EnhancedLSTMModel
from ai.models.transformer_model import TransformerModel
from ai.models.feature_engineering import FeatureEngineering
from ai.models.model_validator import ModelValidator
from config.config import DATA_DIR
from utils.logger import setup_logger
from utils.visualization import plot_model_performance, plot_trading_signals

# Configure logger
logger = setup_logger("model_evaluation")

def load_model(model_path, model_type="lstm"):
    """
    Load the trained model from the specified path
    
    Args:
        model_path: Path to the model file
        model_type: Type of model ('lstm', 'enhanced_lstm', 'transformer')
        
    Returns:
        Loaded model
    """
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return None

    try:
        if model_type.lower() == "lstm":
            model = LSTMModel()
            model.load(model_path)
            logger.info(f"Loaded LSTM model from {model_path}")
            return model
        elif model_type.lower() == "enhanced_lstm":
            model = EnhancedLSTMModel()
            model.load(model_path)
            logger.info(f"Loaded Enhanced LSTM model from {model_path}")
            return model
        elif model_type.lower() == "transformer":
            model = TransformerModel()
            model.load(model_path)
            logger.info(f"Loaded Transformer model from {model_path}")
            return model
        else:
            logger.error(f"Unknown model type: {model_type}")
            return None
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

def load_data(data_path, symbol=None, timeframe=None, start_date=None, end_date=None):
    """
    Load market data from file
    
    Args:
        data_path: Path to the data file
        symbol: Symbol (e.g., 'BTCUSDT') if loading from directory
        timeframe: Timeframe (e.g., '1h') if loading from directory
        start_date: Start date filter (YYYY-MM-DD)
        end_date: End date filter (YYYY-MM-DD)
        
    Returns:
        DataFrame with market data
    """
    if os.path.isfile(data_path):
        # Load directly from specified file
        df = pd.read_csv(data_path)
    else:
        # Try to find file based on symbol and timeframe
        if symbol and timeframe:
            # Look for files matching pattern in data directory
            data_dir = os.path.join(DATA_DIR, "market_data")
            potential_files = [f for f in os.listdir(data_dir) if f.startswith(f"{symbol}_{timeframe}") and f.endswith(".csv")]
            
            if potential_files:
                data_path = os.path.join(data_dir, potential_files[0])
                df = pd.read_csv(data_path)
            else:
                logger.error(f"No data file found for {symbol}_{timeframe}")
                return None
        else:
            logger.error("No data path specified and no symbol/timeframe pair provided")
            return None
    
    # Ensure timestamp is properly formatted
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    elif 'date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'])
        df.set_index('timestamp', inplace=True)
        df.drop('date', axis=1, inplace=True, errors='ignore')
    
    # Apply date filters if provided
    if start_date:
        start_dt = pd.to_datetime(start_date)
        df = df[df.index >= start_dt]
    
    if end_date:
        end_dt = pd.to_datetime(end_date)
        df = df[df.index <= end_dt]
    
    logger.info(f"Loaded {len(df)} data points from {data_path}")
    return df

def evaluate_model(model, data, validator=None, backtest=False, capital=1000, plot=False, output_dir=None):
    """
    Evaluate model performance on the provided data
    
    Args:
        model: Trained model to evaluate
        data: Market data for evaluation
        validator: ModelValidator instance (creates one if None)
        backtest: Whether to run backtest simulation
        capital: Initial capital for backtest
        plot: Whether to generate plots
        output_dir: Directory for saving outputs
        
    Returns:
        Dictionary with evaluation results
    """
    # Create validator if not provided
    if validator is None:
        feature_engineering = FeatureEngineering()
        validator = ModelValidator(model, feature_engineering)
    
    # Set output directory
    if output_dir:
        validator.output_dir = output_dir
    
    # Run evaluation
    eval_results = validator.evaluate_on_test_set(data)
    
    # Print key metrics
    logger.info("----- Model Evaluation Results -----")
    logger.info(f"Accuracy: {eval_results['metrics']['accuracy']:.4f}")
    logger.info(f"F1 Score: {eval_results['metrics']['f1_score']:.4f}")
    logger.info(f"AUC: {eval_results['metrics']['auc']:.4f}")
    
    # Run backtest if requested
    backtest_results = None
    if backtest:
        backtest_results = validator.backtest_model(
            data,
            initial_capital=capital,
            position_size_pct=100.0,
            use_stop_loss=True,
            stop_loss_pct=2.0,
            take_profit_pct=4.0
        )
        
        logger.info("----- Backtest Results -----")
        logger.info(f"Total Return: {backtest_results['total_return_pct']:.2f}%")
        logger.info(f"Win Rate: {backtest_results['win_rate']:.2f}%")
        logger.info(f"Total Trades: {backtest_results['total_trades']}")
        logger.info(f"Max Drawdown: {backtest_results['max_drawdown_pct']:.2f}%")
    
    # Generate plots if requested
    if plot:
        if backtest_results:
            # Create visualization filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            vis_path = os.path.join(validator.output_dir, f"backtest_performance_{timestamp}.png")
            
            # Generate backtest performance visualization
            validator.visualize_performance(backtest_results, vis_path)
            logger.info(f"Backtest visualization saved to {vis_path}")
        
        # Create prediction visualization
        pred_vis_path = os.path.join(validator.output_dir, f"predictions_{timestamp}.png")
        validator.visualize_predictions(data, save_path=pred_vis_path)
        logger.info(f"Prediction visualization saved to {pred_vis_path}")
    
    # Compare with buy & hold
    comparison = validator.compare_with_baseline(data, initial_capital=capital)
    
    logger.info("----- Comparison with Buy & Hold -----")
    logger.info(f"Model Return: {comparison['model_return_pct']:.2f}%")
    logger.info(f"Buy & Hold Return: {comparison['buy_and_hold_return_pct']:.2f}%")
    logger.info(f"Outperformance: {comparison['outperformance_pct']:.2f}%")
    
    # Save results
    results = {
        "evaluation": eval_results,
        "backtest": backtest_results if backtest else None,
        "comparison": comparison
    }
    
    # Save results to file
    result_path = validator.save_results(results, "model_evaluation")
    logger.info(f"Full results saved to {result_path}")
    
    return results

def parse_args():
    parser = argparse.ArgumentParser(description="Évaluation d’un modèle ML sur un jeu de test")
    parser.add_argument("--model_path", type=str, required=True, help="Chemin du modèle entraîné")
    parser.add_argument("--test_data", type=str, required=True, help="Fichier CSV contenant les données de test")
    parser.add_argument("--output", type=str, default=None, help="Chemin pour sauvegarder le rapport JSON et visualisations")
    parser.add_argument("--visualize", action="store_true", help="Générer et sauvegarder des graphiques")
    return parser.parse_args()

def load_test_data(test_data_path):
    df = pd.read_csv(test_data_path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    # Assume the CSV has a column 'target' and features in all other columns
    X = df.drop(columns=['target']).values
    y = df['target'].values
    return X, y

def evaluate(model, X, y):
    # Convertir X en float32 pour être sûr que TensorFlow puisse le traiter
    X = X.astype(np.float32)
    predictions = model.model.predict(X)
    # Handle multi-output: assume single output for evaluation simplicity
    if isinstance(predictions, list):
        predictions = predictions[0]
    y_pred = (predictions.flatten() > 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1_score": f1_score(y, y_pred, zero_division=0),
        "classification_report": classification_report(y, y_pred, output_dict=True)
    }
    cm = confusion_matrix(y, y_pred)
    return metrics, cm, y_pred

def plot_confusion_matrix(cm, output_dir, symbol=""):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Matrice de Confusion {symbol}")
    plt.xlabel("Prédit")
    plt.ylabel("Vrai")
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    logger.info(f"Graphique de la matrice de confusion sauvegardé sous {cm_path}")

def main():
    args = parse_args()
    model = load_model(args.model_path)
    logger.info(f"Modèle chargé depuis {args.model_path}")
    
    # Remplacer le chargement des données via load_test_data par l'utilisation de ModelValidator
    from ai.models.feature_engineering import FeatureEngineering
    validator = ModelValidator(model, FeatureEngineering())
    import pandas as pd
    # Charger les données de test sous forme de DataFrame
    test_data = pd.read_csv(args.test_data, parse_dates=['timestamp'])
    test_data.set_index('timestamp', inplace=True)
    logger.info(f"Test data shape after index set: {test_data.shape}")
    
    # Process the features
    results = validator.evaluate_on_test_set(test_data)
    logger.info(f"Les séquences d'entrée utilisées dans l'évaluation ont {results.get('input_shape', 'N/A')} features.")
    metrics = results['metrics']
    cm = results['confusion_matrix']
    
    logger.info("=== RÉSULTATS DE L'ÉVALUATION ===")
    logger.info(f"Accuracy    : {metrics['accuracy']:.4f}")
    logger.info(f"Precision   : {metrics['precision']:.4f}")
    logger.info(f"Recall      : {metrics['recall']:.4f}")
    logger.info(f"F1 Score    : {metrics['f1_score']:.4f}")
    
    # Sauvegarder les métriques dans un fichier JSON
    output_dir = args.output or os.path.dirname(args.model_path)
    report_path = os.path.join(output_dir, f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Rapport d'évaluation sauvegardé sous {report_path}")
    
    # Générer visualisations si demandé
    if args.visualize:
        plot_confusion_matrix(cm, output_dir)
    
if __name__ == "__main__":
    main()

