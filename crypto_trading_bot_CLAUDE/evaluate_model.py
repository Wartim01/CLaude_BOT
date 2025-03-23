#!/usr/bin/env python
"""
Script for evaluating trained models on historical or new market data
Provides performance metrics, visualization, and backtesting capabilities
"""
import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Fix Unicode encoding issues on Windows
if sys.platform.startswith('win'):
    # Force UTF-8 encoding for stdout/stderr
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    # Set console code page to UTF-8
    os.system('chcp 65001')
    # Set environment variable for child processes
    os.environ["PYTHONIOENCODING"] = "utf-8"

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
    
    # Check for errors in evaluation
    if "error" in eval_results:
        logger.error(f"Error during evaluation: {eval_results['error']}")
    
    # Print key metrics - safely access metrics dictionary
    logger.info("----- Model Evaluation Results -----")
    
    # Ensure metrics dictionary exists and has expected structure
    if 'metrics' not in eval_results:
        eval_results['metrics'] = {}
        logger.warning("No metrics available in evaluation results")
    
    # Log available metrics with safe access
    metrics = eval_results['metrics']
    if metrics:
        # Check if no_labels flag is set
        if metrics.get('no_labels', False):
            logger.warning("No labels were available for evaluation - metrics could not be calculated")
        else:
            # Log metrics that are present, with NA for missing ones
            logger.info(f"Accuracy: {metrics.get('accuracy', 'NA')}")
            logger.info(f"F1 Score: {metrics.get('f1_score', 'NA')}")
            logger.info(f"AUC: {metrics.get('auc', 'NA')}")
            
            # If basic metrics are all missing, but no explicit error, log a warning
            if all(metrics.get(m) is None for m in ['accuracy', 'f1_score', 'precision', 'recall']):
                logger.warning("All basic performance metrics are missing or could not be calculated")
    else:
        logger.warning("No metrics available in evaluation results. This may indicate an issue with the model or data.")
        logger.debug(f"Available keys in eval_results: {eval_results.keys()}")
    
    # Run backtest if requested - only if we have valid input data
    backtest_results = None
    if backtest:
        if eval_results.get('input_shape') is not None:
            try:
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
            except Exception as e:
                logger.error(f"Error during backtesting: {str(e)}")
                backtest_results = None
        else:
            logger.warning("Skipping backtest - no valid input data available from evaluation")
    
    # Generate plots if requested - only if we have valid data
    if plot:
        try:
            if backtest_results:
                # Create visualization filename
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                vis_path = os.path.join(validator.output_dir, f"backtest_performance_{timestamp}.png")
                
                # Generate backtest performance visualization
                validator.visualize_performance(backtest_results, vis_path)
                logger.info(f"Backtest visualization saved to {vis_path}")
            
            # Only create prediction visualization if we have valid predictions
            if 'predictions' in eval_results and eval_results['predictions'].get('y_pred'):
                pred_vis_path = os.path.join(validator.output_dir, f"predictions_{timestamp}.png")
                validator.visualize_predictions(data, save_path=pred_vis_path)
                logger.info(f"Prediction visualization saved to {pred_vis_path}")
            else:
                logger.warning("Skipping prediction visualization - no valid predictions available")
        except Exception as plot_error:
            logger.error(f"Error generating visualizations: {str(plot_error)}")
    
    # Compare with buy & hold - only if backtest was successful
    comparison = None
    if backtest_results:
        try:
            comparison = validator.compare_with_baseline(data, initial_capital=capital)
            
            logger.info("----- Comparison with Buy & Hold -----")
            logger.info(f"Model Return: {comparison['model_return_pct']:.2f}%")
            logger.info(f"Buy & Hold Return: {comparison['buy_and_hold_return_pct']:.2f}%")
            logger.info(f"Outperformance: {comparison['outperformance_pct']:.2f}%")
        except Exception as comp_error:
            logger.error(f"Error comparing with baseline: {str(comp_error)}")
            comparison = None
    
    # Save results
    results = {
        "evaluation": eval_results,
        "backtest": backtest_results,
        "comparison": comparison
    }
    
    # Save results to file
    try:
        result_path = validator.save_results(results, "model_evaluation")
        logger.info(f"Full results saved to {result_path}")
    except Exception as save_error:
        logger.error(f"Error saving results: {str(save_error)}")
    
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

def evaluate(args):
    """
    Point d'entrée pour l'évaluation du modèle depuis la ligne de commande
    
    Args:
        args: Arguments de ligne de commande
    """
    try:
        # Charger les données
        data = load_data(
            args.data_path, 
            args.symbol, 
            args.timeframe, 
            args.start_date, 
            args.end_date
        )
        
        if data is None or data.empty:
            logger.error("Données vides ou non disponibles")
            return
        
        logger.info(f"Data shape: {data.shape}")
        logger.info(f"Data index type: {type(data.index)}")
        logger.info(f"Data date range: {data.index.min()} to {data.index.max()}")
        
        # Chercher un modèle existant
        model_dir = os.path.join(DATA_DIR, "models")
        model_path = os.path.join(model_dir, f"lstm_{args.symbol}_{args.timeframe}.keras")
        
        if not os.path.exists(model_path):
            # Chercher d'autres formats de fichier (.h5)
            alt_model_path = os.path.join(model_dir, f"lstm_{args.symbol}_{args.timeframe}.h5")
            if os.path.exists(alt_model_path):
                model_path = alt_model_path
            else:
                # Essayer de trouver un modèle correspondant dans le répertoire
                model_files = [f for f in os.listdir(model_dir) 
                              if os.path.isfile(os.path.join(model_dir, f)) and 
                              args.symbol in f and 
                              (f.endswith('.keras') or f.endswith('.h5'))]
                        
                if model_files:
                    model_path = os.path.join(model_dir, model_files[0])
                else:
                    logger.error(f"No model found for {args.symbol}_{args.timeframe}")
                    return
        
        logger.info(f"Using model from: {model_path}")
        
        # Charger le modèle
        model = load_model(model_path)
        if model is None:
            logger.error("Failed to load model")
            return
        
        # Créer le validateur
        logger.info("Creating model validator...")
        feature_engineering = FeatureEngineering()
        validator = ModelValidator(model, feature_engineering)
        
        # Exécuter l'évaluation
        output_dir = os.path.join(DATA_DIR, "evaluation", f"{args.symbol}_{args.timeframe}")
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Starting model evaluation, results will be saved to: {output_dir}")
        results = evaluate_model(
            model, 
            data, 
            validator=validator, 
            backtest=True, 
            plot=True,
            output_dir=output_dir
        )
        
        logger.info(f"Evaluation completed and results saved to {output_dir}")
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LSTM Model")
    subparsers = parser.add_subparsers(dest="command")
    
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    eval_parser.add_argument("--symbol", required=True, help="Symbol to evaluate (e.g., BTCUSDT)")
    eval_parser.add_argument("--timeframe", required=True, help="Timeframe to evaluate (e.g., 15m, 1h)")
    eval_parser.add_argument("--start-date", required=True, help="Start date for evaluation (YYYY-MM-DD)")
    eval_parser.add_argument("--end-date", required=True, help="End date for evaluation (YYYY-MM-DD)")
    eval_parser.add_argument("--data_path", required=True, help="Path to market data directory or CSV file")
    eval_parser.add_argument("--model_path", type=str, default=None, 
                          help="Specific model path (optional, will look for model based on symbol/timeframe if not provided)")
    eval_parser.add_argument("--visualize", action="store_true", default=True,
                          help="Generate visualizations")
    
    args = parser.parse_args()
    
    if args.command == "evaluate":
        try:
            # Validate data_path - check if it contains spaces and wasn't quoted
            if " " in args.data_path and not os.path.exists(args.data_path):
                print(f"Error: Path contains spaces and was not found: {args.data_path}")
                print("If your path contains spaces, please enclose it in quotes:")
                print('Example: --data_path "C:\\Users\\timot\\OneDrive\\Bureau\\BOT TRADING BIG 2025\\crypto_trading_bot_CLAUDE\\data\\market_data"')
                sys.exit(1)
            
            # Validate dates - make sure they're in the correct format
            try:
                pd.to_datetime(args.start_date)
                pd.to_datetime(args.end_date)
            except ValueError:
                print("Error: Invalid date format. Use YYYY-MM-DD format.")
                sys.exit(1)
            
            evaluate(args)
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            import traceback
            traceback.print_exc()
            print(f"\nError: {str(e)}")
            print("\nCorrect command format:")
            print("  python evaluate_model.py evaluate --symbol BTCUSDT --timeframe 15m --start-date 2022-01-01 --end-date 2024-12-31 --data_path data/market_data")
            print("\nFor Windows paths with spaces, use quotes:")
            print('  python evaluate_model.py evaluate --symbol BTCUSDT --timeframe 15m --start-date 2022-01-01 --end-date 2024-12-31 --data_path "C:\\Users\\timot\\OneDrive\\Bureau\\BOT TRADING BIG 2025\\crypto_trading_bot_CLAUDE\\data\\market_data"')
            sys.exit(1)

# Example commands:
# python evaluate_model.py evaluate --symbol BTCUSDT --timeframe 15m --start-date 2022-01-01 --end-date 2024-12-31 --data_path data/market_data
# For Windows paths with spaces, use quotes:
# python evaluate_model.py evaluate --symbol BTCUSDT --timeframe 15m --start-date 2022-01-01 --end-date 2024-12-31 --data_path "C:\Users\timot\OneDrive\Bureau\BOT TRADING BIG 2025\crypto_trading_bot_CLAUDE\data\market_data"


