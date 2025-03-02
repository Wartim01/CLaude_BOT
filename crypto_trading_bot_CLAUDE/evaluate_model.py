#!/usr/bin/env python
# evaluate_model.py
"""
Script d'évaluation approfondie du modèle LSTM
"""
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc


from ai.models.lstm_model import LSTMModel
from ai.models.feature_engineering import FeatureEngineering
from ai.models.model_validator import ModelValidator
from ai.models.continuous_learning import ContinuousLearning
from strategies.hybrid_strategy import HybridStrategy
from strategies.technical_bounce import TechnicalBounceStrategy
from core.adaptive_risk_manager import AdaptiveRiskManager
from config.config import DATA_DIR
from utils.logger import setup_logger

logger = setup_logger("evaluate_model")

def load_data(symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Charge les données OHLCV depuis le disque
    
    Args:
        symbol: Paire de trading
        timeframe: Intervalle de temps
        start_date: Date de début (YYYY-MM-DD)
        end_date: Date de fin (YYYY-MM-DD)
        
    Returns:
        DataFrame avec les données OHLCV
    """
    # Construire le chemin du fichier
    data_path = os.path.join(DATA_DIR, "market_data", f"{symbol}_{timeframe}_{start_date}_{end_date}.csv")
    
    # Vérifier si le fichier existe
    if not os.path.exists(data_path):
        logger.error(f"Fichier non trouvé: {data_path}")
        return pd.DataFrame()
    
    # Charger les données
    try:
        data = pd.read_csv(data_path)
        
        # Convertir la colonne timestamp en datetime
        if "timestamp" in data.columns:
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            data.set_index("timestamp", inplace=True)
        
        logger.info(f"Données chargées: {len(data)} lignes")
        return data
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données: {str(e)}")
        return pd.DataFrame()

def evaluate_direction_prediction(args):
    """
    Évalue la précision de la prédiction de direction sur des données de test
    
    Args:
        args: Arguments de ligne de commande
    """
    # Charger les données
    data = load_data(args.symbol, args.timeframe, args.start_date, args.end_date)
    
    if data.empty:
        logger.error("Données vides, impossible de continuer")
        return
    
    # Charger le modèle
    model_path = args.model_path or os.path.join(DATA_DIR, "models", "production", "lstm_final.h5")
    
    # Créer le validateur
    validator = ModelValidator()
    
    try:
        validator.load_model(model_path)
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
        return
    
    # Évaluer le modèle
    logger.info("Évaluation du modèle...")
    evaluation = validator.evaluate_on_test_set(data)
    
    # Générer des graphiques pour chaque horizon
    for horizon_key, metrics in evaluation["horizons"].items():
        horizon_name = horizon_key.replace("horizon_", "h")
        
        logger.info(f"\nHorizon: {horizon_name}")
        logger.info(f"Accuracy: {metrics['direction']['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['direction']['precision']:.4f}")
        logger.info(f"Recall: {metrics['direction']['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['direction']['f1_score']:.4f}")
        
        # Matrice de confusion
        cm = np.array(metrics["direction"]["confusion_matrix"])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                   xticklabels=["Baisse", "Hausse"], 
                   yticklabels=["Baisse", "Hausse"])
        plt.title(f"Matrice de confusion - {horizon_name}")
        plt.ylabel('Réalité')
        plt.xlabel('Prédiction')
        
        # Sauvegarder le graphique
        output_dir = os.path.join(DATA_DIR, "models", "evaluation", "figures")
        os.makedirs(output_dir, exist_ok=True)
        
        plt.savefig(os.path.join(output_dir, f"{args.symbol}_{horizon_name}_confusion_matrix.png"))
        plt.close()
    
    # Sauvegarder les résultats complets
    results_file = os.path.join(DATA_DIR, "models", "evaluation", 
                              f"{args.symbol}_evaluation_{datetime.now().strftime('%Y%m%d')}.json")
    
    with open(results_file, 'w') as f:
        json.dump(evaluation, f, indent=2, default=str)
    
    logger.info(f"Résultats d'évaluation sauvegardés: {results_file}")

def backtest_trading_performance(args):
    """
    Effectue un backtest complet de la stratégie hybride vs stratégie de base
    
    Args:
        args: Arguments de ligne de commande
    """
    # Charger les données
    data = load_data(args.symbol, args.timeframe, args.start_date, args.end_date)
    
    if data.empty:
        logger.error("Données vides, impossible de continuer")
        return
    
    # Initialiser le validateur
    validator = ModelValidator()
    
    try:
        # Charger le modèle
        model_path = args.model_path or os.path.join(DATA_DIR, "models", "production", "lstm_final.h5")
        validator.load_model(model_path)
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
        return
    
    # Backtest complet
    logger.info(f"Backtest sur {args.symbol} du {args.start_date} au {args.end_date}...")
    
    # Utiliser une capital initial personnalisé si spécifié
    initial_capital = args.capital
    
    # Comparer avec la stratégie de base
    comparison = validator.compare_with_baseline(
        data,
        initial_capital=initial_capital
    )
    
    # Afficher les résultats
    baseline = comparison["baseline"]
    lstm = comparison["lstm"]
    diff = comparison["comparison"]
    
    logger.info("\n=== Résultats du backtest ===")
    logger.info(f"Stratégie de base:")
    logger.info(f"  Rendement: {baseline['return_pct']:.2f}%")
    logger.info(f"  Drawdown max: {baseline['max_drawdown_pct']:.2f}%")
    logger.info(f"  Ratio de Sharpe: {baseline['sharpe_ratio']:.2f}")
    logger.info(f"  Nombre de trades: {baseline['num_trades']}")
    logger.info(f"  Taux de réussite: {baseline['win_rate']:.2f}%")
    
    logger.info(f"\nStratégie hybride LSTM:")
    logger.info(f"  Rendement: {lstm['return_pct']:.2f}%")
    logger.info(f"  Drawdown max: {lstm['max_drawdown_pct']:.2f}%")
    logger.info(f"  Ratio de Sharpe: {lstm['sharpe_ratio']:.2f}")
    logger.info(f"  Nombre de trades: {lstm['num_trades']}")
    logger.info(f"  Taux de réussite: {lstm['win_rate']:.2f}%")
    
    logger.info(f"\nDifférence (LSTM - Base):")
    logger.info(f"  Rendement: {diff['return_difference']:.2f}%")
    logger.info(f"  Amélioration drawdown: {diff['drawdown_improvement']:.2f}%")
    logger.info(f"  Amélioration Sharpe: {diff['sharpe_improvement']:.2f}")
    
    # Générer des graphiques
    
    # 1. Courbes d'équité
    plt.figure(figsize=(12, 6))
    
    baseline_equity = comparison["equity_curves"]["baseline"]
    lstm_equity = comparison["equity_curves"]["lstm"]
    
    plt.plot(baseline_equity, label="Stratégie de base", color="blue", linewidth=2)
    plt.plot(lstm_equity, label="Stratégie hybride LSTM", color="green", linewidth=2)
    
    plt.title(f"Comparaison des courbes d'équité - {args.symbol}")
    plt.xlabel("Jours de trading")
    plt.ylabel("Équité (USDT)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(DATA_DIR, "models", "evaluation", "figures", 
                           f"{args.symbol}_equity_curves.png"))
    plt.close()
    
    # 2. Distribution des profits par trade
    plt.figure(figsize=(12, 6))
    
    baseline_profits = [t["profit_pct"] for t in comparison["trades"]["baseline"]]
    lstm_profits = [t["profit_pct"] for t in comparison["trades"]["lstm"]]
    
    plt.hist(baseline_profits, bins=20, alpha=0.5, label="Stratégie de base", color="blue")
    plt.hist(lstm_profits, bins=20, alpha=0.5, label="Stratégie hybride LSTM", color="green")
    
    plt.title(f"Distribution des profits par trade - {args.symbol}")
    plt.xlabel("Profit (%)")
    plt.ylabel("Nombre de trades")
    plt.axvline(x=0, color='red', linestyle='--')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(DATA_DIR, "models", "evaluation", "figures", 
                           f"{args.symbol}_profit_distribution.png"))
    plt.close()
    
    # Sauvegarder les résultats complets
    results_file = os.path.join(DATA_DIR, "models", "evaluation", 
                              f"{args.symbol}_backtest_{datetime.now().strftime('%Y%m%d')}.json")
    
    with open(results_file, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    
    logger.info(f"Résultats du backtest sauvegardés: {results_file}")
    
    # Simulation du trading sur la période complète
    if args.simulate_hybrid:
        logger.info("\nSimulation du trading avec stratégie hybride...")
        simulate_hybrid_strategy(data, args.symbol, initial_capital, model_path)

def simulate_hybrid_strategy(data: pd.DataFrame, symbol: str, 
                          initial_capital: float, model_path: str):
    """
    Simule le trading avec la stratégie hybride complète
    
    Args:
        data: Données OHLCV
        symbol: Paire de trading
        initial_capital: Capital initial
        model_path: Chemin du modèle LSTM
    """
    # Créer les composants nécessaires
    lstm_model = LSTMModel()
    lstm_model.load(model_path)
    
    feature_engineering = FeatureEngineering()
    adaptive_risk_manager = AdaptiveRiskManager(initial_capital=initial_capital)
    
    # Créer un data fetcher simulé pour le backtest
    class MockDataFetcher:
        def __init__(self, data):
            self.data = data
            self.current_idx = 0
        
        def get_current_price(self, symbol):
            return self.data['close'].iloc[self.current_idx]
        
        def get_market_data(self, symbol):
            """Simule la récupération des données de marché pour le backtest"""
            # Obtenir les données récentes (jusqu'à l'indice actuel)
            current_data = self.data.iloc[:self.current_idx+1].copy()
            
            # Calculer les indicateurs sur ces données
            from indicators.trend import calculate_ema, calculate_adx
            from indicators.momentum import calculate_rsi
            from indicators.volatility import calculate_bollinger_bands, calculate_atr
            
            # Obtenir les 100 dernières lignes ou moins
            window_start = max(0, self.current_idx - 99)
            data_window = current_data.iloc[window_start:self.current_idx+1]
            
            ema = calculate_ema(data_window)
            rsi = calculate_rsi(data_window)
            bollinger = calculate_bollinger_bands(data_window)
            atr = calculate_atr(data_window)
            adx = calculate_adx(data_window)
            
            return {
                "symbol": symbol,
                "current_price": data_window["close"].iloc[-1],
                "primary_timeframe": {
                    "ohlcv": data_window,
                    "indicators": {
                        "ema": ema,
                        "rsi": rsi,
                        "bollinger": bollinger,
                        "atr": atr,
                        "adx": adx
                    }
                },
                "secondary_timeframes": {}
            }
    
    # Créer un market analyzer simulé
    class MockMarketAnalyzer:
        def analyze_market_state(self, symbol):
            return {
                "favorable": True,
                "cooldown": False,
                "details": {}
            }
    
    # Créer un position tracker simulé
    class MockPositionTracker:
        def __init__(self):
            self.positions = {}
            self.closed_positions = []
            self.position_id_counter = 0
        
        def add_position(self, position):
            position_id = position["id"]
            self.positions[position_id] = position
            return position_id
        
        def get_position(self, position_id):
            return self.positions.get(position_id)
        
        def get_open_positions(self, symbol=None):
            if symbol:
                return [p for p in self.positions.values() if p["symbol"] == symbol]
            return list(self.positions.values())
        
        def get_closed_positions(self, limit=100):
            return self.closed_positions[:limit]
        
        def close_position(self, position_id, close_data):
            if position_id in self.positions:
                position = self.positions.pop(position_id)
                position["close_time"] = datetime.now()
                position["close_data"] = close_data
                self.closed_positions.append(position)
                return True
            return False
        
        def generate_position_id(self):
            self.position_id_counter += 1
            return f"sim_{self.position_id_counter}"
    
    # Initialiser les composants
    mock_data_fetcher = MockDataFetcher(data)
    mock_market_analyzer = MockMarketAnalyzer()
    mock_position_tracker = MockPositionTracker()
    
    # Initialiser le scoring engine
    from ai.scoring_engine import ScoringEngine
    scoring_engine = ScoringEngine()
    
    # Créer la stratégie hybride
    hybrid_strategy = HybridStrategy(
        mock_data_fetcher,
        mock_market_analyzer,
        scoring_engine,
        lstm_model,
        adaptive_risk_manager
    )
    
    # Simulation
    equity_history = [initial_capital]
    trades = []
    open_positions = {}
    
    # Parcourir les données jour par jour (à partir de l'indice 100 pour avoir assez d'historique)
    window_size = lstm_model.input_length
    
    for i in range(window_size, len(data) - 1):
        # Mettre à jour l'indice courant
        mock_data_fetcher.current_idx = i
        
        # Prix actuel et prochain
        current_price = data['close'].iloc[i]
        next_price = data['close'].iloc[i+1]
        
        # 1. Gérer les positions ouvertes
        positions_to_close = []
        
        for pos_id, position in open_positions.items():
            side = position["side"]
            entry_price = position["entry_price"]
            stop_loss = position["stop_loss"]
            take_profit = position["take_profit"]
            
            # Vérifier si le stop-loss ou take-profit est atteint au prochain pas de temps
            if side == "BUY":
                if next_price <= stop_loss:
                    # Stop-loss atteint
                    profit_pct = (stop_loss - entry_price) / entry_price * 100 * position["leverage"]
                    positions_to_close.append((pos_id, profit_pct, "Stop-Loss"))
                elif next_price >= take_profit:
                    # Take-profit atteint
                    profit_pct = (take_profit - entry_price) / entry_price * 100 * position["leverage"]
                    positions_to_close.append((pos_id, profit_pct, "Take-Profit"))
            else:  # SELL
                if next_price >= stop_loss:
                    # Stop-loss atteint
                    profit_pct = (entry_price - stop_loss) / entry_price * 100 * position["leverage"]
                    positions_to_close.append((pos_id, profit_pct, "Stop-Loss"))
                elif next_price <= take_profit:
                    # Take-profit atteint
                    profit_pct = (entry_price - take_profit) / entry_price * 100 * position["leverage"]
                    positions_to_close.append((pos_id, profit_pct, "Take-Profit"))
            
            # Vérification de fermeture anticipée basée sur les prédictions LSTM
            position_update = hybrid_strategy.should_close_early(symbol, position, current_price)
            
            if position_update["should_close"]:
                if side == "BUY":
                    profit_pct = (next_price - entry_price) / entry_price * 100 * position["leverage"]
                else:
                    profit_pct = (entry_price - next_price) / entry_price * 100 * position["leverage"]
                
                positions_to_close.append((pos_id, profit_pct, "Signal LSTM"))
        
        # Fermer les positions
        for pos_id, profit_pct, reason in positions_to_close:
            position = open_positions.pop(pos_id)
            
            # Mettre à jour l'équité
            equity_change = equity_history[-1] * profit_pct / 100
            new_equity = equity_history[-1] + equity_change
            
            # Enregistrer le trade
            trade = {
                "day": i,
                "entry_day": position["entry_day"],
                "symbol": symbol,
                "side": position["side"],
                "entry_price": position["entry_price"],
                "exit_price": next_price,
                "profit_pct": profit_pct,
                "profit_amount": equity_change,
                "exit_reason": reason
            }
            trades.append(trade)
            
            # Mettre à jour le gestionnaire de risque
            adaptive_risk_manager.update_after_trade_closed({
                "pnl_absolute": equity_change,
                "pnl_percent": profit_pct
            })
        
        # 2. Chercher de nouvelles opportunités de trading
        if len(open_positions) < 3:  # Maximum 3 positions simultanées
            # Vérifier si une nouvelle position peut être ouverte
            risk_check = adaptive_risk_manager.can_open_new_position(mock_position_tracker)
            
            if risk_check["can_open"]:
                # Chercher une opportunité de trading
                opportunity = hybrid_strategy.find_trading_opportunity(symbol)
                
                if opportunity and opportunity["score"] >= hybrid_strategy.min_score:
                    # Calculer la taille de position
                    position_size = adaptive_risk_manager.calculate_position_size(
                        symbol,
                        opportunity,
                        opportunity.get("lstm_prediction")
                    )
                    
                    # Simuler un nouveau trade
                    entry_price = current_price
                    stop_loss = opportunity["stop_loss"]
                    take_profit = opportunity["take_profit"]
                    side = opportunity["side"]
                    
                    # Levier (depuis le profil de risque)
                    risk_profile = adaptive_risk_manager.risk_levels[adaptive_risk_manager.current_risk_profile]
                    leverage = risk_profile["leverage"]
                    
                    # Créer une nouvelle position
                    position_id = mock_position_tracker.generate_position_id()
                    position = {
                        "id": position_id,
                        "symbol": symbol,
                        "side": side,
                        "entry_price": entry_price,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "entry_day": i,
                        "score": opportunity["score"],
                        "leverage": leverage
                    }
                    
                    # Ajouter la position
                    open_positions[position_id] = position
        
        # Mettre à jour l'équité si pas de changement
        if len(positions_to_close) == 0:
            equity_history.append(equity_history[-1])
        else:
            equity_history.append(new_equity)
    
    # Calculer les statistiques finales
    final_equity = equity_history[-1]
    total_return = (final_equity - initial_capital) / initial_capital * 100
    
    # Calculer le drawdown maximum
    peak = initial_capital
    max_drawdown = 0
    
    for equity in equity_history:
        if equity > peak:
            peak = equity
        
        drawdown = (peak - equity) / peak * 100
        max_drawdown = max(max_drawdown, drawdown)
    
    # Calculer le ratio de Sharpe
    daily_returns = []
    
    for i in range(1, len(equity_history)):
        daily_return = (equity_history[i] - equity_history[i-1]) / equity_history[i-1]
        daily_returns.append(daily_return)
    
    if daily_returns:
        avg_return = sum(daily_returns) / len(daily_returns)
        std_return = np.std(daily_returns) if len(daily_returns) > 1 else 0
        
        annual_return = avg_return * 252
        annual_std = std_return * np.sqrt(252)
        
        sharpe_ratio = (annual_return - 0.01) / annual_std if annual_std > 0 else 0
    else:
        sharpe_ratio = 0
    
    # Calculer le taux de réussite
    if trades:
        winning_trades = [t for t in trades if t["profit_pct"] > 0]
        win_rate = len(winning_trades) / len(trades) * 100
    else:
        win_rate = 0
    
    # Afficher les résultats
    logger.info("\n=== Résultats de la simulation hybride ===")
    logger.info(f"Capital initial: {initial_capital} USDT")
    logger.info(f"Capital final: {final_equity:.2f} USDT")
    logger.info(f"Rendement total: {total_return:.2f}%")
    logger.info(f"Drawdown maximum: {max_drawdown:.2f}%")
    logger.info(f"Ratio de Sharpe: {sharpe_ratio:.2f}")
    logger.info(f"Nombre de trades: {len(trades)}")
    logger.info(f"Taux de réussite: {win_rate:.2f}%")
    
    # Générer un graphique de la courbe d'équité
    plt.figure(figsize=(12, 6))
    plt.plot(equity_history, linewidth=2)
    plt.title(f"Courbe d'équité - Stratégie hybride LSTM - {symbol}")
    plt.xlabel("Jours de trading")
    plt.ylabel("Équité (USDT)")
    plt.grid(True, alpha=0.3)
    
    # Ajouter des annotations pour les trades
    for trade in trades:
        day = trade["day"]
        if day < len(equity_history):
            equity = equity_history[day]
            
            if trade["profit_pct"] > 0:
                color = "green"
                marker = "^"
            else:
                color = "red"
                marker = "v"
            
            plt.plot(day, equity, marker=marker, color=color, markersize=8)
    
    plt.savefig(os.path.join(DATA_DIR, "models", "evaluation", "figures", 
                           f"{symbol}_hybrid_equity_curve.png"))
    plt.close()
    
    # Sauvegarder les résultats
    simulation_results = {
        "symbol": symbol,
        "initial_capital": initial_capital,
        "final_equity": float(final_equity),
        "total_return": float(total_return),
        "max_drawdown": float(max_drawdown),
        "sharpe_ratio": float(sharpe_ratio),
        "trades": trades,
        "win_rate": float(win_rate),
        "equity_history": [float(eq) for eq in equity_history],
        "timestamp": datetime.now().isoformat()
    }
    
    results_file = os.path.join(DATA_DIR, "models", "evaluation", 
                              f"{symbol}_hybrid_simulation_{datetime.now().strftime('%Y%m%d')}.json")
    
    with open(results_file, 'w') as f:
        json.dump(simulation_results, f, indent=2, default=str)
    
    logger.info(f"Résultats de la simulation hybride sauvegardés: {results_file}")

def evaluate_incremental_learning(args):
    """
    Évalue l'efficacité de l'apprentissage continu sur des données récentes
    
    Args:
        args: Arguments de ligne de commande
    """
    # Charger les données
    training_data = load_data(args.symbol, args.timeframe, args.train_start, args.train_end)
    test_data = load_data(args.symbol, args.timeframe, args.test_start, args.test_end)
    
    if training_data.empty or test_data.empty:
        logger.error("Données insuffisantes, impossible de continuer")
        return
    
    # Charger ou créer un modèle LSTM
    model_path = args.model_path or os.path.join(DATA_DIR, "models", "production", "lstm_final.h5")
    
    if not os.path.exists(model_path) or args.retrain:
        logger.info("Création d'un nouveau modèle pour l'apprentissage continu...")
        
        # Paramètres du modèle
        model_params = {
            "input_length": args.sequence_length,
            "feature_dim": args.feature_dim,
            "lstm_units": [args.lstm_units, args.lstm_units // 2, args.lstm_units // 4],
            "dropout_rate": 0.3,
            "learning_rate": 0.001,
            "l1_reg": 0.0001,
            "l2_reg": 0.0001,
            "use_attention": True,
            "use_residual": True,
            "prediction_horizons": [12, 24, 96]
        }
        
        # Entraîner un modèle de base
        from ai.models.model_trainer import ModelTrainer
        trainer = ModelTrainer(model_params)
        
        # Préparer les données
        _, normalized_data = trainer.prepare_data(training_data)
        
        # Entraîner le modèle
        train_results = trainer.train_final_model(
            normalized_data,
            epochs=50,
            batch_size=32,
            test_ratio=0.15
        )
        
        logger.info("Modèle de base entraîné")
        model = trainer.model
    else:
        logger.info(f"Chargement du modèle existant: {model_path}")
        model = LSTMModel()
        model.load(model_path)
    
    # Initialiser le module d'apprentissage continu
    continuous_learning = ContinuousLearning(
        model=model,
        feature_engineering=FeatureEngineering(),
        experience_buffer_size=5000,
        drift_threshold=0.15,
        drift_window_size=50
    )
    
    # Évaluer le modèle avant l'apprentissage continu
    validator = ModelValidator(model, continuous_learning.feature_engineering)
    
    logger.info("Évaluation du modèle avant l'apprentissage continu...")
    pre_evaluation = validator.evaluate_on_test_set(test_data)
    
    # Diviser les données de test en mini-batches pour simuler des mises à jour progressives
    batch_size = args.batch_size
    num_batches = len(test_data) // batch_size
    
    logger.info(f"Simulation de l'apprentissage continu sur {num_batches} mini-batches...")
    update_history = []
    
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, len(test_data))
        
        batch = test_data.iloc[batch_start:batch_end]
        
        logger.info(f"Traitement du mini-batch {i+1}/{num_batches} ({len(batch)} échantillons)")
        
        # Traiter le mini-batch
        update_result = continuous_learning.process_new_data(batch, min_samples=args.min_samples)
        
        # Enregistrer le résultat
        update_history.append({
            "batch": i,
            "updated": update_result.get("updated", False),
            "drift_detected": update_result.get("drift_detected", False),
            "evaluation": update_result.get("evaluation", {}),
            "timestamp": datetime.now().isoformat()
        })
    
    # Évaluer le modèle après l'apprentissage continu
    logger.info("Évaluation du modèle après l'apprentissage continu...")
    post_evaluation = validator.evaluate_on_test_set(test_data)
    
    # Calculer les améliorations
    improvements = {}
    
    for horizon_key, pre_metrics in pre_evaluation["horizons"].items():
        post_metrics = post_evaluation["horizons"].get(horizon_key, {})
        
        if "direction" in pre_metrics and "direction" in post_metrics:
            pre_acc = pre_metrics["direction"]["accuracy"]
            post_acc = post_metrics["direction"]["accuracy"]
            
            improvements[horizon_key] = {
                "accuracy_improvement": post_acc - pre_acc,
                "percent_improvement": (post_acc - pre_acc) / pre_acc * 100 if pre_acc > 0 else 0
            }
    
    # Afficher les résultats
    logger.info("\n=== Résultats de l'apprentissage continu ===")
    logger.info(f"Mini-batches traités: {num_batches}")
    logger.info(f"Mises à jour effectuées: {sum(1 for u in update_history if u['updated'])}")
    
    logger.info("\nPrécision de direction avant/après:")
    for horizon_key, improvement in improvements.items():
        pre_acc = pre_evaluation["horizons"][horizon_key]["direction"]["accuracy"]
        post_acc = post_evaluation["horizons"][horizon_key]["direction"]["accuracy"]
        
        logger.info(f"  {horizon_key}: {pre_acc:.4f} -> {post_acc:.4f} ({improvement['percent_improvement']:.2f}%)")
    
    # Sauvegarder les résultats
    results = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "training_period": f"{args.train_start} to {args.train_end}",
        "testing_period": f"{args.test_start} to {args.test_end}",
        "pre_evaluation": pre_evaluation,
        "post_evaluation": post_evaluation,
        "improvements": improvements,
        "update_history": update_history,
        "timestamp": datetime.now().isoformat()
    }
    
    results_file = os.path.join(DATA_DIR, "models", "continuous_learning", 
                              f"{args.symbol}_cl_results_{datetime.now().strftime('%Y%m%d')}.json")
    
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Résultats de l'apprentissage continu sauvegardés: {results_file}")
    
    # Générer des graphiques
    
    # 1. Évolution de la précision au fil des mises à jour
    plt.figure(figsize=(12, 6))
    
    # Collecter les données pour chaque horizon
    horizon_accuracies = {}
    
    for i, update in enumerate(update_history):
        if "evaluation" in update and "horizons" in update["evaluation"]:
            for horizon_key, metrics in update["evaluation"]["horizons"].items():
                if "direction" in metrics:
                    if horizon_key not in horizon_accuracies:
                        horizon_accuracies[horizon_key] = []
                    
                    # Ajouter la précision
                    horizon_accuracies[horizon_key].append((i, metrics["direction"]["accuracy"]))
    
    # Tracer les courbes pour chaque horizon
    for horizon_key, accuracies in horizon_accuracies.items():
        if accuracies:
            x = [a[0] for a in accuracies]
            y = [a[1] for a in accuracies]
            
            plt.plot(x, y, label=horizon_key, linewidth=2, marker='o')
    
    plt.title("Évolution de la précision de direction pendant l'apprentissage continu")
    plt.xlabel("Mini-batch")
    plt.ylabel("Précision")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(DATA_DIR, "models", "continuous_learning", 
                           f"{args.symbol}_cl_accuracy_evolution.png"))
    plt.close()

def main():
    """Point d'entrée principal du script"""
    parser = argparse.ArgumentParser(description="Évaluation approfondie du modèle LSTM")
    
    subparsers = parser.add_subparsers(dest="command", help="Commande à exécuter")
    
    # Parser pour l'évaluation des prédictions de direction
    direction_parser = subparsers.add_parser("direction", help="Évaluer la précision de prédiction de direction")
    
    # Arguments pour les données
    direction_parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Paire de trading")
    direction_parser.add_argument("--timeframe", type=str, default="15m", help="Intervalle de temps")
    direction_parser.add_argument("--start-date", type=str, required=True, help="Date de début (YYYY-MM-DD)")
    direction_parser.add_argument("--end-date", type=str, required=True, help="Date de fin (YYYY-MM-DD)")
    direction_parser.add_argument("--model-path", type=str, help="Chemin vers le modèle à évaluer")
    
    # Parser pour le backtest
    backtest_parser = subparsers.add_parser("backtest", help="Backtest complet de la stratégie")
    
    # Arguments pour les données
    backtest_parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Paire de trading")
    backtest_parser.add_argument("--timeframe", type=str, default="15m", help="Intervalle de temps")
    backtest_parser.add_argument("--start-date", type=str, required=True, help="Date de début (YYYY-MM-DD)")
    backtest_parser.add_argument("--end-date", type=str, required=True, help="Date de fin (YYYY-MM-DD)")
    backtest_parser.add_argument("--model-path", type=str, help="Chemin vers le modèle à évaluer")
    backtest_parser.add_argument("--capital", type=float, default=200, help="Capital initial")
    backtest_parser.add_argument("--simulate-hybrid", action="store_true", help="Simuler la stratégie hybride complète")
    
    # Parser pour l'apprentissage continu
    cl_parser = subparsers.add_parser("continuous", help="Évaluer l'apprentissage continu")
    
    # Arguments pour les données d'entraînement
    cl_parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Paire de trading")
    cl_parser.add_argument("--timeframe", type=str, default="15m", help="Intervalle de temps")
    cl_parser.add_argument("--train-start", type=str, required=True, help="Date de début d'entraînement (YYYY-MM-DD)")
    cl_parser.add_argument("--train-end", type=str, required=True, help="Date de fin d'entraînement (YYYY-MM-DD)")
    cl_parser.add_argument("--test-start", type=str, required=True, help="Date de début de test (YYYY-MM-DD)")
    cl_parser.add_argument("--test-end", type=str, required=True, help="Date de fin de test (YYYY-MM-DD)")
    
    # Arguments pour le modèle
    cl_parser.add_argument("--model-path", type=str, help="Chemin vers le modèle à utiliser")
    cl_parser.add_argument("--retrain", action="store_true", help="Ré-entraîner le modèle de base")
    cl_parser.add_argument("--sequence-length", type=int, default=60, help="Longueur des séquences d'entrée")
    cl_parser.add_argument("--feature-dim", type=int, default=30, help="Dimension des caractéristiques")
    cl_parser.add_argument("--lstm-units", type=int, default=128, help="Nombre d'unités LSTM")
    
    # Arguments pour l'apprentissage continu
    cl_parser.add_argument("--batch-size", type=int, default=100, help="Taille des mini-batches pour l'apprentissage continu")
    cl_parser.add_argument("--min-samples", type=int, default=30, help="Nombre minimum d'échantillons pour mettre à jour le modèle")
    
    args = parser.parse_args()
    
    if args.command == "direction":
        evaluate_direction_prediction(args)
    elif args.command == "backtest":
        backtest_trading_performance(args)
    elif args.command == "continuous":
        evaluate_incremental_learning(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()