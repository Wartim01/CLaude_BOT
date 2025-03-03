# utils/model_backtester.py
"""
Module spécialisé pour le backtesting des modèles LSTM sur les données historiques
Permet d'évaluer les performances des modèles avant leur déploiement en production
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import json

from config.config import DATA_DIR
from config.trading_params import RISK_PER_TRADE_PERCENT, STOP_LOSS_PERCENT, TAKE_PROFIT_PERCENT, LEVERAGE

from ai.models.lstm_model import LSTMModel
from ai.models.feature_engineering import FeatureEngineering
from strategies.hybrid_strategy import HybridStrategy
from strategies.technical_bounce import TechnicalBounceStrategy
from strategies.market_state import MarketStateAnalyzer
from ai.scoring_engine import ScoringEngine
from core.adaptive_risk_manager import AdaptiveRiskManager
from utils.backtest_engine import _simulate_trading
from utils.logger import setup_logger

logger = setup_logger("model_backtester")

class ModelBacktester:
    """
    Backtester spécialisé pour les modèles LSTM et les stratégies hybrides
    Permet de comparer les performances contre les stratégies de base
    """
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialise le backtester
        
        Args:
            model_path: Chemin vers le modèle LSTM préentraîné
        """
        self.output_dir = os.path.join(DATA_DIR, "backtest_results", "model_backtest")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Charger ou initialiser le modèle LSTM
        self.model = None
        if model_path:
            try:
                self.model = LSTMModel()
                self.model.load(model_path)
                logger.info(f"Modèle LSTM chargé: {model_path}")
            except Exception as e:
                logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
        
        # Initialiser le module d'ingénierie des caractéristiques
        self.feature_engineering = FeatureEngineering()
        
        # Initialiser le gestionnaire de risque adaptatif
        self.risk_manager = AdaptiveRiskManager()
        
        # Initialiser le moteur de scoring
        self.scoring_engine = ScoringEngine()
    
    def backtest_model(self, data: pd.DataFrame, symbol: str,
                     initial_capital: float = 200,
                     compare_with_baseline: bool = True,
                     plot_results: bool = True,
                     output_filename: Optional[str] = None) -> Dict:
        """
        Exécute un backtest du modèle LSTM sur les données fournies
        
        Args:
            data: DataFrame avec les données OHLCV
            symbol: Symbole de la paire de trading
            initial_capital: Capital initial (en USDT)
            compare_with_baseline: Comparer avec la stratégie de base
            plot_results: Générer des graphiques des résultats
            output_filename: Nom du fichier de sortie pour les résultats
            
        Returns:
            Résultats du backtest
        """
        if self.model is None:
            logger.error("Aucun modèle LSTM n'a été chargé")
            return {"success": False, "error": "Modèle non initialisé"}
        
        # 1. Préparer les données
        try:
            featured_data, normalized_data = self._prepare_data(data)
        except Exception as e:
            logger.error(f"Erreur lors de la préparation des données: {str(e)}")
            return {"success": False, "error": f"Erreur de préparation: {str(e)}"}
        
        # 2. Initialiser les composants pour le backtest
        lstm_results = self._backtest_lstm_strategy(
            data=data,
            featured_data=featured_data,
            normalized_data=normalized_data,
            symbol=symbol,
            initial_capital=initial_capital
        )
        
        # 3. Comparer avec la stratégie de base si demandé
        baseline_results = None
        if compare_with_baseline:
            baseline_results = self._backtest_baseline_strategy(
                data=data,
                symbol=symbol,
                initial_capital=initial_capital
            )
        
        # 4. Calculer les comparaisons
        comparison = {}
        if baseline_results:
            comparison = self._compare_strategies(lstm_results, baseline_results)
        
        # 5. Générer les graphiques si demandé
        if plot_results:
            self._generate_charts(lstm_results, baseline_results, symbol)
        
        # 6. Sauvegarder les résultats
        if output_filename is None:
            output_filename = f"{symbol}_model_backtest_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        
        output_path = os.path.join(self.output_dir, output_filename)
        
        results = {
            "symbol": symbol,
            "date": datetime.now().isoformat(),
            "initial_capital": initial_capital,
            "lstm": lstm_results,
            "baseline": baseline_results,
            "comparison": comparison
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Résultats sauvegardés: {output_path}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des résultats: {str(e)}")
        
        return results
    
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prépare les données pour le backtest
        
        Args:
            data: DataFrame avec les données OHLCV
            
        Returns:
            Tuple de (featured_data, normalized_data)
        """
        # Créer les caractéristiques avancées
        featured_data = self.feature_engineering.create_features(
            data,
            include_time_features=True,
            include_price_patterns=True
        )
        
        # Normaliser les caractéristiques
        normalized_data = self.feature_engineering.scale_features(
            featured_data,
            is_training=False,
            method='standard',
            feature_group='lstm'
        )
        
        return featured_data, normalized_data
    
    def _backtest_lstm_strategy(self, data: pd.DataFrame,
                              featured_data: pd.DataFrame,
                              normalized_data: pd.DataFrame,
                              symbol: str,
                              initial_capital: float) -> Dict:
        """
        Effectue un backtest de la stratégie hybride avec LSTM
        
        Args:
            data: DataFrame original
            featured_data: DataFrame avec caractéristiques
            normalized_data: DataFrame avec caractéristiques normalisées
            symbol: Symbole de la paire
            initial_capital: Capital initial
            
        Returns:
            Résultats du backtest
        """
        logger.info(f"Démarrage du backtest LSTM pour {symbol}")
        
        # 1. Initialisation des composants
        # Simuler le data fetcher (pour obtenir les données historiques)
        class MockDataFetcher:
            def __init__(self, price_data, featured_data):
                self.price_data = price_data
                self.featured_data = featured_data
                self.current_idx = 0
                
            def set_index(self, idx):
                self.current_idx = idx
                
            def get_current_price(self, symbol):
                return self.price_data.iloc[self.current_idx]['close']
                
            def get_market_data(self, symbol):
                # Fenêtre pour les indicateurs
                window_start = max(0, self.current_idx - 100)
                window_end = self.current_idx + 1
                window_data = self.price_data.iloc[window_start:window_end]
                featured_window = self.featured_data.iloc[window_start:window_end]
                
                # Extraire les indicateurs du DataFrame de caractéristiques
                return {
                    "symbol": symbol,
                    "current_price": self.price_data.iloc[self.current_idx]['close'],
                    "primary_timeframe": {
                        "ohlcv": window_data,
                        "indicators": self._extract_indicators(featured_window)
                    }
                }
                
            def _extract_indicators(self, featured_data):
                # Extraire les indicateurs des colonnes du DataFrame de caractéristiques
                indicators = {}
                
                # Récupérer les indicateurs communs
                if 'rsi' in featured_data.columns:
                    indicators['rsi'] = featured_data['rsi']
                
                if 'bb_upper' in featured_data.columns and 'bb_lower' in featured_data.columns:
                    indicators['bollinger'] = {
                        'upper': featured_data['bb_upper'],
                        'middle': featured_data['bb_middle'] if 'bb_middle' in featured_data.columns else None,
                        'lower': featured_data['bb_lower'],
                        'percent_b': featured_data['bb_percent_b'] if 'bb_percent_b' in featured_data.columns else None
                    }
                
                if 'adx' in featured_data.columns:
                    indicators['adx'] = {
                        'adx': featured_data['adx'],
                        'plus_di': featured_data['plus_di'] if 'plus_di' in featured_data.columns else None,
                        'minus_di': featured_data['minus_di'] if 'minus_di' in featured_data.columns else None
                    }
                
                emas = {}
                for col in featured_data.columns:
                    if col.startswith('ema_'):
                        period = col.split('_')[1]
                        emas[col] = featured_data[col]
                
                if emas:
                    indicators['ema'] = emas
                
                if 'atr' in featured_data.columns:
                    indicators['atr'] = featured_data['atr']
                
                return indicators
        
        # Simuler le market analyzer
        class MockMarketAnalyzer:
            def analyze_market_state(self, symbol):
                return {
                    "favorable": True,
                    "details": {}
                }
        
        # Simuler le position tracker
        class MockPositionTracker:
            def __init__(self):
                self.positions = {}
                self.closed_positions = []
                self.position_id = 0
                
            def add_position(self, position):
                self.position_id += 1
                position_id = f"backtest_{self.position_id}"
                position["id"] = position_id
                self.positions[position_id] = position
                return position_id
                
            def close_position(self, position_id, close_data):
                if position_id in self.positions:
                    position = self.positions.pop(position_id)
                    position.update(close_data)
                    self.closed_positions.append(position)
                    return True
                return False
                
            def get_open_positions(self, symbol=None):
                if symbol:
                    return [p for p in self.positions.values() if p["symbol"] == symbol]
                return list(self.positions.values())
                
            def get_all_open_positions(self):
                result = {}
                for position in self.positions.values():
                    symbol = position["symbol"]
                    if symbol not in result:
                        result[symbol] = []
                    result[symbol].append(position)
                return result
                
            def get_closed_positions(self, limit=None):
                if limit:
                    return self.closed_positions[-limit:]
                return self.closed_positions
        
        # 2. Initialiser les objets
        data_fetcher = MockDataFetcher(data, featured_data)
        market_analyzer = MockMarketAnalyzer()
        position_tracker = MockPositionTracker()
        adaptive_risk_manager = AdaptiveRiskManager(initial_capital=initial_capital)
        
        # 3. Initialiser la stratégie hybride
        hybrid_strategy = HybridStrategy(
            data_fetcher=data_fetcher,
            market_analyzer=market_analyzer,
            scoring_engine=self.scoring_engine,
            lstm_model=self.model,
            adaptive_risk_manager=adaptive_risk_manager
        )
        
        # 4. Effectuer le backtest
        # Initialiser les variables de suivi
        equity_curve = [initial_capital]
        current_equity = initial_capital
        trades = []
        timestamps = [data.index[0]]
        
        # Période de warmup pour avoir assez de données historiques
        warmup_period = max(100, self.model.input_length + max(self.model.horizon_periods))
        
        # Parcourir les données jour par jour
        for i in range(warmup_period, len(data) - 1):
            # Mettre à jour l'indice courant du data fetcher
            data_fetcher.set_index(i)
            
            # Date et prix actuels
            current_date = data.index[i]
            current_price = data['close'].iloc[i]
            next_price = data['close'].iloc[i+1]
            
            # 1. Gérer les positions ouvertes
            open_positions = position_tracker.get_open_positions()
            for position in open_positions:
                # Vérifier si le take profit ou stop loss est atteint
                entry_price = position["entry_price"]
                stop_loss = position["stop_loss"]
                take_profit = position["take_profit"]
                side = position["side"]
                position_id = position["id"]
                leverage = position.get("leverage", LEVERAGE)
                
                # Simuler l'évolution de la position
                hit_tp, hit_sl = False, False
                
                if side == "BUY":
                    if next_price <= stop_loss:
                        hit_sl = True
                        exit_price = stop_loss
                        profit_pct = -abs((entry_price - stop_loss) / entry_price * 100 * leverage)
                    elif next_price >= take_profit:
                        hit_tp = True
                        exit_price = take_profit
                        profit_pct = abs((take_profit - entry_price) / entry_price * 100 * leverage)
                else:  # SELL
                    if next_price >= stop_loss:
                        hit_sl = True
                        exit_price = stop_loss
                        profit_pct = -abs((stop_loss - entry_price) / entry_price * 100 * leverage)
                    elif next_price <= take_profit:
                        hit_tp = True
                        exit_price = take_profit
                        profit_pct = abs((entry_price - take_profit) / entry_price * 100 * leverage)
                
                # Vérifier si la position doit être fermée anticipativement selon le LSTM
                early_close = False
                if not hit_tp and not hit_sl:
                    early_close_check = hybrid_strategy.should_close_early(
                        symbol=symbol,
                        position=position,
                        current_price=current_price
                    )
                    
                    if early_close_check["should_close"]:
                        early_close = True
                        exit_price = next_price
                        if side == "BUY":
                            profit_pct = (exit_price - entry_price) / entry_price * 100 * leverage
                        else:
                            profit_pct = (entry_price - exit_price) / entry_price * 100 * leverage
                
                # Fermer la position si nécessaire
                if hit_tp or hit_sl or early_close:
                    # Calculer le profit en USD
                    position_size = position.get("size", 0)
                    profit_usd = position_size * (profit_pct / 100)
                    
                    # Mettre à jour l'équité
                    current_equity += profit_usd
                    
                    # Déterminer la raison de sortie
                    exit_reason = "Stop Loss" if hit_sl else "Take Profit" if hit_tp else "Signal LSTM"
                    
                    # Créer les données de clôture
                    close_data = {
                        "exit_price": exit_price,
                        "exit_date": current_date,
                        "profit_pct": profit_pct,
                        "profit_usd": profit_usd,
                        "exit_reason": exit_reason
                    }
                    
                    # Fermer la position
                    position_tracker.close_position(position_id, close_data)
                    
                    # Notifier le risk manager
                    adaptive_risk_manager.update_after_trade_closed({
                        "pnl_percent": profit_pct,
                        "pnl_absolute": profit_usd
                    })
                    
                    # Ajouter aux trades
                    trade_info = {
                        "entry_date": position["entry_date"],
                        "exit_date": current_date,
                        "symbol": symbol,
                        "side": side,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "size": position_size,
                        "profit_pct": profit_pct,
                        "profit_usd": profit_usd,
                        "exit_reason": exit_reason
                    }
                    trades.append(trade_info)
            
            # 2. Chercher de nouvelles opportunités si pas trop de positions ouvertes
            if len(position_tracker.get_open_positions()) < 3:  # Max 3 positions simultanées
                # Vérifier les conditions de risque
                risk_check = adaptive_risk_manager.can_open_new_position(position_tracker)
                
                if risk_check["can_open"]:
                    # Chercher une opportunité avec la stratégie hybride
                    opportunity = hybrid_strategy.find_trading_opportunity(symbol)
                    
                    if opportunity and opportunity["score"] >= hybrid_strategy.min_score:
                        # Calculer la taille de position
                        position_size = adaptive_risk_manager.calculate_position_size(
                            symbol=symbol,
                            opportunity=opportunity,
                            lstm_prediction=opportunity.get("lstm_prediction")
                        )
                        
                        # Vérifier qu'il y a des fonds disponibles
                        if position_size > 0:
                            # Créer la position
                            new_position = {
                                "symbol": symbol,
                                "entry_price": current_price,
                                "stop_loss": opportunity["stop_loss"],
                                "take_profit": opportunity["take_profit"],
                                "side": opportunity["side"],
                                "size": position_size,
                                "entry_date": current_date,
                                "score": opportunity["score"],
                                "leverage": opportunity.get("leverage", LEVERAGE)
                            }
                            
                            # Ajouter la position
                            position_tracker.add_position(new_position)
            
            # 3. Mettre à jour l'equity curve
            equity_curve.append(current_equity)
            timestamps.append(current_date)
        
        # 5. Calculer les métriques de performance
        closed_positions = position_tracker.get_closed_positions()
        total_trades = len(closed_positions)
        
        # Séparer les trades gagnants et perdants
        winning_trades = [t for t in closed_positions if t["profit_pct"] > 0]
        losing_trades = [t for t in closed_positions if t["profit_pct"] <= 0]
        
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
        
        # Calcul du profit moyen par trade
        avg_profit = np.mean([t["profit_pct"] for t in closed_positions]) if closed_positions else 0
        avg_win = np.mean([t["profit_pct"] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t["profit_pct"] for t in losing_trades]) if losing_trades else 0
        
        # Calcul du ratio profit/perte
        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 and winning_trades and losing_trades else 0
        
        # Calcul du drawdown maximum
        drawdowns = []
        peak = equity_curve[0]
        
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak * 100
            drawdowns.append(drawdown)
        
        max_drawdown = max(drawdowns)
        
        # Calcul du Sharpe ratio
        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe_ratio = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
        
        # Assembler les résultats
        results = {
            "initial_capital": initial_capital,
            "final_equity": equity_curve[-1],
            "return_pct": (equity_curve[-1] / initial_capital - 1) * 100,
            "num_trades": total_trades,
            "win_rate": win_rate,
            "avg_profit": avg_profit,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_loss_ratio": profit_loss_ratio,
            "max_drawdown_pct": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "equity_curve": equity_curve,
            "timestamps": [str(ts) for ts in timestamps],
            "trades": trades
        }
        
        logger.info(f"Backtest LSTM terminé: {total_trades} trades, win rate: {win_rate:.2f}%, retour: {results['return_pct']:.2f}%")
        
        return results
    
    def _backtest_baseline_strategy(self, data: pd.DataFrame, symbol: str, initial_capital: float) -> Dict:
        """
        Effectue un backtest de la stratégie de base (sans LSTM)
        
        Args:
            data: DataFrame avec les données OHLCV
            symbol: Symbole de la paire
            initial_capital: Capital initial
            
        Returns:
            Résultats du backtest
        """
        logger.info(f"Démarrage du backtest de la stratégie de base pour {symbol}")
        
        # Structure similaire au backtest LSTM, mais en utilisant TechnicalBounceStrategy au lieu de HybridStrategy
        
        # 1. Initialisation des composants (comme dans _backtest_lstm_strategy)
        # Pour éviter la répétition de code, nous pouvons réutiliser les mêmes classes mock
        
        class MockDataFetcher:
            def __init__(self, price_data):
                self.price_data = price_data
                self.current_idx = 0
                
            def set_index(self, idx):
                self.current_idx = idx
                
            def get_current_price(self, symbol):
                return self.price_data.iloc[self.current_idx]['close']
                
            def get_market_data(self, symbol):
                # Fenêtre pour les indicateurs
                window_start = max(0, self.current_idx - 100)
                window_end = self.current_idx + 1
                window_data = self.price_data.iloc[window_start:window_end]
                
                # Calculer les indicateurs
                from indicators.trend import calculate_ema, calculate_adx
                from indicators.momentum import calculate_rsi
                from indicators.volatility import calculate_bollinger_bands, calculate_atr
                
                indicators = {}
                
                # RSI
                indicators['rsi'] = calculate_rsi(window_data)
                
                # Bandes de Bollinger
                bb = calculate_bollinger_bands(window_data)
                indicators['bollinger'] = bb
                
                # EMA
                emas = calculate_ema(window_data)
                indicators['ema'] = emas
                
                # ADX
                adx = calculate_adx(window_data)
                indicators['adx'] = adx
                
                # ATR
                indicators['atr'] = calculate_atr(window_data)
                
                return {
                    "symbol": symbol,
                    "current_price": self.price_data.iloc[self.current_idx]['close'],
                    "primary_timeframe": {
                        "ohlcv": window_data,
                        "indicators": indicators
                    }
                }
        
        class MockMarketAnalyzer:
            def analyze_market_state(self, symbol):
                return {
                    "favorable": True,
                    "details": {}
                }
        
        class MockPositionTracker:
            def __init__(self):
                self.positions = {}
                self.closed_positions = []
                self.position_id = 0
                
            def add_position(self, position):
                self.position_id += 1
                position_id = f"backtest_base_{self.position_id}"
                position["id"] = position_id
                self.positions[position_id] = position
                return position_id
                
            def close_position(self, position_id, close_data):
                if position_id in self.positions:
                    position = self.positions.pop(position_id)
                    position.update(close_data)
                    self.closed_positions.append(position)
                    return True
                return False
                
            def get_open_positions(self, symbol=None):
                if symbol:
                    return [p for p in self.positions.values() if p["symbol"] == symbol]
                return list(self.positions.values())
                
            def get_all_open_positions(self):
                result = {}
                for position in self.positions.values():
                    symbol = position["symbol"]
                    if symbol not in result:
                        result[symbol] = []
                    result[symbol].append(position)
                return result
                
            def get_closed_positions(self, limit=None):
                if limit:
                    return self.closed_positions[-limit:]
                return self.closed_positions
        
        # 2. Initialiser les objets
        data_fetcher = MockDataFetcher(data)
        market_analyzer = MockMarketAnalyzer()
        position_tracker = MockPositionTracker()
        risk_manager = AdaptiveRiskManager(initial_capital=initial_capital)
        
        # 3. Initialiser la stratégie de base (sans LSTM)
        base_strategy = TechnicalBounceStrategy(
            data_fetcher=data_fetcher,
            market_analyzer=market_analyzer,
            scoring_engine=self.scoring_engine
        )
        
        # 4. Effectuer le backtest
        # Initialiser les variables de suivi
        equity_curve = [initial_capital]
        current_equity = initial_capital
        trades = []
        timestamps = [data.index[0]]
        
        # Période de warmup
        warmup_period = 100
        
        # Parcourir les données jour par jour
        for i in range(warmup_period, len(data) - 1):
            # Mettre à jour l'indice courant du data fetcher
            data_fetcher.set_index(i)
            
            # Date et prix actuels
            current_date = data.index[i]
            current_price = data['close'].iloc[i]
            next_price = data['close'].iloc[i+1]
            
            # 1. Gérer les positions ouvertes
            open_positions = position_tracker.get_open_positions()
            for position in open_positions:
                # Vérifier si le take profit ou stop loss est atteint
                entry_price = position["entry_price"]
                stop_loss = position["stop_loss"]
                take_profit = position["take_profit"]
                side = position["side"]
                position_id = position["id"]
                leverage = position.get("leverage", LEVERAGE)
                
                # Simuler l'évolution de la position
                hit_tp, hit_sl = False, False
                
                if side == "BUY":
                    if next_price <= stop_loss:
                        hit_sl = True
                        exit_price = stop_loss
                        profit_pct = -abs((entry_price - stop_loss) / entry_price * 100 * leverage)
                    elif next_price >= take_profit:
                        hit_tp = True
                        exit_price = take_profit
                        profit_pct = abs((take_profit - entry_price) / entry_price * 100 * leverage)
                else:  # SELL
                    if next_price >= stop_loss:
                        hit_sl = True
                        exit_price = stop_loss
                        profit_pct = -abs((stop_loss - entry_price) / entry_price * 100 * leverage)
                    elif next_price <= take_profit:
                        hit_tp = True
                        exit_price = take_profit
                        profit_pct = abs((entry_price - take_profit) / entry_price * 100 * leverage)
                
                # Fermer la position si nécessaire
                if hit_tp or hit_sl:
                    # Calculer le profit en USD
                    position_size = position.get("size", 0)
                    profit_usd = position_size * (profit_pct / 100)
                    
                    # Mettre à jour l'équité
                    current_equity += profit_usd
                    
                    # Déterminer la raison de sortie
                    exit_reason = "Stop Loss" if hit_sl else "Take Profit"
                    
                    # Créer les données de clôture
                    close_data = {
                        "exit_price": exit_price,
                        "exit_date": current_date,
                        "profit_pct": profit_pct,
                        "profit_usd": profit_usd,
                        "exit_reason": exit_reason
                    }
                    
                    # Fermer la position
                    position_tracker.close_position(position_id, close_data)
                    
                    # Notifier le risk manager
                    risk_manager.update_after_trade_closed({
                        "pnl_percent": profit_pct,
                        "pnl_absolute": profit_usd
                    })
                    
                    # Ajouter aux trades
                    trade_info = {
                        "entry_date": position["entry_date"],
                        "exit_date": current_date,
                        "symbol": symbol,
                        "side": side,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "size": position_size,
                        "profit_pct": profit_pct,
                        "profit_usd": profit_usd,
                        "exit_reason": exit_reason
                    }
                    trades.append(trade_info)
            
            # 2. Chercher de nouvelles opportunités si pas trop de positions ouvertes
            if len(position_tracker.get_open_positions()) < 3:  # Max 3 positions simultanées
                # Vérifier les conditions de risque
                risk_check = risk_manager.can_open_new_position(position_tracker)
                
                if risk_check["can_open"]:
                    # Chercher une opportunité avec la stratégie de base
                    opportunity = base_strategy.find_trading_opportunity(symbol)
                    
                    if opportunity and opportunity["score"] >= base_strategy.min_score:
                        # Calculer la taille de position
                        position_size = risk_manager.calculate_position_size(
                            symbol=symbol,
                            opportunity=opportunity
                        )
                        
                        # Vérifier qu'il y a des fonds disponibles
                        if position_size > 0:
                            # Créer la position
                            new_position = {
                                "symbol": symbol,
                                "entry_price": current_price,
                                "stop_loss": opportunity["stop_loss"],
                                "take_profit": opportunity["take_profit"],
                                "side": opportunity["side"],
                                "size": position_size,
                                "entry_date": current_date,
                                "score": opportunity["score"],
                                "leverage": LEVERAGE
                            }
                            
                            # Ajouter la position
                            position_tracker.add_position(new_position)
            
            # 3. Mettre à jour l'equity curve
            equity_curve.append(current_equity)
            timestamps.append(current_date)
        
        # 5. Calculer les métriques de performance (même code que pour le LSTM)
        closed_positions = position_tracker.get_closed_positions()
        total_trades = len(closed_positions)
        
        # Séparer les trades gagnants et perdants
        winning_trades = [t for t in closed_positions if t["profit_pct"] > 0]
        losing_trades = [t for t in closed_positions if t["profit_pct"] <= 0]
        
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
        
        # Calcul du profit moyen par trade
        avg_profit = np.mean([t["profit_pct"] for t in closed_positions]) if closed_positions else 0
        avg_win = np.mean([t["profit_pct"] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t["profit_pct"] for t in losing_trades]) if losing_trades else 0
        
        # Calcul du ratio profit/perte
        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 and winning_trades and losing_trades else 0
        
        # Calcul du drawdown maximum
        drawdowns = []
        peak = equity_curve[0]
        
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak * 100
            drawdowns.append(drawdown)
        
        max_drawdown = max(drawdowns)
        
        # Calcul du Sharpe ratio
        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe_ratio = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
        
        # Assembler les résultats
        results = {
            "initial_capital": initial_capital,
            "final_equity": equity_curve[-1],
            "return_pct": (equity_curve[-1] / initial_capital - 1) * 100,
            "num_trades": total_trades,
            "win_rate": win_rate,
            "avg_profit": avg_profit,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_loss_ratio": profit_loss_ratio,
            "max_drawdown_pct": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "equity_curve": equity_curve,
            "timestamps": [str(ts) for ts in timestamps],
            "trades": trades
        }
        
        logger.info(f"Backtest baseline terminé: {total_trades} trades, win rate: {win_rate:.2f}%, retour: {results['return_pct']:.2f}%")
        
        return results
    
    def _compare_strategies(self, lstm_results: Dict, baseline_results: Dict) -> Dict:
        """
        Compare les résultats des deux stratégies
        
        Args:
            lstm_results: Résultats du backtest LSTM
            baseline_results: Résultats du backtest de la stratégie de base
            
        Returns:
            Comparaison des stratégies
        """
        # Calculer les différences
        return_difference = lstm_results["return_pct"] - baseline_results["return_pct"]
        trade_count_difference = lstm_results["num_trades"] - baseline_results["num_trades"]
        win_rate_difference = lstm_results["win_rate"] - baseline_results["win_rate"]
        drawdown_improvement = baseline_results["max_drawdown_pct"] - lstm_results["max_drawdown_pct"]
        sharpe_improvement = lstm_results["sharpe_ratio"] - baseline_results["sharpe_ratio"]
        
        # Calculer le ratio de surperformance
        if baseline_results["return_pct"] > 0:
            outperformance_ratio = lstm_results["return_pct"] / baseline_results["return_pct"]
        elif baseline_results["return_pct"] < 0 and lstm_results["return_pct"] > 0:
            outperformance_ratio = float('inf')  # Cas où baseline est négatif et LSTM positif
        elif baseline_results["return_pct"] < 0 and lstm_results["return_pct"] < 0:
            outperformance_ratio = baseline_results["return_pct"] / lstm_results["return_pct"]  # Ratio des pertes
        else:
            outperformance_ratio = 1.0  # Par défaut si baseline est 0
        
        # Assembler les résultats de comparaison
        comparison = {
            "return_difference": return_difference,
            "trade_count_difference": trade_count_difference,
            "win_rate_difference": win_rate_difference,
            "drawdown_improvement": drawdown_improvement,
            "sharpe_improvement": sharpe_improvement,
            "outperformance_ratio": outperformance_ratio,
            "conclusion": self._generate_conclusion(
                return_difference, 
                win_rate_difference, 
                drawdown_improvement, 
                sharpe_improvement
            )
        }
        
        return comparison
    
    def _generate_conclusion(self, return_diff: float, win_rate_diff: float, 
                           drawdown_improvement: float, sharpe_improvement: float) -> str:
        """
        Génère une conclusion textuelle basée sur la comparaison
        
        Args:
            return_diff: Différence de rendement
            win_rate_diff: Différence de taux de réussite
            drawdown_improvement: Amélioration du drawdown
            sharpe_improvement: Amélioration du ratio de Sharpe
            
        Returns:
            Conclusion textuelle
        """
        if return_diff > 5 and win_rate_diff > 0 and drawdown_improvement > 0 and sharpe_improvement > 0.2:
            return "La stratégie hybride LSTM surperforme nettement la stratégie de base sur tous les indicateurs clés."
        elif return_diff > 0 and (win_rate_diff > 0 or drawdown_improvement > 0 or sharpe_improvement > 0):
            return "La stratégie hybride LSTM offre des performances supérieures à la stratégie de base."
        elif return_diff < 0 and win_rate_diff < 0 and sharpe_improvement < 0:
            return "La stratégie de base surperforme la stratégie hybride LSTM dans ce backtest."
        else:
            # Résultats mitigés
            advantages = []
            disadvantages = []
            
            if return_diff > 0:
                advantages.append(f"meilleur rendement (+{return_diff:.2f}%)")
            elif return_diff < 0:
                disadvantages.append(f"rendement inférieur ({return_diff:.2f}%)")
                
            if win_rate_diff > 0:
                advantages.append(f"meilleur taux de réussite (+{win_rate_diff:.2f}%)")
            elif win_rate_diff < 0:
                disadvantages.append(f"taux de réussite inférieur ({win_rate_diff:.2f}%)")
                
            if drawdown_improvement > 0:
                advantages.append(f"drawdown réduit (+{drawdown_improvement:.2f}%)")
            elif drawdown_improvement < 0:
                disadvantages.append(f"drawdown supérieur ({-drawdown_improvement:.2f}%)")
                
            if sharpe_improvement > 0:
                advantages.append(f"meilleur ratio de Sharpe (+{sharpe_improvement:.2f})")
            elif sharpe_improvement < 0:
                disadvantages.append(f"ratio de Sharpe inférieur ({sharpe_improvement:.2f})")
                
            conclusion = "Résultats mitigés: "
            
            if advantages:
                conclusion += "La stratégie LSTM présente des avantages en termes de " + ", ".join(advantages)
                
                if disadvantages:
                    conclusion += ", mais des inconvénients en termes de " + ", ".join(disadvantages)
            else:
                conclusion += "La stratégie de base semble généralement plus performante."
                
            return conclusion
    
    def _generate_charts(self, lstm_results: Dict, baseline_results: Optional[Dict], symbol: str) -> None:
        """
        Génère des graphiques de comparaison des résultats
        
        Args:
            lstm_results: Résultats du backtest LSTM
            baseline_results: Résultats du backtest de la stratégie de base
            symbol: Symbole de la paire
        """
        # Créer le répertoire pour les graphiques
        charts_dir = os.path.join(self.output_dir, "charts")
        os.makedirs(charts_dir, exist_ok=True)
        
        # 1. Graphique des courbes d'équité
        plt.figure(figsize=(12, 6))
        
        # Convertir les timestamps en objets datetime
        lstm_timestamps = [pd.to_datetime(ts) for ts in lstm_results["timestamps"]]
        
        # Tracer la courbe d'équité LSTM
        plt.plot(lstm_timestamps, lstm_results["equity_curve"], 
                label=f"Stratégie Hybride LSTM: {lstm_results['return_pct']:.2f}%", 
                color='green', linewidth=2)
        
        # Tracer la courbe d'équité de la stratégie de base si disponible
        if baseline_results:
            baseline_timestamps = [pd.to_datetime(ts) for ts in baseline_results["timestamps"]]
            plt.plot(baseline_timestamps, baseline_results["equity_curve"], 
                    label=f"Stratégie de Base: {baseline_results['return_pct']:.2f}%", 
                    color='blue', linewidth=2)
        
        # Ajouter une ligne horizontale pour le capital initial
        plt.axhline(y=lstm_results["initial_capital"], color='r', linestyle='--', alpha=0.7,
                   label=f"Capital Initial: {lstm_results['initial_capital']} USDT")
        
        # Configurer le graphique
        plt.title(f"Comparaison des Courbes d'Équité - {symbol}")
        plt.xlabel("Date")
        plt.ylabel("Équité (USDT)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Ajouter des annotations pour les performances clés
        if baseline_results:
            diff = lstm_results["return_pct"] - baseline_results["return_pct"]
            diff_text = f"Différence: {diff:.2f}% {'↑' if diff > 0 else '↓'}"
            
            plt.annotate(diff_text, 
                        xy=(0.02, 0.05), 
                        xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Sauvegarder le graphique
        plt.savefig(os.path.join(charts_dir, f"{symbol}_equity_comparison.png"))
        plt.close()
        
        # 2. Graphique des drawdowns
        plt.figure(figsize=(12, 6))
        
        # Calculer les drawdowns pour chaque stratégie
        lstm_drawdowns = []
        lstm_peak = lstm_results["equity_curve"][0]
        
        for equity in lstm_results["equity_curve"]:
            if equity > lstm_peak:
                lstm_peak = equity
            drawdown = (lstm_peak - equity) / lstm_peak * 100
            lstm_drawdowns.append(drawdown)
        
        # Tracer les drawdowns LSTM
        plt.plot(lstm_timestamps, lstm_drawdowns, 
                label=f"Drawdown LSTM (Max: {lstm_results['max_drawdown_pct']:.2f}%)", 
                color='green', linewidth=2)
        
        # Tracer les drawdowns de la stratégie de base si disponible
        if baseline_results:
            baseline_drawdowns = []
            baseline_peak = baseline_results["equity_curve"][0]
            
            for equity in baseline_results["equity_curve"]:
                if equity > baseline_peak:
                    baseline_peak = equity
                drawdown = (baseline_peak - equity) / baseline_peak * 100
                baseline_drawdowns.append(drawdown)
            
            plt.plot(baseline_timestamps, baseline_drawdowns, 
                    label=f"Drawdown Base (Max: {baseline_results['max_drawdown_pct']:.2f}%)", 
                    color='blue', linewidth=2)
        
        # Configurer le graphique
        plt.title(f"Comparaison des Drawdowns - {symbol}")
        plt.xlabel("Date")
        plt.ylabel("Drawdown (%)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Inverser l'axe y pour une meilleure lisibilité (drawdowns négatifs vers le bas)
        plt.gca().invert_yaxis()
        
        # Sauvegarder le graphique
        plt.savefig(os.path.join(charts_dir, f"{symbol}_drawdown_comparison.png"))
        plt.close()
        
        # 3. Graphique de distribution des profits/pertes
        plt.figure(figsize=(12, 6))
        
        # Extraire les profits/pertes de chaque trade
        lstm_profits = [trade["profit_pct"] for trade in lstm_results["trades"]]
        
        if baseline_results:
            baseline_profits = [trade["profit_pct"] for trade in baseline_results["trades"]]
            
            # Créer un histogramme combiné
            plt.hist([lstm_profits, baseline_profits], bins=20, 
                    label=['Stratégie Hybride LSTM', 'Stratégie de Base'],
                    alpha=0.7, color=['green', 'blue'])
        else:
            # Histogramme simple pour LSTM uniquement
            plt.hist(lstm_profits, bins=20, label='Stratégie Hybride LSTM', alpha=0.7, color='green')
        
        # Ajouter une ligne verticale à 0%
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        
        # Configurer le graphique
        plt.title(f"Distribution des Profits/Pertes par Trade - {symbol}")
        plt.xlabel("Profit/Perte (%)")
        plt.ylabel("Nombre de Trades")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Sauvegarder le graphique
        plt.savefig(os.path.join(charts_dir, f"{symbol}_profit_distribution.png"))
        plt.close()
        
        # 4. Graphique des métriques de performance
        if baseline_results:
            # Créer un graphique à barres pour comparer les métriques clés
            plt.figure(figsize=(12, 6))
            
            metrics = ['Win Rate (%)', 'Avg Profit (%)', 'Sharpe Ratio', 'Max Drawdown (%)']
            lstm_values = [lstm_results["win_rate"], lstm_results["avg_profit"], 
                          lstm_results["sharpe_ratio"], lstm_results["max_drawdown_pct"]]
            baseline_values = [baseline_results["win_rate"], baseline_results["avg_profit"], 
                              baseline_results["sharpe_ratio"], baseline_results["max_drawdown_pct"]]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            plt.bar(x - width/2, lstm_values, width, label='Stratégie Hybride LSTM', color='green')
            plt.bar(x + width/2, baseline_values, width, label='Stratégie de Base', color='blue')
            
            plt.title(f"Comparaison des Métriques de Performance - {symbol}")
            plt.xticks(x, metrics)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Ajouter les valeurs au-dessus des barres
            for i, v in enumerate(lstm_values):
                plt.text(i - width/2, v + 0.1, f"{v:.2f}", ha='center', va='bottom', fontweight='bold')
                
            for i, v in enumerate(baseline_values):
                plt.text(i + width/2, v + 0.1, f"{v:.2f}", ha='center', va='bottom', fontweight='bold')
            
            # Sauvegarder le graphique
            plt.savefig(os.path.join(charts_dir, f"{symbol}_metrics_comparison.png"))
            plt.close()


def backtest_multiple_symbols(model_path: str, symbols: List[str], 
                           timeframe: str, start_date: str, end_date: str,
                           initial_capital: float = 200) -> Dict:
    """
    Effectue des backtests sur plusieurs symboles et compile les résultats
    
    Args:
        model_path: Chemin vers le modèle LSTM
        symbols: Liste des symboles à tester
        timeframe: Timeframe des données
        start_date: Date de début (YYYY-MM-DD)
        end_date: Date de fin (YYYY-MM-DD)
        initial_capital: Capital initial pour chaque backtest
        
    Returns:
        Résultats compilés des backtests
    """
    from utils.logger import setup_logger
    logger = setup_logger("multi_backtest")
    
    # Initialiser le backtester
    backtester = ModelBacktester(model_path=model_path)
    
    # Résultats par symbole
    results_by_symbol = {}
    
    # Statistiques globales
    total_lstm_return = 0
    total_baseline_return = 0
    best_symbol = None
    best_return = -float('inf')
    worst_symbol = None
    worst_return = float('inf')
    
    # Parcourir chaque symbole
    for symbol in symbols:
        logger.info(f"Backtest pour {symbol}...")
        
        # Charger les données
        data_path = os.path.join(DATA_DIR, "market_data", f"{symbol}_{timeframe}_{start_date}_{end_date}.csv")
        
        if not os.path.exists(data_path):
            logger.warning(f"Données non disponibles pour {symbol}")
            continue
        
        try:
            # Charger les données
            data = pd.read_csv(data_path, parse_dates=['timestamp'])
            data.set_index('timestamp', inplace=True)
            
            # Effectuer le backtest
            result = backtester.backtest_model(
                data=data,
                symbol=symbol,
                initial_capital=initial_capital,
                compare_with_baseline=True,
                plot_results=True
            )
            
            # Stocker les résultats
            results_by_symbol[symbol] = result
            
            # Mettre à jour les statistiques globales
            lstm_return = result["lstm"]["return_pct"]
            total_lstm_return += lstm_return
            
            if "baseline" in result:
                baseline_return = result["baseline"]["return_pct"]
                total_baseline_return += baseline_return
            
            # Vérifier si c'est le meilleur ou pire symbole
            if lstm_return > best_return:
                best_return = lstm_return
                best_symbol = symbol
            
            if lstm_return < worst_return:
                worst_return = lstm_return
                worst_symbol = symbol
                
        except Exception as e:
            logger.error(f"Erreur lors du backtest de {symbol}: {str(e)}")
    
    # Calculer les moyennes
    num_symbols = len(results_by_symbol)
    
    if num_symbols > 0:
        avg_lstm_return = total_lstm_return / num_symbols
        avg_baseline_return = total_baseline_return / num_symbols
        
        # Compilation des résultats globaux
        compiled_results = {
            "num_symbols_tested": num_symbols,
            "avg_lstm_return": avg_lstm_return,
            "avg_baseline_return": avg_baseline_return,
            "best_symbol": best_symbol,
            "best_return": best_return,
            "worst_symbol": worst_symbol,
            "worst_return": worst_return,
            "outperformance": avg_lstm_return - avg_baseline_return,
            "timeframe": timeframe,
            "period": f"{start_date} to {end_date}",
            "detailed_results": results_by_symbol
        }
        
        # Sauvegarder les résultats compilés
        output_path = os.path.join(
            DATA_DIR, 
            "backtest_results", 
            f"multi_backtest_{start_date}_{end_date}.json"
        )
        
        with open(output_path, 'w') as f:
            json.dump(compiled_results, f, indent=2, default=str)
        
        logger.info(f"Backtests multiples terminés. Résultats sauvegardés: {output_path}")
        
        return compiled_results
    else:
        logger.warning("Aucun backtest réussi")
        return {"error": "Aucun backtest réussi"}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Backtest du modèle LSTM")
    
    # Mode single ou multi
    parser.add_argument("--mode", type=str, choices=["single", "multi"], default="single",
                        help="Mode de backtest: single pour un symbole, multi pour plusieurs")
    
    # Arguments communs
    parser.add_argument("--model", type=str, required=True, help="Chemin vers le modèle LSTM")
    parser.add_argument("--timeframe", type=str, default="15m", help="Timeframe des données")
    parser.add_argument("--start", type=str, required=True, help="Date de début (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="Date de fin (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=200, help="Capital initial")
    
    # Arguments spécifiques au mode
    parser.add_argument("--symbol", type=str, help="Symbole à tester (mode single)")
    parser.add_argument("--symbols", type=str, help="Liste de symboles séparés par des virgules (mode multi)")
    
    args = parser.parse_args()
    
    if args.mode == "single":
        if not args.symbol:
            print("Erreur: L'argument --symbol est requis en mode single")
            exit(1)
        
        # Charger les données
        data_path = os.path.join(DATA_DIR, "market_data", f"{args.symbol}_{args.timeframe}_{args.start}_{args.end}.csv")
        
        if not os.path.exists(data_path):
            print(f"Erreur: Données non disponibles pour {args.symbol}")
            exit(1)
        
        data = pd.read_csv(data_path, parse_dates=['timestamp'])
        data.set_index('timestamp', inplace=True)
        
        # Effectuer le backtest
        backtester = ModelBacktester(model_path=args.model)
        backtester.backtest_model(
            data=data,
            symbol=args.symbol,
            initial_capital=args.capital,
            compare_with_baseline=True,
            plot_results=True
        )
    
    elif args.mode == "multi":
        if not args.symbols:
            print("Erreur: L'argument --symbols est requis en mode multi")
            exit(1)
        
        symbols = [s.strip() for s in args.symbols.split(',')]
        
        backtest_multiple_symbols(
            model_path=args.model,
            symbols=symbols,
            timeframe=args.timeframe,
            start_date=args.start,
            end_date=args.end,
            initial_capital=args.capital
        )