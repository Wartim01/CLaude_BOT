# utils/backtest_engine.py
"""
Moteur de backtest avancé pour stratégies de trading
Permet de simuler des stratégies sur des données historiques avec une approche réaliste
"""
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta

from config.config import DATA_DIR
from config.trading_params import (
    RISK_PER_TRADE_PERCENT, 
    STOP_LOSS_PERCENT,
    TAKE_PROFIT_PERCENT,
    LEVERAGE,
    MINIMUM_SCORE_TO_TRADE
)
from utils.logger import setup_logger

logger = setup_logger("backtest_engine")

class BacktestEngine:
    """
    Moteur de backtest complet pour les stratégies de trading
    Permet de simuler des stratégies sur des données historiques avec prise en compte:
    - Des frais de trading
    - Du slippage
    - De la gestion du risque
    - Des stratégies de sortie dynamiques
    """
    def __init__(self, data_dir: str = None, fee_rate: float = 0.0750):
        """
        Initialise le moteur de backtest
        
        Args:
            data_dir: Répertoire des données
            fee_rate: Taux de frais de trading (en pourcentage)
        """
        self.data_dir = data_dir or os.path.join(DATA_DIR, "market_data")
        self.results_dir = os.path.join(DATA_DIR, "backtest_results")
        self.fee_rate = fee_rate / 100  # Convertir le pourcentage en décimal
        
        # Créer les répertoires si nécessaires
        for directory in [self.data_dir, self.results_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Métriques de performance
        self.metrics = None
    
    def load_data(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Charge les données historiques pour le backtest
        
        Args:
            symbol: Paire de trading
            timeframe: Intervalle de temps
            start_date: Date de début (YYYY-MM-DD)
            end_date: Date de fin (YYYY-MM-DD)
            
        Returns:
            DataFrame avec les données OHLCV
        """
        # Construire le chemin du fichier
        filename = f"{symbol}_{timeframe}_{start_date}_{end_date}.csv"
        filepath = os.path.join(self.data_dir, filename)
        
        # Vérifier si le fichier existe déjà
        if os.path.exists(filepath):
            logger.info(f"Chargement des données depuis {filepath}")
            df = pd.read_csv(filepath, parse_dates=['timestamp'])
            df.set_index('timestamp', inplace=True)
            return df
        
        # Si le fichier n'existe pas, essayer d'autres formats de nom possibles
        alternative_files = [
            f for f in os.listdir(self.data_dir)
            if f.startswith(f"{symbol}_{timeframe}") and f.endswith(".csv")
        ]
        
        if alternative_files:
            logger.info(f"Fichier exact non trouvé, utilisation de {alternative_files[0]}")
            df = pd.read_csv(os.path.join(self.data_dir, alternative_files[0]), parse_dates=['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Filtrer les données selon la plage de dates
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]
            
            return df
        
        logger.error(f"Fichier de données non trouvé: {filepath}")
        return pd.DataFrame()
    
    def run_backtest(self, symbol: str, timeframe: str, start_date: str, end_date: str,
                   initial_capital: float = 200, strategy_name: str = "technical_bounce",
                   risk_per_trade: float = None, stop_loss_percent: float = None,
                   take_profit_percent: float = None, leverage: float = None,
                   trailing_stop: bool = False, preserve_capital: bool = True) -> Dict:
        """
        Exécute un backtest sur la période spécifiée
        
        Args:
            symbol: Paire de trading
            timeframe: Intervalle de temps
            start_date: Date de début (YYYY-MM-DD)
            end_date: Date de fin (YYYY-MM-DD)
            initial_capital: Capital initial (USDT)
            strategy_name: Nom de la stratégie
            risk_per_trade: Pourcentage de risque par trade (None = utiliser valeur par défaut)
            stop_loss_percent: Pourcentage de stop loss (None = utiliser valeur par défaut)
            take_profit_percent: Pourcentage de take profit (None = utiliser valeur par défaut)
            leverage: Effet de levier (None = utiliser valeur par défaut)
            trailing_stop: Activer les stops suiveurs
            preserve_capital: Réduire le risque après des pertes pour préserver le capital
            
        Returns:
            Résultats du backtest
        """
        # Charger les données
        data = self.load_data(symbol, timeframe, start_date, end_date)
        
        if data.empty:
            return {
                "success": False,
                "message": "Données non disponibles pour le backtest"
            }
        
        # Utiliser les paramètres spécifiés ou les valeurs par défaut
        risk_per_trade = risk_per_trade if risk_per_trade is not None else RISK_PER_TRADE_PERCENT
        stop_loss_percent = stop_loss_percent if stop_loss_percent is not None else STOP_LOSS_PERCENT
        take_profit_percent = take_profit_percent if take_profit_percent is not None else TAKE_PROFIT_PERCENT
        leverage = leverage if leverage is not None else LEVERAGE
        
        # Configurer les paramètres de backtest
        backtest_params = {
            "risk_per_trade": risk_per_trade,
            "stop_loss_percent": stop_loss_percent,
            "take_profit_percent": take_profit_percent,
            "leverage": leverage,
            "trailing_stop": trailing_stop,
            "fee_rate": self.fee_rate,
            "preserve_capital": preserve_capital
        }
        
        # Sélectionner la stratégie
        if strategy_name == "technical_bounce":
            from strategies.technical_bounce import TechnicalBounceStrategy
            from ai.scoring_engine import ScoringEngine
            
            # Créer les composants nécessaires
            scoring_engine = ScoringEngine()
            strategy = self._create_technical_bounce_strategy(scoring_engine)
            
        elif strategy_name == "hybrid":
            from strategies.hybrid_strategy import HybridStrategy
            from ai.scoring_engine import ScoringEngine
            from ai.models.lstm_model import LSTMModel
            from core.adaptive_risk_manager import AdaptiveRiskManager
            
            # Créer les composants nécessaires
            scoring_engine = ScoringEngine()
            
            # Charger le modèle LSTM
            model_path = os.path.join(DATA_DIR, "models", "production", "lstm_final.h5")
            if not os.path.exists(model_path):
                logger.error(f"Modèle LSTM non trouvé: {model_path}")
                return {
                    "success": False,
                    "message": "Modèle LSTM requis pour la stratégie hybride"
                }
            
            lstm_model = LSTMModel()
            lstm_model.load(model_path)
            
            adaptive_risk_manager = AdaptiveRiskManager(initial_capital=initial_capital)
            
            strategy = self._create_hybrid_strategy(
                scoring_engine=scoring_engine,
                lstm_model=lstm_model,
                adaptive_risk_manager=adaptive_risk_manager
            )
        else:
            return {
                "success": False,
                "message": f"Stratégie non reconnue: {strategy_name}"
            }
        
        # Simuler le trading
        backtest_results = self._simulate_trading(
            data=data,
            strategy=strategy,
            strategy_name=strategy_name,
            initial_capital=initial_capital,
            symbol=symbol,
            backtest_params=backtest_params
        )
        
        # Calculer des métriques supplémentaires
        self._calculate_performance_metrics(backtest_results)
        
        # Sauvegarder les résultats
        self._save_backtest_results(backtest_results, symbol, strategy_name, start_date, end_date)
        
        return backtest_results
    
    def _create_technical_bounce_strategy(self, scoring_engine):
        """
        Crée une instance de la stratégie de rebond technique
        
        Args:
            scoring_engine: Instance du moteur de scoring
            
        Returns:
            Instance de la stratégie
        """
        from strategies.technical_bounce import TechnicalBounceStrategy
        
        # Créer un data fetcher simulé
        class MockDataFetcher:
            def __init__(self, backtest_data=None):
                self.backtest_data = backtest_data
            
            def get_current_price(self, symbol):
                if self.backtest_data is None or self.backtest_data.empty:
                    return 0
                return self.backtest_data["close"].iloc[-1]
            
            def get_ohlcv(self, symbol, timeframe, limit=100):
                if self.backtest_data is None or self.backtest_data.empty:
                    return pd.DataFrame()
                return self.backtest_data.tail(limit)
            
            def get_market_data(self, symbol):
                """
                Simule la méthode get_market_data pour le backtest
                
                Args:
                    symbol: Paire de trading
                    
                Returns:
                    Dictionnaire avec les données de marché simulées
                """
                if self.backtest_data is None or self.backtest_data.empty:
                    return {
                        "symbol": symbol,
                        "current_price": 0,
                        "primary_timeframe": {"ohlcv": pd.DataFrame()},
                        "secondary_timeframes": {}
                    }
                
                # Calculer les indicateurs
                from indicators.trend import calculate_ema, calculate_adx
                from indicators.momentum import calculate_rsi
                from indicators.volatility import calculate_bollinger_bands, calculate_atr
                
                # Obtenir les 100 dernières lignes pour les calculs
                data = self.backtest_data.tail(100).copy()
                
                # Calculer les indicateurs
                ema = calculate_ema(data)
                rsi = calculate_rsi(data)
                bollinger = calculate_bollinger_bands(data)
                atr = calculate_atr(data)
                adx = calculate_adx(data)
                
                # Créer le dictionnaire de données de marché avec timeframes secondaires simulés
                market_data = {
                    "symbol": symbol,
                    "current_price": data["close"].iloc[-1],
                    "primary_timeframe": {
                        "ohlcv": data,
                        "indicators": {
                            "ema": ema,
                            "rsi": rsi,
                            "bollinger": bollinger,
                            "atr": atr,
                            "adx": adx
                        }
                    },
                    "secondary_timeframes": {
                        "1h": {
                            "ohlcv": data,  # Pour simplifier, on utilise les mêmes données
                            "indicators": {
                                "rsi": rsi,
                                "bollinger": bollinger
                            }
                        }
                    }
                }
                
                return market_data
            
            def detect_volume_spike(self, symbol):
                return {
                    "spike": False,
                    "ratio": 1.0,
                    "bullish": None,
                    "details": {}
                }
        
        # Créer un market analyzer simulé
        class MockMarketAnalyzer:
            def analyze_market_state(self, symbol):
                return {
                    "favorable": True,
                    "cooldown": False,
                    "details": {}
                }
        
        # Créer les instances
        mock_data_fetcher = MockDataFetcher()
        mock_market_analyzer = MockMarketAnalyzer()
        
        # Créer et retourner l'instance de la stratégie
        return TechnicalBounceStrategy(mock_data_fetcher, mock_market_analyzer, scoring_engine)
    
    def _create_hybrid_strategy(self, scoring_engine, lstm_model, adaptive_risk_manager):
        """
        Crée une instance de la stratégie hybride
        
        Args:
            scoring_engine: Instance du moteur de scoring
            lstm_model: Instance du modèle LSTM
            adaptive_risk_manager: Instance du gestionnaire de risque adaptatif
            
        Returns:
            Instance de la stratégie
        """
        from strategies.hybrid_strategy import HybridStrategy
        
        # Créer un data fetcher simulé (utiliser la même classe que pour technical_bounce)
        class MockDataFetcher:
            def __init__(self, backtest_data=None, featured_data=None):
                self.backtest_data = backtest_data
                self.featured_data = featured_data
            
            def get_current_price(self, symbol):
                if self.backtest_data is None or self.backtest_data.empty:
                    return 0
                return self.backtest_data["close"].iloc[-1]
            
            def get_ohlcv(self, symbol, timeframe, limit=100):
                if self.backtest_data is None or self.backtest_data.empty:
                    return pd.DataFrame()
                return self.backtest_data.tail(limit)
            
            def get_market_data(self, symbol):
                """
                Simule la méthode get_market_data pour le backtest
                
                Args:
                    symbol: Paire de trading
                    
                Returns:
                    Dictionnaire avec les données de marché simulées
                """
                if self.backtest_data is None or self.backtest_data.empty:
                    return {
                        "symbol": symbol,
                        "current_price": 0,
                        "primary_timeframe": {"ohlcv": pd.DataFrame()},
                        "secondary_timeframes": {}
                    }
                
                # Calculer les indicateurs
                from indicators.trend import calculate_ema, calculate_adx
                from indicators.momentum import calculate_rsi
                from indicators.volatility import calculate_bollinger_bands, calculate_atr
                
                # Obtenir les 100 dernières lignes pour les calculs
                data = self.backtest_data.tail(100).copy()
                
                # Calculer les indicateurs ou les récupérer depuis les données featured
                if self.featured_data is not None and len(self.featured_data) >= len(data):
                    featured_slice = self.featured_data.tail(100)
                    
                    # Extraire les indicateurs des données featured
                    indicators = {}
                    
                    # RSI
                    if 'rsi' in featured_slice.columns:
                        indicators['rsi'] = featured_slice['rsi']
                    else:
                        indicators['rsi'] = calculate_rsi(data)
                    
                    # Bandes de Bollinger
                    if all(col in featured_slice.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
                        indicators['bollinger'] = {
                            'upper': featured_slice['bb_upper'],
                            'middle': featured_slice['bb_middle'],
                            'lower': featured_slice['bb_lower']
                        }
                        
                        if 'bb_percent_b' in featured_slice.columns:
                            indicators['bollinger']['percent_b'] = featured_slice['bb_percent_b']
                    else:
                        indicators['bollinger'] = calculate_bollinger_bands(data)
                    
                    # EMA
                    ema_columns = [col for col in featured_slice.columns if col.startswith('ema_')]
                    if ema_columns:
                        indicators['ema'] = {col: featured_slice[col] for col in ema_columns}
                    else:
                        indicators['ema'] = calculate_ema(data)
                    
                    # ADX
                    if all(col in featured_slice.columns for col in ['adx', 'plus_di', 'minus_di']):
                        indicators['adx'] = {
                            'adx': featured_slice['adx'],
                            'plus_di': featured_slice['plus_di'],
                            'minus_di': featured_slice['minus_di']
                        }
                    else:
                        indicators['adx'] = calculate_adx(data)
                    
                    # ATR
                    if 'atr' in featured_slice.columns:
                        indicators['atr'] = featured_slice['atr']
                    else:
                        indicators['atr'] = calculate_atr(data)
                else:
                    # Calculer les indicateurs classiques
                    indicators = {
                        'rsi': calculate_rsi(data),
                        'bollinger': calculate_bollinger_bands(data),
                        'ema': calculate_ema(data),
                        'adx': calculate_adx(data),
                        'atr': calculate_atr(data)
                    }
                
                # Créer le dictionnaire de données de marché
                market_data = {
                    "symbol": symbol,
                    "current_price": data["close"].iloc[-1],
                    "primary_timeframe": {
                        "ohlcv": data,
                        "indicators": indicators
                    },
                    "secondary_timeframes": {}  # Simplifié pour le backtest
                }
                
                return market_data
        
        # Créer un market analyzer simulé
        class MockMarketAnalyzer:
            def analyze_market_state(self, symbol):
                return {
                    "favorable": True,
                    "cooldown": False,
                    "details": {}
                }
        
        # Créer les instances
        mock_data_fetcher = MockDataFetcher()
        mock_market_analyzer = MockMarketAnalyzer()
        
        # Créer et retourner l'instance de la stratégie
        return HybridStrategy(mock_data_fetcher, mock_market_analyzer, 
                             scoring_engine, lstm_model, adaptive_risk_manager)
    
    def _calculate_performance_metrics(self, backtest_results: Dict) -> None:
        """
        Calcule des métriques de performance avancées
        
        Args:
            backtest_results: Résultats du backtest
        """
        # Extraire les courbes d'équité et les trades
        equity_curve = backtest_results["equity_curve"]
        trades = backtest_results["trades"]
        
        # Calcul des rendements quotidiens
        daily_returns = []
        for i in range(1, len(equity_curve)):
            daily_return = (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
            daily_returns.append(daily_return)
        
        # Métriques avancées
        # 1. Sortino Ratio (comme Sharpe mais en ne considérant que les pertes)
        negative_returns = [r for r in daily_returns if r < 0]
        sortino_ratio = 0
        if negative_returns:
            downside_deviation = np.std(negative_returns) * np.sqrt(252)
            avg_return = np.mean(daily_returns) * 252
            sortino_ratio = avg_return / downside_deviation if downside_deviation > 0 else 0
        
        # 2. Calmar Ratio (rendement annualisé / drawdown max)
        calmar_ratio = 0
        if backtest_results["max_drawdown"] > 0:
            annualized_return = (1 + backtest_results["total_return"]/100) ** (252 / len(equity_curve)) - 1
            calmar_ratio = annualized_return / (backtest_results["max_drawdown"]/100)
        
        # 3. Métriques de consistance des trades
        if trades:
            wins = [t["pnl_percent"] for t in trades if t["pnl_percent"] > 0]
            losses = [abs(t["pnl_percent"]) for t in trades if t["pnl_percent"] <= 0]
            
            win_std = np.std(wins) if wins else 0
            loss_std = np.std(losses) if losses else 0
            
            # Consistency score (ratio gain moyen / écart-type des gains)
            win_consistency = backtest_results["avg_win"] / win_std if win_std > 0 and wins else 0
            loss_consistency = backtest_results["avg_loss"] / loss_std if loss_std > 0 and losses else 0
            
            # Streaks (séquences gagnantes/perdantes)
            max_win_streak = 0
            current_win_streak = 0
            max_loss_streak = 0
            current_loss_streak = 0
            
            for trade in trades:
                if trade["pnl_percent"] > 0:
                    current_win_streak += 1
                    current_loss_streak = 0
                    max_win_streak = max(max_win_streak, current_win_streak)
                else:
                    current_loss_streak += 1
                    current_win_streak = 0
                    max_loss_streak = max(max_loss_streak, current_loss_streak)
            
            # Ajouter ces métriques aux résultats
            backtest_results["advanced_metrics"] = {
                "sortino_ratio": sortino_ratio,
                "calmar_ratio": calmar_ratio,
                "win_consistency": win_consistency,
                "loss_consistency": loss_consistency,
                "max_win_streak": max_win_streak,
                "max_loss_streak": max_loss_streak,
                "risk_of_ruin": self._calculate_risk_of_ruin(backtest_results)
            }
        
        # Stocker les métriques
        self.metrics = backtest_results.get("advanced_metrics", {})
    
    def _calculate_risk_of_ruin(self, backtest_results: Dict) -> float:
        """
        Calcule le risque de ruine (probabilité de perdre tout le capital)
        
        Args:
            backtest_results: Résultats du backtest
            
        Returns:
            Risque de ruine (0-1)
        """
        win_rate = backtest_results["win_rate"] / 100
        avg_win = backtest_results["avg_win"]
        avg_loss = -backtest_results["avg_loss"]  # Convertir en positif
        
        if win_rate == 0 or avg_win == 0 or avg_loss == 0:
            return 1.0  # Risque maximal
        
        # Calculer le ratio de gain (R)
        r_ratio = avg_win / avg_loss if avg_loss > 0 else 1
        
        # Calculer le ratio de probabilité (P)
        p_ratio = win_rate / (1 - win_rate) if win_rate < 1 else 1
        
        # Calculer le risque de ruine en utilisant la formule de la ruine du joueur
        if r_ratio * p_ratio <= 1:
            return 1.0  # Risque maximal
        
        risk_of_ruin = ((1 - (r_ratio * p_ratio)) / (1 + (r_ratio * p_ratio))) ** 20  # Exposant 20 = capital / taille position
        
        return min(1.0, max(0.0, risk_of_ruin))
    
    def _calculate_sharpe_ratio(self, equity_curve: List[float], risk_free_rate: float = 0.01) -> float:
        """
        Calcule le ratio de Sharpe
        
        Args:
            equity_curve: Liste des valeurs d'équité
            risk_free_rate: Taux sans risque annuel
            
        Returns:
            Ratio de Sharpe
        """
        # Calculer les rendements quotidiens
        daily_returns = []
        
        for i in range(1, len(equity_curve)):
            daily_return = (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
            daily_returns.append(daily_return)
        
        # Calculer la moyenne et l'écart-type des rendements
        if not daily_returns:
            return 0
        
        avg_return = sum(daily_returns) / len(daily_returns)
        std_return = np.std(daily_returns) if len(daily_returns) > 1 else 0
        
        # Annualiser les rendements (252 jours de trading par an)
        annual_return = avg_return * 252
        annual_std = std_return * np.sqrt(252)
        
        # Calculer le ratio de Sharpe
        if annual_std == 0:
            return 0
        
        sharpe_ratio = (annual_return - risk_free_rate) / annual_std
        
        return sharpe_ratio
    
    def _simulate_trading(self, data: pd.DataFrame, strategy, strategy_name: str,
                       initial_capital: float, symbol: str, backtest_params: Dict) -> Dict:
        """
        Simule le trading sur des données historiques
        
        Args:
            data: DataFrame avec les données OHLCV
            strategy: Stratégie de trading
            strategy_name: Nom de la stratégie
            initial_capital: Capital initial
            symbol: Paire de trading
            backtest_params: Paramètres de backtest
            
        Returns:
            Résultats de la simulation
        """
        # Extraire les paramètres de backtest
        risk_per_trade = backtest_params["risk_per_trade"]
        stop_loss_percent = backtest_params["stop_loss_percent"]
        take_profit_percent = backtest_params["take_profit_percent"]
        leverage = backtest_params["leverage"]
        trailing_stop = backtest_params["trailing_stop"]
        fee_rate = backtest_params["fee_rate"]
        preserve_capital = backtest_params["preserve_capital"]
        
        # Initialiser les variables de simulation
        equity = initial_capital
        position = None
        trades = []
        equity_curve = [initial_capital]
        dates = [data.index[0]]
        drawdowns = [0]
        current_drawdown = 0
        peak_equity = initial_capital
        risk_adjustment = 1.0  # Facteur d'ajustement du risque (réduit après des pertes)
        consecutive_losses = 0
        
        # Période de warmup pour les indicateurs (au moins 100 bougies)
        warmup_period = 100
        
        # Spécifique à la stratégie hybride
        featured_data = None
        normalized_data = None
        if strategy_name == "hybrid":
            # Créer les caractéristiques pour la stratégie hybride
            from ai.models.feature_engineering import FeatureEngineering
            feature_engineering = FeatureEngineering()
            
            # Créer les caractéristiques avancées
            featured_data = feature_engineering.create_features(
                data,
                include_time_features=True,
                include_price_patterns=True
            )
            
            # Normaliser les caractéristiques
            normalized_data = feature_engineering.scale_features(
                featured_data,
                is_training=False,
                method='standard',
                feature_group='lstm'
            )
            
            # Mettre à jour le data fetcher
            strategy.data_fetcher.featured_data = featured_data
        
        # Mettre à jour les données dans le data fetcher simulé
        strategy.data_fetcher.backtest_data = data.iloc[:warmup_period]
        
        # Simuler chaque jour de trading
        for i in range(warmup_period, len(data)):
            # Mettre à jour les données simulées (fenêtre glissante)
            current_data = data.iloc[i-warmup_period:i+1]
            strategy.data_fetcher.backtest_data = current_data
            
            current_date = current_data.index[-1]
            current_price = current_data["close"].iloc[-1]
            
            # Gérer les positions ouvertes
            if position:
                # Vérifier si le stop-loss est atteint
                if current_price <= position["stop_loss"]:
                    # Calculer le PnL (avec frais de trading)
                    entry_price = position["entry_price"]
                    position_size = position["size"]
                    actual_exit_price = position["stop_loss"] * (1 - fee_rate)  # Tenir compte du slippage et des frais
                    
                    pnl_percent = ((actual_exit_price - entry_price) / entry_price) * leverage * 100
                    pnl_amount = position_size * pnl_percent / 100
                    
                    # Mettre à jour l'équité
                    equity += pnl_amount
                    
                    # Enregistrer le trade
                    trade = {
                        "entry_date": position["entry_date"],
                        "exit_date": current_date,
                        "entry_price": entry_price,
                        "exit_price": actual_exit_price,
                        "size": position_size,
                        "pnl_percent": pnl_percent,
                        "pnl_amount": pnl_amount,
                        "exit_reason": "Stop Loss",
                        "fees_paid": position_size * fee_rate
                    }
                    trades.append(trade)
                    
                    # Réinitialiser la position
                    position = None
                    
                    # Mettre à jour les compteurs de pertes consécutives
                    if pnl_percent < 0:
                        consecutive_losses += 1
                        
                        # Réduire le risque après des pertes consécutives si preserve_capital est activé
                        if preserve_capital:
                            risk_adjustment = max(0.5, 1.0 - (consecutive_losses * 0.1))
                    else:
                        consecutive_losses = 0
                        risk_adjustment = 1.0
                
                # Vérifier si le take-profit est atteint
                elif current_price >= position["take_profit"]:
                    # Calculer le PnL (avec frais de trading)
                    entry_price = position["entry_price"]
                    position_size = position["size"]
                    actual_exit_price = position["take_profit"] * (1 - fee_rate)  # Tenir compte du slippage et des frais
                    
                    pnl_percent = ((actual_exit_price - entry_price) / entry_price) * leverage * 100
                    pnl_amount = position_size * pnl_percent / 100
                    
                    # Mettre à jour l'équité
                    equity += pnl_amount
                    
                    # Enregistrer le trade
                    trade = {
                        "entry_date": position["entry_date"],
                        "exit_date": current_date,
                        "entry_price": entry_price,
                        "exit_price": actual_exit_price,
                        "size": position_size,
                        "pnl_percent": pnl_percent,
                        "pnl_amount": pnl_amount,
                        "exit_reason": "Take Profit",
                        "fees_paid": position_size * fee_rate
                    }
                    trades.append(trade)
                    
                    # Réinitialiser la position
                    position = None
                    
                    # Réinitialiser les compteurs de pertes consécutives
                    consecutive_losses = 0
                    risk_adjustment = 1.0
                
                # Stratégie de sortie dynamique (pour la stratégie hybride)
                elif strategy_name == "hybrid" and position:
                    # Vérifier si on doit fermer la position de manière anticipée
                    early_close = strategy.should_close_early(
                        symbol=symbol,
                        position=position,
                        current_price=current_price
                    )
                    
                    if early_close.get("should_close", False):
                        # Calculer le PnL (avec frais de trading)
                        entry_price = position["entry_price"]
                        position_size = position["size"]
                        actual_exit_price = current_price * (1 - fee_rate)  # Tenir compte des frais
                        
                        pnl_percent = ((actual_exit_price - entry_price) / entry_price) * leverage * 100
                        pnl_amount = position_size * pnl_percent / 100
                        
                        # Mettre à jour l'équité
                        equity += pnl_amount
                        
                        # Enregistrer le trade
                        trade = {
                            "entry_date": position["entry_date"],
                            "exit_date": current_date,
                            "entry_price": entry_price,
                            "exit_price": actual_exit_price,
                            "size": position_size,
                            "pnl_percent": pnl_percent,
                            "pnl_amount": pnl_amount,
                            "exit_reason": f"Signal LSTM: {early_close.get('reason', 'N/A')}",
                            "fees_paid": position_size * fee_rate
                        }
                        trades.append(trade)
                        
                        # Réinitialiser la position
                        position = None
                        
                        # Mettre à jour les compteurs de pertes consécutives
                        if pnl_percent < 0:
                            consecutive_losses += 1
                            
                            # Réduire le risque après des pertes consécutives
                            if preserve_capital:
                                risk_adjustment = max(0.5, 1.0 - (consecutive_losses * 0.1))
                        else:
                            consecutive_losses = 0
                            risk_adjustment = 1.0
                
                # Trailing stop (si activé)
                elif trailing_stop and position and current_price > position["entry_price"]:
                    # Calculer la distance du trailing stop (en pourcentage du profit actuel)
                    trail_percent = stop_loss_percent * 0.75  # 75% du stop-loss initial
                    
                    # Calculer le nouveau stop-loss potentiel
                    current_profit_percent = (current_price - position["entry_price"]) / position["entry_price"] * 100
                    if current_profit_percent > trail_percent * 2:  # Activer uniquement si profit > 2 x trail
                        new_stop = current_price * (1 - trail_percent / 100)
                        
                        # Mettre à jour le stop-loss si le nouveau est plus élevé que l'actuel
                        if new_stop > position["stop_loss"]:
                            position["stop_loss"] = new_stop
            
            # Chercher de nouvelles opportunités si aucune position n'est ouverte
            if not position:
                opportunity = strategy.find_trading_opportunity(symbol)
                
                if opportunity and opportunity["score"] >= MINIMUM_SCORE_TO_TRADE:
                    # Calculer le capital risqué (taille de position)
                    risk_amount = equity * (risk_per_trade / 100) * risk_adjustment
                    position_size = risk_amount / (stop_loss_percent / 100) * leverage
                    
                    # Position size ne peut pas dépasser l'équité disponible
                    position_size = min(position_size, equity)
                    
                    # Appliquer les frais d'entrée
                    entry_fee = position_size * fee_rate
                    position_size -= entry_fee
                    
                    # Créer une nouvelle position
                    position = {
                        "entry_date": current_date,
                        "entry_price": current_price,
                        "stop_loss": current_price * (1 - stop_loss_percent / 100),
                        "take_profit": current_price * (1 + take_profit_percent / 100),
                        "size": position_size,
                        "leverage": leverage,
                        "score": opportunity["score"]
                    }
            
            # Mettre à jour l'equity curve
            equity_curve.append(equity)
            dates.append(current_date)
            
            # Calculer le drawdown
            if equity > peak_equity:
                peak_equity = equity
            
            current_drawdown = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
            drawdowns.append(current_drawdown)
        
        # Clôturer la position à la fin de la simulation si nécessaire
        if position:
            # Calculer le PnL (avec frais de trading)
            entry_price = position["entry_price"]
            position_size = position["size"]
            actual_exit_price = data["close"].iloc[-1] * (1 - fee_rate)  # Tenir compte des frais
            
            pnl_percent = ((actual_exit_price - entry_price) / entry_price) * leverage * 100
            pnl_amount = position_size * pnl_percent / 100
            
            # Mettre à jour l'équité
            equity += pnl_amount
            
            # Enregistrer le trade
            trade = {
                "entry_date": position["entry_date"],
                "exit_date": dates[-1],
                "entry_price": entry_price,
                "exit_price": actual_exit_price,
                "size": position_size,
                "pnl_percent": pnl_percent,
                "pnl_amount": pnl_amount,
                "exit_reason": "Fin de simulation",
                "fees_paid": position_size * fee_rate
            }
            trades.append(trade)
            
            # Mettre à jour le dernier point de l'equity curve
            equity_curve[-1] = equity
        
        # Calculer les statistiques du backtest
        total_trades = len(trades)
        winning_trades = [t for t in trades if t["pnl_percent"] > 0]
        losing_trades = [t for t in trades if t["pnl_percent"] <= 0]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        avg_win = sum(t["pnl_percent"] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t["pnl_percent"] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        profit_factor = abs(sum(t["pnl_percent"] for t in winning_trades)) / abs(sum(t["pnl_percent"] for t in losing_trades)) if losing_trades and sum(t["pnl_percent"] for t in losing_trades) != 0 else float('inf')
        
        max_drawdown = max(drawdowns) if drawdowns else 0
        sharpe_ratio = self._calculate_sharpe_ratio(equity_curve)
        
        # Calculer le rendement total
        total_return = (equity - initial_capital) / initial_capital * 100
        
        # Calculer les frais totaux
        total_fees = sum(t.get("fees_paid", 0) for t in trades)
        
        # Préparer les résultats
        results = {
            "success": True,
            "symbol": symbol,
            "strategy": strategy_name,
            "params": backtest_params,
            "initial_capital": initial_capital,
            "final_equity": equity,
            "total_return": total_return,
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "total_fees": total_fees,
            "trades": trades,
            "equity_curve": equity_curve,
            "dates": [str(d) for d in dates],
            "drawdowns": drawdowns
        }
        
        return results