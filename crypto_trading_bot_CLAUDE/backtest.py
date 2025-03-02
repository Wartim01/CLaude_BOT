# backtest.py
"""
Script de backtest pour la stratégie de trading
"""
import os
import json
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta

from config.config import DATA_DIR
from config.trading_params import (
    RISK_PER_TRADE_PERCENT, 
    STOP_LOSS_PERCENT,
    TAKE_PROFIT_PERCENT,
    LEVERAGE,
    MINIMUM_SCORE_TO_TRADE
)

from strategies.technical_bounce import TechnicalBounceStrategy
from ai.scoring_engine import ScoringEngine
from utils.logger import setup_logger

logger = setup_logger("backtest")

class BacktestEngine:
    """
    Moteur de backtest pour les stratégies de trading
    """
    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.join(DATA_DIR, "market_data")
        self.results_dir = os.path.join(DATA_DIR, "backtest_results")
        
        # Créer les répertoires si nécessaires
        for directory in [self.data_dir, self.results_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Initialiser les composants
        self.scoring_engine = ScoringEngine()
    
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
        
        # Si le fichier n'existe pas, vous pouvez implémenter la récupération des données
        # depuis une API externe (Binance, etc.)
        logger.error(f"Fichier de données non trouvé: {filepath}")
        return pd.DataFrame()
    
    def run_backtest(self, symbol: str, timeframe: str, start_date: str, end_date: str,
                   initial_capital: float = 200, strategy_name: str = "technical_bounce") -> Dict:
        """
        Exécute un backtest sur la période spécifiée
        
        Args:
            symbol: Paire de trading
            timeframe: Intervalle de temps
            start_date: Date de début (YYYY-MM-DD)
            end_date: Date de fin (YYYY-MM-DD)
            initial_capital: Capital initial (USDT)
            strategy_name: Nom de la stratégie
            
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
        
        # Sélectionner la stratégie
        if strategy_name == "technical_bounce":
            strategy = self._create_technical_bounce_strategy()
        else:
            return {
                "success": False,
                "message": f"Stratégie non reconnue: {strategy_name}"
            }
        
        # Simuler le trading
        backtest_results = self._simulate_trading(data, strategy, initial_capital, symbol)
        
        # Sauvegarder les résultats
        self._save_backtest_results(backtest_results, symbol, strategy_name, start_date, end_date)
        
        return backtest_results
    
    def _create_technical_bounce_strategy(self) -> TechnicalBounceStrategy:
        """
        Crée une instance de la stratégie de rebond technique
        
        Returns:
            Instance de la stratégie
        """
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
        return TechnicalBounceStrategy(mock_data_fetcher, mock_market_analyzer, self.scoring_engine)
    
    def _simulate_trading(self, data: pd.DataFrame, strategy, initial_capital: float, symbol: str) -> Dict:
        """
        Simule le trading sur des données historiques
        
        Args:
            data: DataFrame avec les données OHLCV
            strategy: Stratégie de trading
            initial_capital: Capital initial
            symbol: Paire de trading
            
        Returns:
            Résultats de la simulation
        """
        # Initialiser les variables de simulation
        equity = initial_capital
        position = None
        trades = []
        equity_curve = [initial_capital]
        dates = [data.index[0]]
        
        # Mettre à jour les données dans le data fetcher simulé
        strategy.data_fetcher.backtest_data = data.iloc[:50]  # Commencer avec les 50 premières lignes
        
        # Simuler chaque jour de trading
        for i in range(51, len(data)):
            # Mettre à jour les données simulées (fenêtre glissante)
            current_data = data.iloc[i-50:i]
            strategy.data_fetcher.backtest_data = current_data
            
            current_price = current_data["close"].iloc[-1]
            current_date = current_data.index[-1]
            
            # Gérer les positions ouvertes
            if position:
                # Vérifier si le stop-loss est atteint
                if current_price <= position["stop_loss"]:
                    pnl = (current_price - position["entry_price"]) / position["entry_price"] * 100 * LEVERAGE
                    equity = equity * (1 + pnl/100)
                    
                    trades.append({
                        "entry_date": position["entry_date"],
                        "exit_date": current_date,
                        "entry_price": position["entry_price"],
                        "exit_price": current_price,
                        "pnl_percent": pnl,
                        "exit_reason": "Stop-Loss"
                    })
                    
                    position = None
                
                # Vérifier si le take-profit est atteint
                elif current_price >= position["take_profit"]:
                    pnl = (current_price - position["entry_price"]) / position["entry_price"] * 100 * LEVERAGE
                    equity = equity * (1 + pnl/100)
                    
                    trades.append({
                        "entry_date": position["entry_date"],
                        "exit_date": current_date,
                        "entry_price": position["entry_price"],
                        "exit_price": current_price,
                        "pnl_percent": pnl,
                        "exit_reason": "Take-Profit"
                    })
                    
                    position = None
            
            # Chercher de nouvelles opportunités si aucune position n'est ouverte
            if not position:
                opportunity = strategy.find_trading_opportunity(symbol)
                
                if opportunity and opportunity["score"] >= strategy.min_score:
                    # Calculer la taille de position
                    position_size = equity * (RISK_PER_TRADE_PERCENT/100) / (STOP_LOSS_PERCENT/100) * LEVERAGE
                    
                    # Ouvrir une position
                    position = {
                        "entry_date": current_date,
                        "entry_price": current_price,
                        "stop_loss": current_price * (1 - STOP_LOSS_PERCENT/100),
                        "take_profit": current_price * (1 + TAKE_PROFIT_PERCENT/100),
                        "size": position_size,
                        "score": opportunity["score"]
                    }
            
            # Enregistrer l'équité
            equity_curve.append(equity)
            dates.append(current_date)
        
        # Clôturer la position à la fin de la simulation si nécessaire
        if position:
            final_price = data["close"].iloc[-1]
            pnl = (final_price - position["entry_price"]) / position["entry_price"] * 100 * LEVERAGE
            equity = equity * (1 + pnl/100)
            
            trades.append({
                "entry_date": position["entry_date"],
                "exit_date": data.index[-1],
                "entry_price": position["entry_price"],
                "exit_price": final_price,
                "pnl_percent": pnl,
                "exit_reason": "Fin de simulation"
            })
            
            equity_curve[-1] = equity
        
        # Calculer les statistiques du backtest
        total_trades = len(trades)
        winning_trades = [t for t in trades if t["pnl_percent"] > 0]
        losing_trades = [t for t in trades if t["pnl_percent"] <= 0]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        avg_win = sum(t["pnl_percent"] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t["pnl_percent"] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        profit_factor = abs(sum(t["pnl_percent"] for t in winning_trades)) / abs(sum(t["pnl_percent"] for t in losing_trades)) if losing_trades and sum(t["pnl_percent"] for t in losing_trades) != 0 else float('inf')
        
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        sharpe_ratio = self._calculate_sharpe_ratio(equity_curve)
        
        # Préparer les résultats
        results = {
            "success": True,
            "symbol": symbol,
            "initial_capital": initial_capital,
            "final_equity": equity,
            "total_return": (equity - initial_capital) / initial_capital * 100,
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "trades": trades,
            "equity_curve": equity_curve,
            "dates": [str(d) for d in dates]
        }
        
        return results
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """
        Calcule le drawdown maximum
        
        Args:
            equity_curve: Liste des valeurs d'équité
            
        Returns:
            Drawdown maximum en pourcentage
        """
        max_dd = 0
        peak = equity_curve[0]
        
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            
            dd = (peak - equity) / peak * 100
            max_dd = max(max_dd, dd)
        
        return max_dd
    
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
    
    def _save_backtest_results(self, results: Dict, symbol: str, strategy_name: str, 
                             start_date: str, end_date: str) -> None:
        """
        Sauvegarde les résultats du backtest
        
        Args:
            results: Résultats du backtest
            symbol: Paire de trading
            strategy_name: Nom de la stratégie
            start_date: Date de début
            end_date: Date de fin
        """
        if not results.get("success", False):
            return
        
        # Créer le nom du fichier
        filename = f"{symbol}_{strategy_name}_{start_date}_{end_date}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        # Sauvegarder les résultats
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Résultats du backtest sauvegardés: {filepath}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des résultats: {str(e)}")
        
        # Générer les graphiques
        self._generate_backtest_charts(results, symbol, strategy_name, start_date, end_date)
    
    def _generate_backtest_charts(self, results: Dict, symbol: str, strategy_name: str,
                                start_date: str, end_date: str) -> None:
        """
        Génère des graphiques pour les résultats du backtest
        
        Args:
            results: Résultats du backtest
            symbol: Paire de trading
            strategy_name: Nom de la stratégie
            start_date: Date de début
            end_date: Date de fin
        """
        if not results.get("success", False):
            return
        
        # Créer le répertoire pour les graphiques
        charts_dir = os.path.join(self.results_dir, "charts")
        if not os.path.exists(charts_dir):
            os.makedirs(charts_dir)
        
        # Créer le nom de base pour les fichiers
        base_filename = f"{symbol}_{strategy_name}_{start_date}_{end_date}"
        
        # 1. Graphique de la courbe d'équité
        plt.figure(figsize=(12, 6))
        plt.plot(results["equity_curve"])
        plt.title(f"Courbe d'Équité - {symbol} ({start_date} à {end_date})")
        plt.xlabel("Jours")
        plt.ylabel("Équité (USDT)")
        plt.grid(True)
        
        # Ajouter des annotations
        plt.annotate(f"Rendement total: {results['total_return']:.2f}%\n"
                    f"Drawdown max: {results['max_drawdown']:.2f}%\n"
                    f"Ratio de Sharpe: {results['sharpe_ratio']:.2f}",
                    xy=(0.02, 0.95),
                    xycoords='axes fraction',
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Sauvegarder le graphique
        equity_chart_path = os.path.join(charts_dir, f"{base_filename}_equity.png")
        plt.savefig(equity_chart_path)
        plt.close()
        
        # 2. Graphique de la distribution des profits/pertes
        pnl_values = [t["pnl_percent"] for t in results["trades"]]
        
        plt.figure(figsize=(10, 6))
        plt.hist(pnl_values, bins=20, alpha=0.7, color='skyblue')
        plt.title(f"Distribution des Profits/Pertes - {symbol}")
        plt.xlabel("Profit/Perte (%)")
        plt.ylabel("Fréquence")
        plt.axvline(x=0, color='r', linestyle='--')
        plt.grid(True, alpha=0.3)
        
        # Ajouter des annotations
        plt.annotate(f"Trades: {results['total_trades']}\n"
                    f"Win Rate: {results['win_rate']:.1f}%\n"
                    f"Gain moyen: {results['avg_win']:.2f}%\n"
                    f"Perte moyenne: {results['avg_loss']:.2f}%",
                    xy=(0.02, 0.95),
                    xycoords='axes fraction',
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Sauvegarder le graphique
        distribution_chart_path = os.path.join(charts_dir, f"{base_filename}_distribution.png")
        plt.savefig(distribution_chart_path)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest de stratégies de trading")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Paire de trading")
    parser.add_argument("--timeframe", type=str, default="15m", help="Intervalle de temps")
    parser.add_argument("--start", type=str, required=True, help="Date de début (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="Date de fin (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=200, help="Capital initial (USDT)")
    parser.add_argument("--strategy", type=str, default="technical_bounce", help="Stratégie de trading")
    
    args = parser.parse_args()
    
    # Exécuter le backtest
    engine = BacktestEngine()
    results = engine.run_backtest(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        strategy_name=args.strategy
    )
    
    if results.get("success", False):
        print(f"Backtest réussi pour {args.symbol} ({args.start} à {args.end})")
        print(f"Rendement total: {results['total_return']:.2f}%")
        print(f"Nombre de trades: {results['total_trades']}")
        print(f"Win rate: {results['win_rate']:.1f}%")
        print(f"Drawdown maximum: {results['max_drawdown']:.2f}%")
        print(f"Ratio de Sharpe: {results['sharpe_ratio']:.2f}")
    else:
        print(f"Échec du backtest: {results.get('message', 'Erreur inconnue')}")