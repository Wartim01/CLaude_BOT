# core/data_fetcher.py
"""
Récupération et traitement des données de marché
"""
import pandas as pd
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

from config.config import PRIMARY_TIMEFRAME, SECONDARY_TIMEFRAMES
from core.api_connector import BinanceConnector
from utils.logger import setup_logger

logger = setup_logger("data_fetcher")

class MarketDataFetcher:
    """
    Récupère et traite les données de marché depuis l'API Binance
    """
    def __init__(self, api_connector: BinanceConnector):
        self.api = api_connector
        self.data_cache = {}  # Cache pour les données OHLCV
        self.last_update = {}  # Dernière mise à jour des données
        self.cache_duration = 60  # Durée de validité du cache en secondes
        
    def get_current_price(self, symbol: str) -> float:
        """
        Récupère le prix actuel d'un symbole
        
        Args:
            symbol: Paire de trading
            
        Returns:
            Prix actuel
        """
        try:
            # Utilisation des trades récents pour obtenir le dernier prix
            recent_trades = self.api.get_recent_trades(symbol, limit=1)
            return float(recent_trades[0]['price'])
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du prix actuel pour {symbol}: {str(e)}")
            # Fallback: utiliser le dernier prix des données OHLCV
            ohlcv = self.get_ohlcv(symbol, PRIMARY_TIMEFRAME, limit=1)
            if not ohlcv.empty:
                return ohlcv['close'].iloc[-1]
            raise
    
    def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 100, 
             start_time: Optional[int] = None, end_time: Optional[int] = None) -> pd.DataFrame:
        """
        Récupère les données OHLCV (Open, High, Low, Close, Volume) avec cache adaptatif
        
        Args:
            symbol: Paire de trading
            timeframe: Intervalle de temps
            limit: Nombre de chandeliers à récupérer
            start_time: Timestamp de début (millisecondes)
            end_time: Timestamp de fin (millisecondes)
            
        Returns:
            DataFrame pandas avec les données OHLCV
        """
    
        cache_key = f"{symbol}_{timeframe}_{limit}_{start_time}_{end_time}"
        current_time = time.time()
        
        if timeframe in ["1m", "5m"]:
            cache_duration = 20 if self._is_market_volatile(symbol) else 60
        elif timeframe in ["15m", "30m"]:
            cache_duration = 60 if self._is_market_volatile(symbol) else 180
        else:  # Timeframes plus longs
            cache_duration = 120 if self._is_market_volatile(symbol) else 300
        # Déterminer si le marché est volatil pour adapter la durée du cache
        is_volatile = self._is_market_volatile(symbol)
        cache_duration = 30 if is_volatile else 90  # 30s si volatil, 90s sinon
        
        # Vérification du cache
        if (cache_key in self.data_cache and cache_key in self.last_update and
            current_time - self.last_update[cache_key] < cache_duration):
            return self.data_cache[cache_key]
        
        try:
            # Récupération des données depuis l'API
            klines = self.api.get_klines(symbol, timeframe, limit, start_time, end_time)
            
            # Conversion en DataFrame pandas
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Conversion des types de données
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            for col in ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # Définition de l'index
            df.set_index('timestamp', inplace=True)
            
            # Mise à jour du cache
            self.data_cache[cache_key] = df
            self.last_update[cache_key] = current_time
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données OHLCV pour {symbol} ({timeframe}): {str(e)}")
            
            # Si les données sont dans le cache, renvoyer les données du cache même si elles sont périmées
            if cache_key in self.data_cache:
                logger.info(f"Utilisation des données en cache périmées pour {symbol} ({timeframe})")
                return self.data_cache[cache_key]
            
            # Sinon, retourner un DataFrame vide
            columns = ['open', 'high', 'low', 'close', 'volume']
            return pd.DataFrame(columns=columns)

    
    def get_market_data(self, symbol: str, indicators: bool = True) -> Dict:
        """
        Récupère toutes les données de marché pertinentes pour un symbole
        
        Args:
            symbol: Paire de trading
            indicators: Indique si les indicateurs techniques doivent être calculés
            
        Returns:
            Dictionnaire contenant toutes les données de marché
        """
        data = {
            "symbol": symbol,
            "current_price": self.get_current_price(symbol),
            "primary_timeframe": {},
            "secondary_timeframes": {}
        }
        
        # Données du timeframe principal
        primary_data = self.get_ohlcv(symbol, PRIMARY_TIMEFRAME, limit=100)
        data["primary_timeframe"]["ohlcv"] = primary_data
        
        # Données des timeframes secondaires
        for tf in SECONDARY_TIMEFRAMES:
            secondary_data = self.get_ohlcv(symbol, tf, limit=100)
            data["secondary_timeframes"][tf] = {"ohlcv": secondary_data}
        
        # Calcul des indicateurs techniques si demandé
        if indicators:
            from indicators.trend import calculate_ema, calculate_adx
            from indicators.momentum import calculate_rsi
            from indicators.volatility import calculate_bollinger_bands, calculate_atr
            
            # Indicateurs pour le timeframe principal
            data["primary_timeframe"]["indicators"] = {
                "ema": calculate_ema(primary_data),
                "rsi": calculate_rsi(primary_data),
                "bollinger": calculate_bollinger_bands(primary_data),
                "atr": calculate_atr(primary_data),
                "adx": calculate_adx(primary_data)
            }
            
            # Indicateurs pour les timeframes secondaires
            for tf, tf_data in data["secondary_timeframes"].items():
                data["secondary_timeframes"][tf]["indicators"] = {
                    "ema": calculate_ema(tf_data["ohlcv"]),
                    "rsi": calculate_rsi(tf_data["ohlcv"]),
                    "adx": calculate_adx(tf_data["ohlcv"])
                }
        
        return data
    
    def get_order_book_analysis(self, symbol: str, depth: int = 20) -> Dict:
        """
        Analyse le carnet d'ordres pour déterminer la pression d'achat/vente
        
        Args:
            symbol: Paire de trading
            depth: Profondeur du carnet à analyser
            
        Returns:
            Analyse du carnet d'ordres
        """
        try:
            order_book = self.api.get_order_book(symbol, limit=depth)
            
            # Extraction des offres (asks) et des demandes (bids)
            bids = np.array([[float(price), float(qty)] for price, qty in order_book["bids"]])
            asks = np.array([[float(price), float(qty)] for price, qty in order_book["asks"]])
            
            # Calcul de la pression d'achat/vente
            bid_volume = np.sum(bids[:, 1])
            ask_volume = np.sum(asks[:, 1])
            
            # Calcul des murs d'achat/vente (concentrations importantes d'ordres)
            bid_walls = []
            ask_walls = []
            
            # Seuil pour considérer un niveau comme un mur (% du volume total)
            wall_threshold = 0.15
            
            for price, qty in bids:
                if qty / bid_volume > wall_threshold:
                    bid_walls.append({"price": price, "quantity": qty, "percentage": qty / bid_volume * 100})
                    
            for price, qty in asks:
                if qty / ask_volume > wall_threshold:
                    ask_walls.append({"price": price, "quantity": qty, "percentage": qty / ask_volume * 100})
            
            # Calcul du déséquilibre achat/vente
            if ask_volume > 0:
                buy_sell_ratio = bid_volume / ask_volume
            else:
                buy_sell_ratio = float('inf')
            
            return {
                "bid_volume": bid_volume,
                "ask_volume": ask_volume,
                "buy_sell_ratio": buy_sell_ratio,
                "bid_walls": bid_walls,
                "ask_walls": ask_walls,
                "buy_pressure": buy_sell_ratio > 1.2,  # Forte pression d'achat
                "sell_pressure": buy_sell_ratio < 0.8,  # Forte pression de vente
                "timestamp": datetime.now().timestamp()
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse du carnet d'ordres pour {symbol}: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().timestamp()
            }
    
    def get_volume_profile(self, symbol: str, timeframe: str, periods: int = 24) -> Dict:
        """
        Calcule le profil de volume pour identifier les niveaux de prix significatifs
        
        Args:
            symbol: Paire de trading
            timeframe: Intervalle de temps
            periods: Nombre de périodes à analyser
            
        Returns:
            Profil de volume
        """
        try:
            # Récupération des données OHLCV
            ohlcv = self.get_ohlcv(symbol, timeframe, limit=periods)
            
            if ohlcv.empty:
                return {"error": "Données OHLCV vides", "timestamp": datetime.now().timestamp()}
            
            # Trouver le prix min et max sur la période
            price_min = ohlcv['low'].min()
            price_max = ohlcv['high'].max()
            
            # Création de tranches de prix (10 tranches)
            price_range = price_max - price_min
            slice_height = price_range / 10
            
            volume_profile = []
            
            for i in range(10):
                price_level_min = price_min + i * slice_height
                price_level_max = price_min + (i + 1) * slice_height
                
                # Sélection des chandeliers qui traversent cette tranche de prix
                mask = (ohlcv['high'] >= price_level_min) & (ohlcv['low'] <= price_level_max)
                volume_in_range = ohlcv.loc[mask, 'volume'].sum()
                
                volume_profile.append({
                    "price_level_min": price_level_min,
                    "price_level_max": price_level_max,
                    "volume": volume_in_range
                })
            
            # Tri par volume décroissant
            volume_profile.sort(key=lambda x: x["volume"], reverse=True)
            
            # Identification des niveaux de prix à fort volume (Value Area)
            total_volume = ohlcv['volume'].sum()
            cumulative_volume = 0
            value_area = []
            
            for level in volume_profile:
                cumulative_volume += level["volume"]
                level["percentage"] = level["volume"] / total_volume * 100
                
                if cumulative_volume <= total_volume * 0.7:  # 70% du volume total
                    value_area.append({
                        "price_min": level["price_level_min"],
                        "price_max": level["price_level_max"],
                        "volume": level["volume"],
                        "percentage": level["percentage"]
                    })
            
            return {
                "volume_profile": volume_profile,
                "value_area": value_area,
                "point_of_control": volume_profile[0],  # Niveau de prix avec le plus de volume
                "timestamp": datetime.now().timestamp()
            }
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul du profil de volume pour {symbol}: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().timestamp()
            }
    def _is_market_volatile(self, symbol: str) -> bool:
        """
        Détermine si le marché est actuellement volatil
        
        Args:
            symbol: Paire de trading
            
        Returns:
            True si le marché est volatil, False sinon
        """
        try:
            # Récupérer les dernières données
            ohlcv = self.get_ohlcv(symbol, "1m", limit=10)
            
            if ohlcv.empty:
                return False
            
            # Calculer la volatilité (ATR sur les 10 dernières minutes)
            high_low = ohlcv['high'] - ohlcv['low']
            high_close = abs(ohlcv['high'] - ohlcv['close'].shift())
            low_close = abs(ohlcv['low'] - ohlcv['close'].shift())
            
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.mean()
            
            # Calculer la volatilité relative au prix
            current_price = ohlcv['close'].iloc[-1]
            volatility_percent = (atr / current_price) * 100
            
            # Considérer comme volatil si la volatilité est supérieure à 0.5% sur 10 minutes
            return volatility_percent > 0.5
            
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation de la volatilité pour {symbol}: {str(e)}")
            return False  # Par défaut, considérer comme non volatil