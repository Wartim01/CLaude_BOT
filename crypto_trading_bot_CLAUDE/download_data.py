# download_data.py
import os
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import requests
from config.config import DATA_DIR

def download_binance_data(symbol, interval, start_date, end_date, save_path=None):
    """
    Télécharge les données historiques OHLCV depuis l'API publique de Binance
    
    Args:
        symbol: Paire de trading (ex: BTCUSDT)
        interval: Intervalle de temps (1m, 5m, 15m, 1h, 4h, 1d, etc.)
        start_date: Date de début au format 'YYYY-MM-DD'
        end_date: Date de fin au format 'YYYY-MM-DD'
        save_path: Chemin pour sauvegarder les données (optionnel)
        
    Returns:
        DataFrame pandas avec les données OHLCV
    """
    # Convertir les dates en millisecondes pour l'API Binance
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
    
    # URL de l'API Binance
    url = 'https://api.binance.com/api/v3/klines'
    
    # Liste pour stocker toutes les données
    all_klines = []
    
    # Binance limite à 1000 chandeliers par requête
    # Nous devons faire plusieurs requêtes pour couvrir toute la période
    current_ts = start_ts
    
    while current_ts < end_ts:
        # Paramètres de la requête
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_ts,
            'endTime': end_ts,
            'limit': 1000
        }
        
        # Effectuer la requête
        response = requests.get(url, params=params)
        
        # Vérifier la réponse
        if response.status_code != 200:
            print(f"Erreur lors de la requête : {response.text}")
            return None
        
        # Convertir la réponse en JSON
        data = response.json()
        
        if not data:
            break
        
        # Ajouter les données à la liste
        all_klines.extend(data)
        
        # Mettre à jour le timestamp pour la prochaine requête
        current_ts = data[-1][0] + 1
        
        # Attendre un peu pour éviter de dépasser les limites de l'API
        time.sleep(0.5)
        
        print(f"Téléchargement en cours... {len(all_klines)} chandeliers récupérés")
    
    # Convertir les données en DataFrame pandas
    df = pd.DataFrame(all_klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    # Convertir les types de données
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sauvegarder les données si un chemin est fourni
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Données sauvegardées dans {save_path}")
    
    return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Téléchargement de données historiques Binance")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Paire de trading")
    parser.add_argument("--interval", type=str, default="15m", help="Intervalle de temps")
    parser.add_argument("--start", type=str, required=True, help="Date de début (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="Date de fin (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    # Créer le répertoire de données s'il n'existe pas
    market_data_dir = os.path.join(DATA_DIR, "market_data")
    if not os.path.exists(market_data_dir):
        os.makedirs(market_data_dir)
    
    # Construire le chemin de sauvegarde
    save_path = os.path.join(
        market_data_dir, 
        f"{args.symbol}_{args.interval}_{args.start}_{args.end}.csv"
    )
    
    # Télécharger les données
    download_binance_data(
        args.symbol, 
        args.interval, 
        args.start, 
        args.end, 
        save_path
    )

    def load_historical_data(symbol, interval, start_date, end_date,):
        """
        Charge les données historiques depuis le stockage local ou les télécharge si nécessaires
        
        Args:
            symbol: Paire de trading (ex: BTCUSDT)
            interval: Intervalle de temps (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            start_date: Date de début au format 'YYYY-MM-DD'
            end_date: Date de fin au format 'YYYY-MM-DD'
            force_download: Si True, force le téléchargement même si les données existent
            
        Returns:
            DataFrame pandas avec les données OHLCV
        """
        # Construire le chemin du fichier
        market_data_dir = os.path.join(DATA_DIR, "market_data")
        file_path = os.path.join(market_data_dir, f"{symbol}_{interval}_{start_date}_{end_date}.csv")
        
        # Vérifier si le fichier existe déjà
        if os.path.exists(file_path):
            print(f"Chargement des données locales pour {symbol} ({interval}) du {start_date} au {end_date}")
            df = pd.read_csv(file_path)
            
            # Convertir la colonne timestamp en datetime si nécessaire
            if 'timestamp' in df.columns and not pd.api.types.is_datetime64_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df