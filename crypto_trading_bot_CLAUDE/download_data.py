"""
Script pour télécharger des données historiques depuis Binance
"""
import os
import sys
import argparse
import pandas as pd
from datetime import datetime
import time
from binance.client import Client

# Ajouter le répertoire parent au PYTHONPATH pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import DATA_DIR, API_KEYS
from utils.logger import setup_logger

logger = setup_logger("data_downloader")

def parse_args():
    """Parse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(description="Téléchargement de données historiques Binance")
    
    parser.add_argument("--symbol", type=str, required=True,
                      help="Symbole de trading (ex: BTCUSDT)")
    parser.add_argument("--interval", type=str, required=True,
                      help="Intervalle de temps (ex: 15m, 4h, 1d)")
    parser.add_argument("--start", type=str, required=True,
                      help="Date de début (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True,
                      help="Date de fin (YYYY-MM-DD)")
    parser.add_argument("--output", type=str,
                      help="Chemin du fichier de sortie")
    parser.add_argument("--use_testnet", action="store_true",
                      help="Utiliser le testnet Binance")
    
    return parser.parse_args()

def init_client(use_testnet=False):
    """Initialise le client Binance"""
    api_key = API_KEYS.get("BINANCE", {}).get("key", "")
    api_secret = API_KEYS.get("BINANCE", {}).get("secret", "")
    
    if not api_key or not api_secret:
        logger.warning("Clés API Binance non configurées, utilisation du client sans authentification")
        return Client("", "")  # Client sans clés API (limité aux données publiques)
    
    try:
        client = Client(api_key, api_secret, testnet=use_testnet)
        logger.info("Client Binance initialisé avec succès")
        return client
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du client Binance: {str(e)}")
        logger.info("Initialisation du client Binance sans authentification")
        return Client("", "")  # Client sans clés API (limité aux données publiques)

def interval_to_milliseconds(interval):
    """Convertit un intervalle en millisecondes"""
    seconds_per_unit = {
        "m": 60,
        "h": 60 * 60,
        "d": 24 * 60 * 60,
        "w": 7 * 24 * 60 * 60
    }
    
    try:
        unit = interval[-1]
        if unit not in seconds_per_unit:
            return None
        
        value = int(interval[:-1])
        return value * seconds_per_unit[unit] * 1000
    except:
        return None

def download_historical_data(client, symbol, interval, start_date, end_date, output_file=None):
    """
    Télécharge les données historiques de Binance
    
    Args:
        client: Client Binance
        symbol: Symbole de trading (ex: 'BTCUSDT')
        interval: Intervalle de temps (ex: '15m', '4h', '1d')
        start_date: Date de début (YYYY-MM-DD)
        end_date: Date de fin (YYYY-MM-DD)
        output_file: Chemin de fichier de sortie
        
    Returns:
        Le DataFrame avec les données et le chemin du fichier
    """
    # Convertir les dates en timestamps
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
    
    logger.info(f"Téléchargement des données {symbol} {interval} du {start_date} au {end_date}")
    
    # Générer le nom du fichier si non spécifié
    if not output_file:
        os.makedirs(os.path.join(DATA_DIR, "market_data"), exist_ok=True)
        output_file = os.path.join(
            DATA_DIR, 
            "market_data", 
            f"{symbol}_{interval}_{start_date}_to_{end_date}.csv"
        )
    
    # Assurer que le répertoire de sortie existe
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        # Télécharger les données par tranches pour éviter les limitations de l'API
        all_klines = []
        current_start = start_ts
        
        while current_start < end_ts:
            # Calculer la date de fin pour cette tranche (pas plus de 1000 bougies)
            chunk_end = min(current_start + (1000 * interval_to_milliseconds(interval)), end_ts)
            
            # Télécharger cette tranche
            klines = client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=current_start,
                end_str=chunk_end
            )
            
            if not klines:
                logger.warning(f"Aucune donnée trouvée pour {symbol} entre {datetime.fromtimestamp(current_start/1000)} et {datetime.fromtimestamp(chunk_end/1000)}")
                break
            
            all_klines.extend(klines)
            current_start = chunk_end
            
            # Attendre un peu pour éviter de surcharger l'API
            time.sleep(0.5)
            
            logger.info(f"Téléchargé {len(all_klines)} bougies jusqu'à {datetime.fromtimestamp(current_start/1000)}")
        
        # Convertir en DataFrame
        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignored'
        ])
        
        # Convertir les types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        # Sauvegarder en CSV
        df.to_csv(output_file, index=False)
        logger.info(f"Données sauvegardées: {output_file} ({len(df)} lignes)")
        
        return df, output_file
        
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement des données: {str(e)}")
        return None, None

def main():
    """Fonction principale"""
    # Parser les arguments
    args = parse_args()
    
    # Initialiser le client Binance
    client = init_client(args.use_testnet)
    
    # Télécharger les données
    df, output_file = download_historical_data(
        client=client,
        symbol=args.symbol,
        interval=args.interval,
        start_date=args.start,
        end_date=args.end,
        output_file=args.output
    )
    
    if df is not None:
        print(f"Téléchargement terminé. {len(df)} lignes sauvegardées dans {output_file}")

if __name__ == "__main__":
    main()