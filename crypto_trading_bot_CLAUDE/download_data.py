"""
Script to download historical cryptocurrency market data from Binance API
"""
import os
import argparse
import pandas as pd
from datetime import datetime, timedelta
from binance.client import Client
import time

# Add the project directory to the Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import DATA_DIR, API_KEYS

def download_historical_data(symbol, interval, start_date, end_date=None, output_dir=None):
    """
    Download historical market data from Binance
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Timeframe interval (e.g., '1h', '15m')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format (default: current date)
        output_dir: Directory to save the data (default: DATA_DIR/market_data)
    
    Returns:
        Path to the saved data file
    """
    # Initialize Binance client
    client = Client(
        api_key=API_KEYS["binance"]["testnet"]["API_KEY"],
        api_secret=API_KEYS["binance"]["testnet"]["API_SECRET"]
    )
    
    # Set up output directory
    if output_dir is None:
        output_dir = os.path.join(DATA_DIR, "market_data")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse dates
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if end_date is None:
        end_date = datetime.now()
    else:
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Download data in chunks to avoid API limitations
    chunk_size = timedelta(days=90)
    current_start = start_date
    all_candles = []
    
    print(f"Downloading {symbol} {interval} data from {start_date} to {end_date}")
    
    while current_start < end_date:
        current_end = min(current_start + chunk_size, end_date)
        
        print(f"  Fetching chunk: {current_start} to {current_end}")
        
        # Convert dates to milliseconds for the Binance API
        start_ms = int(current_start.timestamp() * 1000)
        end_ms = int(current_end.timestamp() * 1000)
        
        # Get candlestick data from Binance
        candles = client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start_ms,
            end_str=end_ms
        )
        
        all_candles.extend(candles)
        
        # Move to next chunk
        current_start = current_end
        
        # Add a small delay to avoid hitting API rate limits
        time.sleep(1)
    
    # Create DataFrame
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
               'close_time', 'quote_asset_volume', 'number_of_trades',
               'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    
    df = pd.DataFrame(all_candles, columns=columns)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Convert numeric columns to float
    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']:
        df[col] = df[col].astype(float)
    
    # Save to CSV
    output_file = os.path.join(output_dir, f"{symbol}_{interval}.csv")
    df.to_csv(output_file, index=False)
    
    print(f"Downloaded {len(df)} candles")
    print(f"Data saved to {output_file}")
    
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download historical crypto data from Binance')
    parser.add_argument('--symbol', type=str, required=True, help='Trading pair symbol (e.g., BTCUSDT)')
    parser.add_argument('--interval', type=str, required=True, help='Timeframe interval (e.g., 1h, 15m)')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, help='Output directory')
    
    args = parser.parse_args()
    
    download_historical_data(
        symbol=args.symbol,
        interval=args.interval,
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output
    )