#!/usr/bin/env python
"""
Script to prepare validation data for model evaluation
Extracts a subset of market data and saves it in the processed directory
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path to allow imports from other modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import project modules
from config.config import MARKET_DATA_DIR, PROCESSED_DATA_DIR
from utils.logger import setup_logger
from ai.models.feature_engineering import FeatureEngineering

# Set up logger
logger = setup_logger('prepare_validation')

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Prepare validation data for model evaluation")
    
    parser.add_argument("--symbol", type=str, default="BTCUSDT",
                      help="Trading pair symbol (e.g., BTCUSDT)")
    parser.add_argument("--timeframe", type=str, default="15m",
                      help="Timeframe for the data (e.g., 15m, 1h)")
    parser.add_argument("--market_data_dir", type=str, default=None,
                      help="Market data directory (default: from config)")
    parser.add_argument("--output_dir", type=str, default=None,
                      help="Output directory (default: from config)")
    parser.add_argument("--validation_size", type=float, default=0.2,
                      help="Size of validation set as fraction of total data")
    parser.add_argument("--start_date", type=str, default=None,
                      help="Start date for validation data (format: YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default=None,
                      help="End date for validation data (format: YYYY-MM-DD)")
    parser.add_argument("--random_seed", type=int, default=42,
                      help="Random seed for reproducibility")
    parser.add_argument("--add_features", action="store_true",
                      help="Add technical indicators and other features to the data")
    parser.add_argument("--target_horizon", type=int, default=None,
                      help="Target prediction horizon (in periods)")
    
    return parser.parse_args()

def load_market_data(symbol, timeframe, data_dir):
    """
    Load market data for the specified symbol and timeframe
    
    Args:
        symbol: Trading pair symbol
        timeframe: Timeframe string
        data_dir: Directory containing market data files
        
    Returns:
        DataFrame with market data or None if file not found
    """
    # Try direct filename
    file_path = os.path.join(data_dir, f"{symbol}_{timeframe}.csv")
    
    if not os.path.exists(file_path):
        # Try case-insensitive search
        for file in os.listdir(data_dir):
            if f"{symbol.lower()}_{timeframe.lower()}" in file.lower() and file.endswith('.csv'):
                file_path = os.path.join(data_dir, file)
                logger.info(f"Found matching file: {file}")
                break
        else:
            logger.error(f"No market data file found for {symbol} {timeframe} in {data_dir}")
            return None
    
    logger.info(f"Loading market data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        
        # Check if timestamp column exists, create it if not
        if 'timestamp' not in df.columns:
            if 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'])
            else:
                logger.error(f"No timestamp or date column found in {file_path}")
                return None
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        logger.info(f"Loaded {len(df)} rows of data from {df.index.min()} to {df.index.max()}")
        return df
    
    except Exception as e:
        logger.error(f"Error loading market data: {str(e)}")
        return None

def prepare_validation_data(df, args):
    """
    Prepare validation data from the input DataFrame
    
    Args:
        df: Input DataFrame with market data
        args: Command line arguments
        
    Returns:
        DataFrame with validation data
    """
    # Filter by date range if specified
    if args.start_date:
        start_date = pd.to_datetime(args.start_date)
        df = df[df.index >= start_date]
        
    if args.end_date:
        end_date = pd.to_datetime(args.end_date)
        df = df[df.index <= end_date]
    
    # If no date range specified, select a validation set from the end of the data
    if not args.start_date and not args.end_date:
        # Determine size of validation set
        val_size = int(len(df) * args.validation_size)
        
        # Take the last portion as validation set
        df = df.iloc[-val_size:]
    
    logger.info(f"Selected {len(df)} rows for validation data from {df.index.min()} to {df.index.max()}")
    
    # Add technical indicators and other features if requested
    if args.add_features:
        logger.info("Adding technical indicators and features...")
        feature_engineering = FeatureEngineering()
        df = feature_engineering.create_features(df, include_time_features=True, include_price_patterns=True)
        
        # Update to use newer pandas methods
        df = df.ffill()  # Forward fill
        df = df.fillna(0)  # Fill any remaining NaNs with 0
    
    # Set a default horizon based on timeframe
    target_horizon = 4  # Default to 4 periods
    
    if args.timeframe == '15m':
        target_horizon = 4  # 4 periods of 15 min = 1 hour
    elif args.timeframe == '1h':
        target_horizon = 1  # 1 period = 1 hour
    elif args.timeframe == '5m':
        target_horizon = 12  # 12 periods of 5 min = 1 hour
    
    # Override with provided value if it exists
    if args.target_horizon is not None:
        target_horizon = args.target_horizon
        
    logger.info(f"Using target horizon: {target_horizon}")
    
    # Generate target labels
    logger.info(f"Generating target labels...")
    
    # Create directional target (1 if price goes up, 0 if down)
    df['future_close'] = df['close'].shift(-target_horizon)
    df['target'] = (df['future_close'] > df['close']).astype(int)
    
    # Drop rows with NaN targets at the end
    df.dropna(subset=['target'], inplace=True)
    
    # Remove future_close column to avoid feature engineering conflicts
    df = df.drop(columns=['future_close'])
    
    logger.info(f"Generated target labels with horizon {target_horizon}, resulting in {len(df)} valid rows")
    
    return df

def generate_target_labels(df, horizon=4):
    """
    Generate target labels for directional prediction
    
    Args:
        df: DataFrame with market data
        horizon: Number of periods to look ahead for prediction
        
    Returns:
        DataFrame with added target columns
    """
    # Check if 'close' column exists in the DataFrame
    if 'close' not in df.columns:
        logger.error("The input DataFrame does not contain a 'close' column required for target generation.")
        raise ValueError("Missing 'close' column in DataFrame")
    
    # Ensure horizon is an integer and not None
    if horizon is None:
        horizon = 4
        logger.warning("Received None for horizon in generate_target_labels, defaulting to 4")
    
    # Convert to int in case it's a float or string
    try:
        horizon = int(horizon)
    except (TypeError, ValueError):
        logger.warning(f"Could not convert horizon value to int: {horizon}, defaulting to 4")
        horizon = 4
    
    logger.info(f"Using horizon value of {horizon} for target generation")
    
    # Create directional target (1 if price goes up, 0 if down)
    df['future_close'] = df['close'].shift(-horizon)
    df['target'] = (df['future_close'] > df['close']).astype(int)
    
    # Drop rows with NaN targets at the end
    df.dropna(subset=['target'], inplace=True)
    
    logger.info(f"Generated target labels with horizon {horizon}, resulting in {len(df)} valid rows")
    return df

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Set directories
    market_data_dir = args.market_data_dir or MARKET_DATA_DIR
    output_dir = args.output_dir or PROCESSED_DATA_DIR
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load market data
    df = load_market_data(args.symbol, args.timeframe, market_data_dir)
    if df is None:
        sys.exit(1)
    
    # Prepare validation data
    val_data = prepare_validation_data(df, args)
    
    # Save validation data
    output_path = os.path.join(output_dir, f"{args.symbol}_{args.timeframe}_val.csv")
    val_data.reset_index().to_csv(output_path, index=False)
    logger.info(f"Validation data saved to {output_path}")
    
    # Print summary
    logger.info(f"Validation data summary:")
    logger.info(f"  Symbol: {args.symbol}")
    logger.info(f"  Timeframe: {args.timeframe}")
    logger.info(f"  Rows: {len(val_data)}")
    logger.info(f"  Date range: {val_data.index.min()} to {val_data.index.max()}")
    logger.info(f"  Columns: {', '.join(val_data.columns)}")
    
    # Check if target column exists
    if 'target' in val_data.columns:
        target_distribution = val_data['target'].value_counts(normalize=True) * 100
        logger.info(f"  Target distribution: Up {target_distribution.get(1, 0):.1f}%, Down {target_distribution.get(0, 0):.1f}%")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error preparing validation data: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
