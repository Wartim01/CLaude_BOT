"""
Advanced technical indicators and features for cryptocurrency market analysis
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
import talib

try:
    import pywt  # PyWavelets for wavelet transform
except ImportError:
    pywt = None
    print("Warning: Module 'pywt' not installed. Wavelet transform features will be disabled. To install, run 'pip install PyWavelets'.")

class AdvancedFeatures:
    """Class providing advanced market features beyond basic indicators"""
    
    @staticmethod
    def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
        """Apply all advanced features to dataframe"""
        df = df.copy()
        
        # Check required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            raise ValueError(f"DataFrame missing required columns: {[col for col in required if col not in df.columns]}")
            
        # Apply features
        df = AdvancedFeatures.add_volatility_features(df)
        df = AdvancedFeatures.add_volume_features(df)
        df = AdvancedFeatures.add_price_pattern_features(df)
        df = AdvancedFeatures.add_statistical_features(df)
        df = AdvancedFeatures.add_cyclical_features(df)
        df = AdvancedFeatures.add_market_regime_features(df)
        
        return df

    @staticmethod
    def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features"""
        # ATR-based features
        atr = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        df['atr'] = atr
        df['atr_percent'] = atr / df['close'] * 100
        
        # Rolling volatility
        for period in [5, 10, 21]:
            df[f'volatility_{period}'] = df['close'].pct_change().rolling(period).std() * np.sqrt(period)
        
        # Volatility ratio (short-term vs long-term)
        df['volatility_ratio'] = df['volatility_5'] / df['volatility_21']
        
        # Squeeze indicator (Bollinger Bands vs Keltner Channels)
        bb_width = AdvancedFeatures.bollinger_bandwidth(df, 20, 2)
        df['bb_width'] = bb_width
        df['squeeze_on'] = bb_width < df['bb_width'].rolling(50).quantile(0.25)
        
        # VPIN (Volume-synchronized Probability of Informed Trading)
        df['vpin'] = df['volume'] * abs(df['close'].pct_change())
        df['vpin_ma'] = df['vpin'].rolling(20).mean()
        
        return df

    @staticmethod
    def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        # Volume change rate
        df['volume_change'] = df['volume'].pct_change()
        
        # Relative volume
        for period in [5, 10, 21]:
            df[f'rel_volume_{period}'] = df['volume'] / df['volume'].rolling(period).mean()
        
        # Money Flow Index
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['money_flow'] = df['typical_price'] * df['volume']
        
        # Calculate positive and negative money flow
        df['pos_flow'] = np.where(df['typical_price'] > df['typical_price'].shift(1), df['money_flow'], 0)
        df['neg_flow'] = np.where(df['typical_price'] < df['typical_price'].shift(1), df['money_flow'], 0)
        
        # Money Flow Index
        for period in [7, 14]:
            pos_sum = df['pos_flow'].rolling(period).sum()
            neg_sum = df['neg_flow'].rolling(period).sum()
            money_ratio = np.where(neg_sum != 0, pos_sum / neg_sum, 100)
            df[f'mfi_{period}'] = 100 - (100 / (1 + money_ratio))
        
        # Volume Weighted MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        macd = ema12 - ema26
        df['vw_macd'] = macd * df['rel_volume_5']
        df['vw_macd_signal'] = df['vw_macd'].ewm(span=9).mean()
        
        # OBV
        df['obv'] = talib.OBV(df['close'].values, df['volume'].values)
        
        # Chaikin Money Flow
        df['cmf'] = talib.ADOSC(df['high'].values, df['low'].values, 
                               df['close'].values, df['volume'].values, 
                               fastperiod=3, slowperiod=10)
        
        # Volume Price Trend
        df['vpt'] = df['volume'] * df['close'].pct_change().cumsum()
        
        return df

    @staticmethod
    def add_price_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern recognition features"""
        # Basic price patterns using TA-Lib
        bullish_patterns = [
            'CDL3WHITESOLDIERS', 'CDLMORNINGSTAR', 'CDLHAMMER', 'CDLINVERTEDHAMMER',
            'CDLPIERCING', 'CDLENGULFING', 'CDLHARAMI'
        ]
        
        bearish_patterns = [
            'CDL3BLACKCROWS', 'CDLEVENINGSTAR', 'CDLHANGINGMAN', 'CDLSHOOTINGSTAR',
            'CDLDARKCLOUDCOVER', 'CDLENGULFING', 'CDLHARAMI'
        ]
        
        # Add bullish patterns
        df['bullish_patterns'] = 0
        for pattern in bullish_patterns:
            try:
                pattern_func = getattr(talib, pattern)
                result = pattern_func(df['open'].values, df['high'].values, 
                                     df['low'].values, df['close'].values)
                # Only add positive signals (bullish)
                df[pattern.lower()] = np.where(result > 0, 1, 0)
                df['bullish_patterns'] += df[pattern.lower()]
                # Remove the individual pattern column to save space
                df.drop(columns=[pattern.lower()], inplace=True)
            except:
                pass
        
        # Add bearish patterns
        df['bearish_patterns'] = 0
        for pattern in bearish_patterns:
            try:
                pattern_func = getattr(talib, pattern)
                result = pattern_func(df['open'].values, df['high'].values, 
                                     df['low'].values, df['close'].values)
                # Only add negative signals (bearish)
                df[pattern.lower()] = np.where(result < 0, 1, 0)
                df['bearish_patterns'] += df[pattern.lower()]
                # Remove the individual pattern column to save space
                df.drop(columns=[pattern.lower()], inplace=True)
            except:
                pass
        
        # Candlestick body and wick features
        df['body_size'] = abs(df['close'] - df['open'])
        df['body_size_pct'] = df['body_size'] / df['open'] * 100
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['upper_wick_pct'] = df['upper_wick'] / df[['open', 'close']].max(axis=1) * 100
        df['lower_wick_pct'] = df['lower_wick'] / df[['open', 'close']].min(axis=1) * 100
        
        # Gap detection
        df['gap_up'] = ((df['low'] > df['high'].shift(1)) * 1.0).fillna(0)
        df['gap_down'] = ((df['high'] < df['low'].shift(1)) * 1.0).fillna(0)
        df['gap_size'] = np.where(df['gap_up'] == 1, 
                                 (df['low'] - df['high'].shift(1)) / df['high'].shift(1) * 100,
                                 np.where(df['gap_down'] == 1,
                                         (df['high'] - df['low'].shift(1)) / df['low'].shift(1) * 100,
                                         0))
        
        return df
        
    @staticmethod
    def add_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features and anomaly indicators"""
        # Z-score of returns
        for period in [5, 10, 21]:
            returns = df['close'].pct_change(1)
            mean = returns.rolling(period).mean()
            std = returns.rolling(period).std()
            df[f'return_zscore_{period}'] = (returns - mean) / std
        
        # Price distance from moving averages
        for ma_period in [20, 50, 200]:
            ma = df['close'].rolling(ma_period).mean()
            df[f'dist_ma_{ma_period}'] = (df['close'] - ma) / ma * 100
        
        # Fractal dimension (measure of price complexity)
        window = 10
        df['fractal_dimension'] = AdvancedFeatures.calculate_fractal_dimension(df['close'], window)
        
        # Hurst exponent (measure of mean reversion vs trend following)
        df['hurst_exponent'] = AdvancedFeatures.calculate_hurst_exponent(df['close'], 20)
        
        # Detrended Price Oscillator (DPO)
        for period in [14, 21]:
            ma = df['close'].rolling(period).mean()
            df[f'dpo_{period}'] = df['close'] - ma.shift(period // 2 + 1)
        
        return df
    
    @staticmethod
    def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical and spectral features"""
        # Fourier Transform Features (simplified)
        if len(df) >= 128:  # Need sufficient data
            try:
                # Get detrended close prices
                detrended = df['close'].values - df['close'].rolling(20).mean().values
                detrended = np.nan_to_num(detrended)
                
                # Take last 128 points for FFT
                data_segment = detrended[-128:]
                
                # Calculate FFT
                fft_values = np.fft.rfft(data_segment)
                fft_norm = np.abs(fft_values) / len(data_segment)
                
                # Get dominant frequencies (top 3)
                dominant_freqs = np.argsort(fft_norm)[-3:]
                
                # Create features for each top frequency
                for i, freq_idx in enumerate(dominant_freqs):
                    freq = freq_idx / len(data_segment)  # Normalized frequency
                    power = fft_norm[freq_idx]
                    
                    # Add as features
                    df[f'fft_freq_{i+1}'] = freq
                    df[f'fft_power_{i+1}'] = power
                    
                # Calculate spectral entropy
                spectral_entropy = -np.sum(fft_norm * np.log2(fft_norm + 1e-10)) / np.log2(len(fft_norm))
                df['spectral_entropy'] = spectral_entropy
                
            except Exception as e:
                print(f"Error calculating FFT features: {str(e)}")
        
        # Wavelet transform features (only if pywt is available)
        if pywt is not None and len(df) >= 64:
            try:
                # Apply wavelet decomposition
                data = df['close'].values[-64:]
                coeffs = pywt.wavedec(data, 'db4', level=3)
                
                # Calculate energy of each level
                energy = [np.sum(c**2) for c in coeffs]
                total_energy = sum(energy)
                
                # Energy distribution
                for i, e in enumerate(energy):
                    df[f'wavelet_energy_{i}'] = e / total_energy
                
            except Exception as e:
                print(f"Error calculating wavelet features: {str(e)}")
        else:
            if pywt is None:
                print("Skipping wavelet features due to missing PyWavelets module.")
        
        return df
    
    @staticmethod
    def add_market_regime_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime and sentiment indicators"""
        # ADX (trend strength)
        df['adx'] = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        df['trend_strength'] = np.where(df['adx'] < 20, 'weak',
                                      np.where(df['adx'] < 40, 'moderate', 'strong'))
        
        # Choppiness Index
        period = 14
        df['choppiness'] = 100 * np.log10(
            df['atr'].rolling(period).sum() /
            (df['high'].rolling(period).max() - df['low'].rolling(period).min())
        ) / np.log10(period)
        
        # RSI trend and divergence
        df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)
        
        # RSI divergence - bullish and bearish
        df['price_higher_high'] = (df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))
        df['price_lower_low'] = (df['low'] < df['low'].shift(1)) & (df['low'].shift(1) < df['low'].shift(2))
        df['rsi_higher_high'] = (df['rsi'] > df['rsi'].shift(1)) & (df['rsi'].shift(1) > df['rsi'].shift(2))
        df['rsi_lower_low'] = (df['rsi'] < df['rsi'].shift(1)) & (df['rsi'].shift(1) < df['rsi'].shift(2))
        
        df['bullish_divergence'] = df['price_lower_low'] & ~df['rsi_lower_low']
        df['bearish_divergence'] = df['price_higher_high'] & ~df['rsi_higher_high']
        
        # Market mode based on BB width and ADX
        df['market_mode'] = np.where(
            (df['bb_width'] < df['bb_width'].rolling(50).mean()) & (df['adx'] < 25), 'ranging',
            np.where(df['adx'] > 25, 'trending', 'uncertain')
        )
        
        return df
    
    # Helper methods
    @staticmethod
    def bollinger_bandwidth(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.Series:
        """Calculate Bollinger Bandwidth"""
        ma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        upper = ma + (std * num_std)
        lower = ma - (std * num_std)
        return (upper - lower) / ma * 100
    
    @staticmethod
    def calculate_hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
        """
        Calculate Hurst Exponent to determine if a time series is mean-reverting, 
        random walk, or trending.
        H < 0.5: Mean-reverting
        H = 0.5: Random walk
        H > 0.5: Trending
        """
        # Convert to numpy array and fill NaNs
        series = np.asarray(series)
        series = np.nan_to_num(series, nan=np.nanmean(series))
        
        # Calculate lags (use final max_lag points for efficiency)
        lags = range(2, min(max_lag, len(series) // 4))
        
        # Calculate variance of differences for each lag
        tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
        
        # Linear fit to measure power law relationship
        m = np.polyfit(np.log(lags), np.log(tau), 1)
        
        # Hurst exponent is the slope
        hurst = m[0]
        return hurst
        
    @staticmethod
    def calculate_fractal_dimension(series: pd.Series, window: int = 10) -> pd.Series:
        """
        Calculate the fractal dimension of the time series using box-counting method
        """
        result = pd.Series(index=series.index, dtype='float64')
        
        for i in range(window, len(series)):
            # Get data window
            ts = series.iloc[i-window:i].values
            ts = np.nan_to_num(ts, nan=np.nanmean(ts))
            
            # Normalize to [0, 1]
            ts_min = np.min(ts)
            ts_max = np.max(ts)
            if (ts_max - ts_min) != 0:
                ts_norm = (ts - ts_min) / (ts_max - ts_min)
                
                # Calculate curve length
                length = np.sum(np.sqrt(1 + np.diff(ts_norm)**2))
                
                # Calculate fractal dimension
                if length > 0:
                    result.iloc[i] = 1 + np.log(length) / np.log(2*(window-1))
                else:
                    result.iloc[i] = 1.0
            else:
                result.iloc[i] = 1.0  # Default value
                
        return result

def calculate_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate advanced features for the given DataFrame using the AdvancedFeatures class.
    This function performs preprocessing by filling missing values, then computes advanced features,
    drops fully NaN columns, and standardizes numeric features.
    
    Args:
        df: DataFrame with market data.
        
    Returns:
        DataFrame with advanced features added and standardized.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    
    # Preprocessing: Fill missing values
    df_filled = df.fillna(method='ffill').fillna(method='bfill')
    
    # Compute advanced features using the AdvancedFeatures class methods
    features_df = AdvancedFeatures.create_all_features(df_filled)
    
    # Postprocessing: Drop any column that is entirely NaN
    features_df.dropna(axis=1, how='all', inplace=True)
    
    # Standardize numeric features
    numeric_cols = features_df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        std_val = features_df[col].std()
        if std_val != 0:
            features_df[col] = (features_df[col] - features_df[col].mean()) / std_val
    return features_df