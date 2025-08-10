"""
Technical indicators module for calculating various trading indicators.
Implements RSI, MACD, Moving Averages, and other technical indicators used in trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from loguru import logger

from ..config import Config


class TechnicalIndicators:
    """Class for calculating technical indicators from OHLCV data."""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices: Series of closing prices
            window: Period for RSI calculation (default: 14)
            
        Returns:
            Series with RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        # Handle division by zero
        rs = gain / (loss + 1e-10)  # Add small epsilon to avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        # Clip values to valid RSI range
        rsi = np.clip(rsi, 0, 100)
        
        return rsi
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: Series of closing prices
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line EMA period (default: 9)
            
        Returns:
            Dictionary with 'macd', 'signal', and 'histogram' series
        """
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        
        return {
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def calculate_moving_averages(prices: pd.Series, windows: list = [20, 50]) -> Dict[str, pd.Series]:
        """
        Calculate Simple Moving Averages for multiple periods.
        
        Args:
            prices: Series of closing prices
            windows: List of window periods
            
        Returns:
            Dictionary with moving averages for each window
        """
        ma_dict = {}
        for window in windows:
            ma_dict[f'sma_{window}'] = prices.rolling(window=window).mean()
        
        return ma_dict
    
    @staticmethod
    def calculate_exponential_moving_averages(prices: pd.Series, windows: list = [20, 50]) -> Dict[str, pd.Series]:
        """
        Calculate Exponential Moving Averages for multiple periods.
        
        Args:
            prices: Series of closing prices
            windows: List of window periods
            
        Returns:
            Dictionary with EMAs for each window
        """
        ema_dict = {}
        for window in windows:
            ema_dict[f'ema_{window}'] = prices.ewm(span=window).mean()
        
        return ema_dict
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: Series of closing prices
            window: Period for moving average (default: 20)
            num_std: Number of standard deviations (default: 2)
            
        Returns:
            Dictionary with 'upper', 'middle', and 'lower' bands
        """
        middle = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }
    
    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                           k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            k_period: %K period (default: 14)
            d_period: %D period (default: 3)
            
        Returns:
            Dictionary with '%K' and '%D' series
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'k_percent': k_percent,
            'd_percent': d_percent
        }
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            window: Period for ATR calculation (default: 14)
            
        Returns:
            Series with ATR values
        """
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        
        return atr
    
    @staticmethod
    def calculate_volume_indicators(volume: pd.Series, close: pd.Series, window: int = 20) -> Dict[str, pd.Series]:
        """
        Calculate volume-based indicators.
        
        Args:
            volume: Series of volume data
            close: Series of closing prices
            window: Period for calculations
            
        Returns:
            Dictionary with volume indicators
        """
        # Volume Moving Average
        volume_ma = volume.rolling(window=window).mean()
        
        # Volume Ratio (current volume / average volume) - handle division by zero
        volume_ratio = volume / (volume_ma + 1e-10)
        volume_ratio = np.clip(volume_ratio, 0, 100)  # Clip extreme values
        
        # On Balance Volume (OBV)
        obv = (volume * np.sign(close.diff())).cumsum()
        
        # Volume Weighted Average Price (VWAP) - simplified
        cumulative_volume = volume.cumsum()
        vwap = (close * volume).cumsum() / (cumulative_volume + 1e-10)
        
        return {
            'volume_ma': volume_ma,
            'volume_ratio': volume_ratio,
            'obv': obv,
            'vwap': vwap
        }
    
    @staticmethod
    def calculate_volatility(prices: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate price volatility (rolling standard deviation of returns).
        
        Args:
            prices: Series of closing prices
            window: Period for volatility calculation
            
        Returns:
            Series with volatility values
        """
        returns = prices.pct_change()
        volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
        
        return volatility
    
    @classmethod
    def calculate_all_indicators(cls, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for a given OHLCV dataset.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all indicators added
        """
        if data.empty:
            logger.warning("Empty dataset provided for indicator calculation")
            return data
        
        result_df = data.copy()
        
        try:
            # RSI
            result_df['rsi'] = cls.calculate_rsi(data['close'], Config.RSI_PERIOD)
            
            # MACD
            macd_data = cls.calculate_macd(data['close'])
            for key, series in macd_data.items():
                result_df[key] = series
            
            # Moving Averages
            ma_data = cls.calculate_moving_averages(data['close'], [Config.SHORT_MA, Config.LONG_MA])
            for key, series in ma_data.items():
                result_df[key] = series
            
            # Exponential Moving Averages
            ema_data = cls.calculate_exponential_moving_averages(data['close'], [12, 26])
            for key, series in ema_data.items():
                result_df[key] = series
            
            # Bollinger Bands
            bb_data = cls.calculate_bollinger_bands(data['close'])
            for key, series in bb_data.items():
                result_df[f'bb_{key}'] = series
            
            # Stochastic
            stoch_data = cls.calculate_stochastic(data['high'], data['low'], data['close'])
            for key, series in stoch_data.items():
                result_df[key] = series
            
            # ATR
            result_df['atr'] = cls.calculate_atr(data['high'], data['low'], data['close'])
            
            # Volume indicators
            if 'volume' in data.columns:
                volume_data = cls.calculate_volume_indicators(data['volume'], data['close'])
                for key, series in volume_data.items():
                    result_df[key] = series
            
            # Volatility
            result_df['volatility'] = cls.calculate_volatility(data['close'])
            
            # Price change indicators
            result_df['price_change'] = data['close'].pct_change()
            result_df['price_change_5d'] = data['close'].pct_change(5)
            result_df['price_change_20d'] = data['close'].pct_change(20)
            
            # Support and Resistance levels (simplified)
            result_df['resistance'] = data['high'].rolling(window=20).max()
            result_df['support'] = data['low'].rolling(window=20).min()
            
            logger.success(f"Successfully calculated all technical indicators")
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            
        return result_df
    
    @staticmethod
    def get_signal_strength(rsi: float, macd_histogram: float, volume_ratio: float) -> str:
        """
        Determine signal strength based on multiple indicators.
        
        Args:
            rsi: RSI value
            macd_histogram: MACD histogram value
            volume_ratio: Volume ratio
            
        Returns:
            Signal strength ('STRONG', 'MODERATE', 'WEAK')
        """
        strength_score = 0
        
        # RSI strength
        if rsi < 25 or rsi > 75:
            strength_score += 2
        elif rsi < 35 or rsi > 65:
            strength_score += 1
        
        # MACD momentum
        if abs(macd_histogram) > 0.5:
            strength_score += 2
        elif abs(macd_histogram) > 0.2:
            strength_score += 1
        
        # Volume confirmation
        if volume_ratio > 1.5:
            strength_score += 2
        elif volume_ratio > 1.2:
            strength_score += 1
        
        if strength_score >= 4:
            return 'STRONG'
        elif strength_score >= 2:
            return 'MODERATE'
        else:
            return 'WEAK'