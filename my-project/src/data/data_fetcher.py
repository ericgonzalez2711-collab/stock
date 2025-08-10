"""
Data fetcher module for retrieving stock data from Alpha Vantage API.
Implements data ingestion for NIFTY 50 stocks with proper error handling and caching.
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger
import json
import os

from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators

from ..config import Config


class DataFetcher:
    """Handles stock data fetching from Alpha Vantage API."""
    
    def __init__(self, api_key: str = None):
        """Initialize DataFetcher with API key."""
        self.api_key = api_key or Config.ALPHA_VANTAGE_API_KEY
        if not self.api_key:
            raise ValueError("Alpha Vantage API key is required")
            
        self.ts = TimeSeries(key=self.api_key, output_format='pandas')
        self.ti = TechIndicators(key=self.api_key, output_format='pandas')
        
        # Rate limiting: Alpha Vantage free tier allows 5 requests per minute
        self.last_request_time = 0
        self.min_request_interval = 12  # seconds between requests
        
        # Create cache directory if it doesn't exist
        self.cache_dir = "cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info("DataFetcher initialized successfully")
    
    def _rate_limit(self):
        """Implement rate limiting to respect API limits."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _get_cache_filename(self, symbol: str, data_type: str, interval: str = "daily") -> str:
        """Generate cache filename for a given symbol and data type."""
        return os.path.join(self.cache_dir, f"{symbol}_{data_type}_{interval}.json")
    
    def _load_from_cache(self, symbol: str, data_type: str, interval: str = "daily") -> Optional[pd.DataFrame]:
        """Load data from cache if it exists and is recent."""
        cache_file = self._get_cache_filename(symbol, data_type, interval)
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Check if cache is recent (less than 1 hour old for intraday, 1 day for daily)
                cache_time = datetime.fromisoformat(cache_data['timestamp'])
                max_age = timedelta(hours=1) if interval != "daily" else timedelta(days=1)
                
                if datetime.now() - cache_time < max_age:
                    logger.info(f"Loading {symbol} {data_type} data from cache")
                    return pd.read_json(cache_data['data'], orient='index')
                    
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Cache file corrupted for {symbol}: {e}")
                
        return None
    
    def _save_to_cache(self, data: pd.DataFrame, symbol: str, data_type: str, interval: str = "daily"):
        """Save data to cache."""
        cache_file = self._get_cache_filename(symbol, data_type, interval)
        
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'data': data.to_json(orient='index')
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
                
            logger.debug(f"Saved {symbol} {data_type} data to cache")
        except Exception as e:
            logger.warning(f"Failed to save cache for {symbol}: {e}")
    
    def get_daily_data(self, symbol: str, outputsize: str = 'full') -> pd.DataFrame:
        """
        Fetch daily stock data for a given symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.BSE')
            outputsize: 'compact' (last 100 days) or 'full' (20+ years)
            
        Returns:
            DataFrame with OHLCV data
        """
        # Try to load from cache first
        cached_data = self._load_from_cache(symbol, 'daily')
        if cached_data is not None:
            return cached_data
        
        try:
            self._rate_limit()
            logger.info(f"Fetching daily data for {symbol}")
            
            data, meta_data = self.ts.get_daily(symbol=symbol, outputsize=outputsize)
            
            if data.empty:
                logger.error(f"No data received for {symbol}")
                return pd.DataFrame()
            
            # Clean column names and sort by date
            data.columns = ['open', 'high', 'low', 'close', 'volume']
            data.index = pd.to_datetime(data.index)
            data = data.sort_index()
            
            # Cache the data
            self._save_to_cache(data, symbol, 'daily')
            
            logger.success(f"Successfully fetched {len(data)} days of data for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching daily data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_intraday_data(self, symbol: str, interval: str = '5min') -> pd.DataFrame:
        """
        Fetch intraday stock data for a given symbol.
        
        Args:
            symbol: Stock symbol
            interval: Time interval ('1min', '5min', '15min', '30min', '60min')
            
        Returns:
            DataFrame with OHLCV data
        """
        # Try to load from cache first
        cached_data = self._load_from_cache(symbol, 'intraday', interval)
        if cached_data is not None:
            return cached_data
        
        try:
            self._rate_limit()
            logger.info(f"Fetching {interval} intraday data for {symbol}")
            
            data, meta_data = self.ts.get_intraday(symbol=symbol, interval=interval, outputsize='full')
            
            if data.empty:
                logger.error(f"No intraday data received for {symbol}")
                return pd.DataFrame()
            
            # Clean column names and sort by date
            data.columns = ['open', 'high', 'low', 'close', 'volume']
            data.index = pd.to_datetime(data.index)
            data = data.sort_index()
            
            # Cache the data
            self._save_to_cache(data, symbol, 'intraday', interval)
            
            logger.success(f"Successfully fetched {len(data)} intervals of intraday data for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching intraday data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_technical_indicators(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch technical indicators for a given symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary containing RSI, MACD, and other technical indicators
        """
        indicators = {}
        
        try:
            # RSI
            self._rate_limit()
            logger.info(f"Fetching RSI for {symbol}")
            rsi_data, _ = self.ti.get_rsi(symbol=symbol, interval='daily', time_period=Config.RSI_PERIOD)
            indicators['rsi'] = rsi_data
            
            # MACD
            self._rate_limit()
            logger.info(f"Fetching MACD for {symbol}")
            macd_data, _ = self.ti.get_macd(symbol=symbol, interval='daily')
            indicators['macd'] = macd_data
            
            # Simple Moving Averages
            self._rate_limit()
            logger.info(f"Fetching SMA 20 for {symbol}")
            sma20_data, _ = self.ti.get_sma(symbol=symbol, interval='daily', time_period=Config.SHORT_MA)
            indicators['sma_20'] = sma20_data
            
            self._rate_limit()
            logger.info(f"Fetching SMA 50 for {symbol}")
            sma50_data, _ = self.ti.get_sma(symbol=symbol, interval='daily', time_period=Config.LONG_MA)
            indicators['sma_50'] = sma50_data
            
            logger.success(f"Successfully fetched technical indicators for {symbol}")
            
        except Exception as e:
            logger.error(f"Error fetching technical indicators for {symbol}: {e}")
            
        return indicators
    
    def get_multiple_stocks_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks with proper rate limiting.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        stocks_data = {}
        
        for symbol in symbols:
            logger.info(f"Processing {symbol}...")
            
            try:
                data = self.get_daily_data(symbol)
                if not data.empty:
                    stocks_data[symbol] = data
                    logger.success(f"Successfully processed {symbol}")
                else:
                    logger.warning(f"No data available for {symbol}")
                    
            except Exception as e:
                logger.error(f"Failed to process {symbol}: {e}")
                continue
        
        logger.info(f"Successfully fetched data for {len(stocks_data)} out of {len(symbols)} stocks")
        return stocks_data
    
    def get_nifty50_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch data for configured NIFTY 50 stocks."""
        return self.get_multiple_stocks_data(Config.NIFTY_50_STOCKS)
    
    def validate_data_quality(self, data: pd.DataFrame, symbol: str) -> Dict[str, any]:
        """
        Validate the quality of fetched data.
        
        Args:
            data: DataFrame to validate
            symbol: Stock symbol for logging
            
        Returns:
            Dictionary with validation results
        """
        if data.empty:
            return {'valid': False, 'reason': 'No data available'}
        
        validation_results = {
            'valid': True,
            'symbol': symbol,
            'data_points': len(data),
            'date_range': {
                'start': data.index.min().strftime('%Y-%m-%d'),
                'end': data.index.max().strftime('%Y-%m-%d')
            },
            'missing_values': data.isnull().sum().to_dict(),
            'warnings': []
        }
        
        # Check for missing values
        if data.isnull().any().any():
            validation_results['warnings'].append('Missing values detected')
        
        # Check for unrealistic price movements (>50% in a day)
        if 'close' in data.columns:
            daily_returns = data['close'].pct_change()
            extreme_moves = abs(daily_returns) > 0.5
            if extreme_moves.any():
                validation_results['warnings'].append(f'{extreme_moves.sum()} days with >50% price movement')
        
        # Check data recency (should be within last 7 days for daily data)
        last_date = data.index.max()
        days_old = (datetime.now() - last_date).days
        if days_old > 7:
            validation_results['warnings'].append(f'Data is {days_old} days old')
        
        return validation_results