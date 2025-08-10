"""
Configuration module for the algo-trading prototype.
Handles environment variables and application settings.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the algo-trading application."""
    
    # Alpha Vantage API Configuration
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
    
    # Google Sheets Configuration
    GOOGLE_SHEETS_CREDENTIALS_FILE = os.getenv('GOOGLE_SHEETS_CREDENTIALS_FILE', 'credentials/google_sheets_credentials.json')
    GOOGLE_SHEETS_SPREADSHEET_ID = os.getenv('GOOGLE_SHEETS_SPREADSHEET_ID')
    
    # Telegram Bot Configuration
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
    
    # Trading Configuration
    INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', 100000))
    RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', 0.02))
    MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', 5))
    
    # Data Configuration
    DATA_SOURCE = os.getenv('DATA_SOURCE', 'alpha_vantage')
    UPDATE_FREQUENCY = os.getenv('UPDATE_FREQUENCY', '1D')
    BACKTEST_PERIOD = os.getenv('BACKTEST_PERIOD', '6M')
    
    # NIFTY 50 Stocks (top 3 for this prototype)
    NIFTY_50_STOCKS = [
        'RELIANCE.BSE',  # Reliance Industries
        'TCS.BSE',       # Tata Consultancy Services
        'INFY.BSE'       # Infosys
    ]
    
    # Technical Indicator Parameters
    RSI_PERIOD = 14
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    SHORT_MA = 20
    LONG_MA = 50
    
    # ML Model Parameters
    ML_FEATURES = ['rsi', 'macd', 'volume_ratio', 'price_change', 'volatility']
    TRAIN_TEST_SPLIT = 0.8
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = 'logs/trading_bot.log'
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Validate that all required configuration is present."""
        validation_results = {
            'valid': True,
            'missing_configs': [],
            'warnings': []
        }
        
        if not cls.ALPHA_VANTAGE_API_KEY:
            validation_results['missing_configs'].append('ALPHA_VANTAGE_API_KEY')
            validation_results['valid'] = False
            
        if not cls.GOOGLE_SHEETS_SPREADSHEET_ID:
            validation_results['warnings'].append('GOOGLE_SHEETS_SPREADSHEET_ID not set - Google Sheets integration will be disabled')
            
        if not cls.TELEGRAM_BOT_TOKEN:
            validation_results['warnings'].append('TELEGRAM_BOT_TOKEN not set - Telegram alerts will be disabled')
            
        return validation_results
    
    @classmethod
    def get_all_config(cls) -> Dict[str, Any]:
        """Get all configuration as a dictionary."""
        return {
            'alpha_vantage_api_key': cls.ALPHA_VANTAGE_API_KEY[:10] + '...' if cls.ALPHA_VANTAGE_API_KEY else None,
            'initial_capital': cls.INITIAL_CAPITAL,
            'risk_per_trade': cls.RISK_PER_TRADE,
            'max_positions': cls.MAX_POSITIONS,
            'nifty_stocks': cls.NIFTY_50_STOCKS,
            'rsi_parameters': {
                'period': cls.RSI_PERIOD,
                'oversold': cls.RSI_OVERSOLD,
                'overbought': cls.RSI_OVERBOUGHT
            },
            'ma_parameters': {
                'short_ma': cls.SHORT_MA,
                'long_ma': cls.LONG_MA
            }
        }