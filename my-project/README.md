# Algo-Trading Prototype 🚀

A comprehensive Python-based algorithmic trading system implementing RSI + Moving Average crossover strategy with machine learning predictions, automated Google Sheets logging, and Telegram alerts.

## 🎯 Project Overview

This project implements a complete algo-trading solution with the following key features:

- **Data Ingestion**: Real-time stock data from Alpha Vantage API for NIFTY 50 stocks
- **Trading Strategy**: RSI < 30 buy signals confirmed by 20-DMA crossing above 50-DMA
- **Machine Learning**: Random Forest model for next-day price movement prediction
- **Automation**: Scheduled market scanning and signal generation
- **Google Sheets Integration**: Automatic trade logging and P&L tracking
- **Telegram Alerts**: Real-time notifications for signals and trades
- **Backtesting**: Comprehensive 6-month historical strategy validation

## 📊 Key Features

### 🔍 Data & Analysis
- Alpha Vantage API integration with caching
- Technical indicators: RSI, MACD, Moving Averages, Bollinger Bands, ATR
- Real-time market data processing for 3 NIFTY 50 stocks (RELIANCE, TCS, INFOSYS)

### 📈 Trading Strategy
- **Entry Signal**: RSI < 30 AND 20-DMA crosses above 50-DMA
- **Exit Signal**: RSI > 70 OR 20-DMA crosses below 50-DMA OR Stop Loss/Take Profit
- Risk management with position sizing and stop losses
- Win rate tracking and performance analytics

### 🤖 Machine Learning
- Random Forest classifier for price direction prediction
- Technical features engineering (30+ features)
- Model validation with cross-validation and accuracy metrics
- Real-time prediction confidence scoring

### 📋 Google Sheets Automation
- **Trade Log**: Complete trade history with entry/exit details
- **Portfolio Summary**: Daily portfolio value and performance metrics
- **Signal Log**: All generated signals with ML predictions
- **Performance Metrics**: Key strategy statistics
- **Dashboard**: Real-time performance visualization

### 📱 Telegram Integration
- Trading signal alerts with emoji indicators
- Trade execution notifications
- Daily performance summaries
- Error alerts and system status updates
- Portfolio performance updates

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Alpha Vantage API key (free tier available)
- Google Sheets API credentials (optional)
- Telegram Bot token (optional)

### 1. Clone and Install
```bash
git clone <repository-url>
cd algo-trading-prototype
pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file from the template:
```bash
cp .env.example .env
```

Edit `.env` with your API keys:
```env
# Required
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here

# Optional (for full functionality)
GOOGLE_SHEETS_CREDENTIALS_FILE=credentials/google_sheets_credentials.json
GOOGLE_SHEETS_SPREADSHEET_ID=your_google_sheets_spreadsheet_id
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id

# Trading Configuration
INITIAL_CAPITAL=100000
RISK_PER_TRADE=0.02
MAX_POSITIONS=5
```

### 3. API Keys Setup

#### Alpha Vantage (Required)
1. Get free API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. Add to `.env` file

#### Google Sheets (Optional)
1. Create Google Cloud project
2. Enable Google Sheets API
3. Create service account and download credentials JSON
4. Place in `credentials/google_sheets_credentials.json`
5. Share your spreadsheet with the service account email

#### Telegram Bot (Optional)
1. Create bot via [@BotFather](https://t.me/BotFather)
2. Get bot token and your chat ID
3. Add to `.env` file

## 🚀 Usage

### Command Line Interface

```bash
# Show configuration status
python main.py config

# Test API connections
python main.py test

# Run strategy backtest
python main.py backtest

# Train ML model
python main.py train-ml

# Run single market scan
python main.py scan

# Start automated trading engine
python main.py run
```

### Example Usage

#### 1. Configuration Check
```bash
python main.py config
```
Output:
```
==================================================
CONFIGURATION STATUS
==================================================
Status: ✅ Valid
Capital: ₹1,00,000.00
Risk per Trade: 2.00%
Max Positions: 5
Stocks Monitored: 3

Stocks:
  - RELIANCE.BSE
  - TCS.BSE
  - INFY.BSE
==================================================
```

#### 2. Integration Testing
```bash
python main.py test
```
Output:
```
==================================================
INTEGRATION TESTS
==================================================
Testing Alpha Vantage API...
✅ Alpha Vantage API: Connected
   Sample data points: 100

Testing Google Sheets...
✅ Google Sheets: Connected
   Worksheets: 4

Testing Telegram Bot...
✅ Telegram Bot: Connected
==================================================
```

#### 3. Strategy Backtesting
```bash
python main.py backtest
```
Output:
```
==================================================
BACKTEST RESULTS
==================================================
Period: 2023-06-01 to 2023-12-01
Total Return: 15.67%
Annualized Return: 31.34%
Sharpe Ratio: 1.45
Max Drawdown: -8.23%
Win Rate: 67.50%
Total Trades: 24
Winning Trades: 16
Losing Trades: 8
==================================================
```

#### 4. ML Model Training
```bash
python main.py train-ml
```
Output:
```
==================================================
ML MODEL TRAINING RESULTS
==================================================
Model Type: random_forest
Training Accuracy: 0.8245
Test Accuracy: 0.7632
Cross-Validation: 0.7584 ± 0.0321
Number of Features: 28
==================================================
```

## 📁 Project Structure

```
algo-trading-prototype/
├── src/
│   ├── config.py                 # Configuration management
│   ├── data/
│   │   ├── data_fetcher.py      # Alpha Vantage API integration
│   │   └── technical_indicators.py # Technical analysis
│   ├── strategies/
│   │   └── rsi_ma_strategy.py   # Trading strategy implementation
│   ├── ml/
│   │   └── predictive_model.py  # Machine learning models
│   ├── utils/
│   │   ├── sheets_logger.py     # Google Sheets automation
│   │   └── telegram_alerts.py  # Telegram bot integration
│   └── automation/
│       └── trading_engine.py    # Main orchestration engine
├── main.py                      # Command-line interface
├── requirements.txt             # Python dependencies
├── .env.example                # Environment template
├── README.md                   # This file
└── examples/                   # Example notebooks and scripts
```

## 🔧 Core Components

### 1. Data Fetcher (`src/data/data_fetcher.py`)
- Alpha Vantage API integration with rate limiting
- Data caching for efficiency
- Multiple timeframe support (daily, intraday)
- Data quality validation

### 2. Technical Indicators (`src/data/technical_indicators.py`)
- RSI, MACD, Moving Averages, Bollinger Bands
- Volume indicators and volatility metrics
- Support/resistance levels
- Signal strength calculation

### 3. Trading Strategy (`src/strategies/rsi_ma_strategy.py`)
- RSI + MA crossover implementation
- Position sizing and risk management
- Stop loss and take profit logic
- Comprehensive backtesting framework

### 4. ML Model (`src/ml/predictive_model.py`)
- Feature engineering from technical indicators
- Random Forest, Decision Tree, Logistic Regression
- Model validation and performance metrics
- Prediction confidence scoring

### 5. Trading Engine (`src/automation/trading_engine.py`)
- Component orchestration
- Scheduled market scanning
- Signal generation and filtering
- Performance monitoring

## 📊 Strategy Details

### Entry Conditions
1. **RSI < 30** (Oversold condition)
2. **20-DMA crosses above 50-DMA** (Bullish crossover)
3. **Volume confirmation** (Above average volume)

### Exit Conditions
1. **RSI > 70** (Overbought condition)
2. **20-DMA crosses below 50-DMA** (Bearish crossover)
3. **Stop Loss**: 2 × ATR below entry price
4. **Take Profit**: 3 × ATR above entry price

### Risk Management
- **Position Sizing**: Based on ATR and account risk (default 2%)
- **Maximum Positions**: 5 concurrent positions
- **Stop Loss**: Mandatory on all positions
- **Cash Buffer**: 5% minimum cash reserve

## 🤖 Machine Learning Features

The ML model uses 30+ engineered features including:

### Price-based Features
- Price momentum (5, 10, 20 days)
- Support/resistance distances
- High/low ratios

### Technical Indicators
- RSI momentum and moving averages
- MACD momentum and signal differences
- Bollinger Bands position and squeeze

### Volume Features
- Volume momentum and ratios
- Price-volume trends
- On-balance volume

### Market Regime Features
- Trend strength indicators
- Volatility regime classification
- Stochastic oscillator states

## 📈 Performance Metrics

The system tracks comprehensive performance metrics:

### Strategy Metrics
- Total and annualized returns
- Sharpe ratio and volatility
- Maximum drawdown
- Win rate and profit factor

### Trade Metrics
- Average win/loss amounts
- Trade duration analysis
- Success rate by signal strength
- Risk-adjusted returns

### System Metrics
- API call efficiency
- Error rates and uptime
- Memory and CPU usage
- Data freshness indicators

## 🔄 Automation Schedule

The trading engine runs on the following schedule:

### Market Hours (9:15 AM - 3:30 PM IST)
- **Market Scans**: 9:15 AM, 11:00 AM, 1:00 PM, 3:00 PM
- **Signal Generation**: Real-time during scans
- **Trade Monitoring**: Continuous during market hours

### After Hours
- **Daily Summary**: 4:00 PM (market close)
- **Google Sheets Update**: 4:15 PM
- **System Health Check**: Every hour

### Weekly Tasks
- **Strategy Backtest**: Saturday 10:00 AM
- **ML Model Retraining**: As needed
- **Performance Reporting**: Weekly summaries

## 📱 Google Sheets Integration

The system automatically maintains four worksheets:

### 1. Trade Log
- Complete trade history with entry/exit details
- P&L calculations and trade duration
- Signal strength and ML predictions
- Exit reasons and performance metrics

### 2. Portfolio Summary
- Daily portfolio value tracking
- Cash and position values
- Return calculations and drawdown
- Active position counts

### 3. Signal Log
- All generated signals with timestamps
- Technical indicator values
- ML predictions and confidence
- Action taken for each signal

### 4. Performance Metrics
- Key strategy statistics
- Risk-adjusted returns
- Win rates and profit factors
- System performance indicators

## 🔔 Telegram Alerts

The Telegram bot sends various types of alerts:

### Signal Alerts
- Buy/Sell signals with strength indicators
- Price and technical indicator values
- ML predictions with confidence scores
- Actionable recommendations

### Trade Alerts
- Trade execution notifications
- Position updates and P&L
- Stop loss and take profit hits
- Portfolio performance updates

### System Alerts
- Daily trading summaries
- Error notifications
- System status updates
- Performance milestones

## ⚙️ Configuration Options

### Trading Parameters
```python
INITIAL_CAPITAL = 100000      # Starting capital
RISK_PER_TRADE = 0.02        # Risk 2% per trade
MAX_POSITIONS = 5            # Maximum concurrent positions
```

### Technical Indicators
```python
RSI_PERIOD = 14              # RSI calculation period
RSI_OVERSOLD = 30           # RSI buy threshold
RSI_OVERBOUGHT = 70         # RSI sell threshold
SHORT_MA = 20               # Short moving average
LONG_MA = 50                # Long moving average
```

### ML Model
```python
ML_FEATURES = [              # Features for ML model
    'rsi', 'macd', 'volume_ratio', 
    'price_change', 'volatility'
]
TRAIN_TEST_SPLIT = 0.8      # 80% training, 20% testing
```

## 📚 Examples

### Basic Market Scan
```python
from src.automation.trading_engine import TradingEngine

engine = TradingEngine()
results = engine.run_single_scan()
print(f"Generated {results['signals_generated']} signals")
```

### Manual Backtesting
```python
from src.data.data_fetcher import DataFetcher
from src.strategies.rsi_ma_strategy import RSIMACrossoverStrategy

# Fetch data
data_fetcher = DataFetcher()
data = data_fetcher.get_nifty50_data()

# Run backtest
strategy = RSIMACrossoverStrategy()
results = strategy.backtest(data, start_date='2023-01-01')
print(f"Total Return: {results['total_return']:.2%}")
```

### ML Prediction
```python
from src.ml.predictive_model import StockPredictiveModel

# Load trained model
model = StockPredictiveModel()
model.load_model('models/trading_model.pkl')

# Make prediction
prediction = model.predict(stock_data)
print(f"Direction: {'UP' if prediction['prediction'] else 'DOWN'}")
print(f"Confidence: {prediction['confidence']:.2%}")
```

## 🚨 Important Notes

### API Limits
- **Alpha Vantage Free Tier**: 5 requests per minute, 500 per day
- The system implements automatic rate limiting
- Consider premium tier for production use

### Risk Disclaimer
- This is a prototype for educational purposes
- Not financial advice - use at your own risk
- Past performance doesn't guarantee future results
- Always validate strategies before live trading

### Data Accuracy
- Market data may have delays
- Technical indicators calculated from available data
- ML predictions are probabilistic, not guaranteed

## 🔧 Troubleshooting

### Common Issues

#### API Connection Failed
```bash
# Check API key configuration
python main.py config

# Test API connection
python main.py test
```

#### Google Sheets Not Working
1. Verify service account email has access to spreadsheet
2. Check credentials file path
3. Ensure Google Sheets API is enabled

#### Telegram Alerts Not Sending
1. Verify bot token and chat ID
2. Check if bot is blocked
3. Test connection with `python main.py test`

#### ML Model Training Fails
- Ensure sufficient historical data
- Check for missing technical indicators
- Verify data quality and completeness

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Alpha Vantage for market data API
- Google Sheets API for data logging
- Telegram Bot API for notifications
- scikit-learn for machine learning capabilities
- The open-source Python trading community

## 📞 Support

For questions, issues, or contributions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the example code

---

**Happy Trading! 🚀📈**

*Remember: This is a prototype for educational purposes. Always conduct thorough testing and risk assessment before any live trading.*