#!/usr/bin/env python3
"""
Main entry point for the Algo-Trading Prototype.
Provides command-line interface for running different components of the trading system.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Handle missing dependencies gracefully
try:
    from src.automation.trading_engine import TradingEngine
    from src.data.data_fetcher import DataFetcher
    from src.strategies.rsi_ma_strategy import RSIMACrossoverStrategy
    from src.ml.predictive_model import StockPredictiveModel
    from src.config import Config
    from loguru import logger
    CORE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Core modules not available: {e}")
    CORE_AVAILABLE = False

# Try to import Google Sheets logger, fall back to mock
try:
    from src.utils.sheets_logger import GoogleSheetsLogger
    SHEETS_AVAILABLE = True
except ImportError:
    try:
        from src.utils.mock_sheets_logger import MockGoogleSheetsLogger as GoogleSheetsLogger
        SHEETS_AVAILABLE = True
        print("Note: Using mock Google Sheets logger (dependencies not installed)")
    except ImportError:
        SHEETS_AVAILABLE = False

# Try to import Telegram bot
try:
    from src.utils.telegram_alerts import TelegramAlertsBot
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False


def setup_logging():
    """Setup logging for the main script."""
    logger.add(
        "logs/main.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="INFO",
        rotation="10 MB"
    )


def run_engine():
    """Run the main trading engine."""
    logger.info("Starting Algo-Trading Engine...")
    engine = TradingEngine()
    engine.start_automated_trading()


def run_backtest():
    """Run strategy backtesting for exactly 6 months with Google Sheets integration."""
    if not CORE_AVAILABLE:
        print("❌ Core modules not available. Please install dependencies:")
        print("pip install pandas numpy alpha-vantage scikit-learn matplotlib loguru")
        return
    
    logger.info("Running 6-month backtest...")
    
    # Initialize components
    try:
        data_fetcher = DataFetcher()
        # Use simple improved strategy for better performance (works with existing indicators)
        from src.strategies.simple_improved_strategy import SimpleImprovedStrategy
        strategy = SimpleImprovedStrategy(
            initial_capital=Config.INITIAL_CAPITAL,
            risk_per_trade=Config.RISK_PER_TRADE * 0.75  # Reduced risk (1.5% instead of 2%)
        )
        logger.info("Using SIMPLE IMPROVED RSI+MA strategy for backtesting")
    except Exception as e:
        print(f"❌ Failed to initialize components: {e}")
        print("Please check your configuration and dependencies.")
        return
    
    # Initialize Google Sheets logger (optional)
    sheets_logger = None
    if SHEETS_AVAILABLE:
        try:
            sheets_logger = GoogleSheetsLogger()
            if hasattr(sheets_logger, 'is_connected') and sheets_logger.is_connected():
                logger.info("Google Sheets integration enabled for backtest logging")
            else:
                logger.warning("Google Sheets not configured - using mock logger for demo")
        except Exception as e:
            logger.warning(f"Google Sheets integration failed: {e}")
    
    # Fetch data
    logger.info("Fetching historical data for 6-month backtest...")
    try:
        data = data_fetcher.get_nifty50_data()
    except Exception as e:
        print(f"❌ Failed to fetch data: {e}")
        print("Please check your Alpha Vantage API key configuration.")
        return
    
    if not data:
        logger.error("No data available for backtesting")
        return
    
    # Filter data to exactly 6 months from the latest available date
    from datetime import datetime, timedelta
    
    # Find the latest date and go back exactly 6 months
    latest_date = None
    six_month_data = {}
    
    for symbol, stock_data in data.items():
        if not stock_data.empty:
            stock_latest = stock_data.index.max()
            if latest_date is None or stock_latest > latest_date:
                latest_date = stock_latest
    
    if latest_date is None:
        logger.error("No valid data found")
        return
    
    # Calculate 6 months back (approximately 180 trading days)
    start_date = latest_date - timedelta(days=180)
    
    logger.info(f"Backtest period: {start_date.date()} to {latest_date.date()}")
    
    # Filter data for each stock to 6-month period
    for symbol, stock_data in data.items():
        if not stock_data.empty:
            # Filter to 6-month period
            mask = (stock_data.index >= start_date) & (stock_data.index <= latest_date)
            filtered_data = stock_data[mask]
            
            if len(filtered_data) > 0:
                six_month_data[symbol] = filtered_data
                logger.info(f"Filtered {symbol}: {len(filtered_data)} trading days")
    
    if not six_month_data:
        logger.error("No data available for the 6-month period")
        return
    
    # Run backtest with 6-month data
    logger.info("Running strategy backtest...")
    try:
        results = strategy.backtest(six_month_data, start_date=start_date.strftime('%Y-%m-%d'))
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        return
    
    # Enhanced results display
    print("\n" + "="*60)
    print("6-MONTH BACKTEST RESULTS")
    print("="*60)
    print(f"Period: {start_date.date()} to {latest_date.date()}")
    print(f"Duration: ~6 months ({(latest_date - start_date).days} days)")
    print(f"Total Return: {results.get('total_return', 0):.2%}")
    print(f"Annualized Return: {results.get('annualized_return', 0):.2%}")
    print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown: {results.get('max_drawdown', 0):.2%}")
    print(f"Win Rate: {results.get('win_rate', 0):.2%}")
    print(f"Total Trades: {results.get('total_trades', 0)}")
    print(f"Winning Trades: {results.get('winning_trades', 0)}")
    print(f"Losing Trades: {results.get('losing_trades', 0)}")
    
    if results.get('trades'):
        avg_win = sum([t.pnl for t in results['trades'] if t.pnl > 0]) / max(1, results.get('winning_trades', 1))
        avg_loss = sum([t.pnl for t in results['trades'] if t.pnl < 0]) / max(1, results.get('losing_trades', 1))
        print(f"Average Win: ₹{avg_win:.2f}")
        print(f"Average Loss: ₹{avg_loss:.2f}")
        print(f"Profit Factor: {abs(avg_win/avg_loss) if avg_loss != 0 else 'N/A':.2f}")
    
    print("="*60)
    
    # Log results to Google Sheets if available
    if sheets_logger:
        logger.info("Logging backtest results to Google Sheets...")
        try:
            # Log individual trades
            if results.get('trades'):
                for trade in results['trades']:
                    trade_data = {
                        'timestamp': trade.timestamp,
                        'symbol': trade.symbol,
                        'action': trade.action,
                        'quantity': trade.quantity,
                        'entry_price': trade.price,
                        'exit_price': trade.exit_price,
                        'entry_date': trade.timestamp.strftime('%Y-%m-%d') if trade.timestamp else '',
                        'exit_date': trade.exit_timestamp.strftime('%Y-%m-%d') if trade.exit_timestamp else '',
                        'exit_reason': trade.exit_reason or 'N/A',
                        'pnl': trade.pnl,
                        'stop_loss': trade.stop_loss,
                        'take_profit': trade.take_profit
                    }
                    sheets_logger.log_trade(trade_data)
            
            # Log final portfolio summary
            portfolio_summary = {
                'date': latest_date,
                'total_capital': strategy.current_capital,
                'total_pnl': strategy.current_capital - strategy.initial_capital,
                'cumulative_return': results.get('total_return', 0),
                'active_positions': len(strategy.positions),
                'max_drawdown': results.get('max_drawdown', 0),
                'win_rate': results.get('win_rate', 0)
            }
            sheets_logger.update_portfolio_summary(portfolio_summary)
            
            # Log performance metrics
            performance_metrics = {
                'total_return': results.get('total_return', 0),
                'annualized_return': results.get('annualized_return', 0),
                'sharpe_ratio': results.get('sharpe_ratio', 0),
                'max_drawdown': results.get('max_drawdown', 0),
                'win_rate': results.get('win_rate', 0),
                'total_trades': results.get('total_trades', 0),
                'winning_trades': results.get('winning_trades', 0),
                'losing_trades': results.get('losing_trades', 0)
            }
            sheets_logger.update_performance_metrics(performance_metrics)
            
            print("\n✅ Backtest results logged to Google Sheets successfully!")
            print("Check your Google Sheet for detailed trade logs and performance data.")
            
        except Exception as e:
            logger.error(f"Failed to log to Google Sheets: {e}")
            print(f"\n❌ Google Sheets logging failed: {e}")
    
    # Plot results
    try:
        strategy.plot_backtest_results(results, save_path="backtest_results.png")
        print(f"\n📊 Backtest chart saved as 'backtest_results.png'")
    except Exception as e:
        logger.warning(f"Could not generate plot: {e}")
    
    return results


def train_ml_model():
    """Train the ML prediction model."""
    logger.info("Training ML model...")
    
    # Initialize components
    data_fetcher = DataFetcher()
    ml_model = StockPredictiveModel(model_type='random_forest')
    
    # Fetch training data
    logger.info("Fetching training data...")
    data = data_fetcher.get_nifty50_data()
    
    if not data:
        logger.error("No data available for training")
        return
    
    # Train model
    results = ml_model.train(data)
    
    # Display results
    print("\n" + "="*50)
    print("ML MODEL TRAINING RESULTS")
    print("="*50)
    print(f"Model Type: {ml_model.model_type}")
    print(f"Training Accuracy: {results.get('train_accuracy', 0):.4f}")
    print(f"Test Accuracy: {results.get('test_accuracy', 0):.4f}")
    print(f"Cross-Validation: {results.get('cv_mean', 0):.4f} ± {results.get('cv_std', 0):.4f}")
    print(f"Number of Features: {len(ml_model.feature_names)}")
    print("="*50)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    ml_model.save_model('models/trading_model.pkl')
    print("Model saved to 'models/trading_model.pkl'")
    
    # Plot analysis
    try:
        ml_model.plot_model_analysis(save_path="ml_model_analysis.png")
        print("Model analysis chart saved as 'ml_model_analysis.png'")
    except Exception as e:
        logger.warning(f"Could not generate plot: {e}")


def scan_market():
    """Run a single market scan."""
    logger.info("Running market scan...")
    
    engine = TradingEngine()
    results = engine.run_single_scan()
    
    print("\n" + "="*50)
    print("MARKET SCAN RESULTS")
    print("="*50)
    print(f"Timestamp: {results['timestamp']}")
    print(f"Symbols Scanned: {results['symbols_scanned']}")
    print(f"Signals Generated: {results['signals_generated']}")
    print(f"Errors: {results['errors']}")
    
    if results['signals']:
        print("\nSIGNALS DETECTED:")
        for symbol, signal in results['signals'].items():
            if signal['signal'] != 'NONE':
                print(f"  {symbol}: {signal['signal']} (Strength: {signal['strength']})")
                print(f"    Price: ₹{signal['price']:.2f}, RSI: {signal['rsi']:.2f}")
    
    print("="*50)


def test_integrations():
    """Test external integrations."""
    logger.info("Testing integrations...")
    
    print("\n" + "="*50)
    print("INTEGRATION TESTS")
    print("="*50)
    
    # Test Alpha Vantage API
    print("Testing Alpha Vantage API...")
    try:
        data_fetcher = DataFetcher()
        test_data = data_fetcher.get_daily_data("RELIANCE.BSE", outputsize="compact")
        if not test_data.empty:
            print("✅ Alpha Vantage API: Connected")
            print(f"   Sample data points: {len(test_data)}")
        else:
            print("❌ Alpha Vantage API: No data received")
    except Exception as e:
        print(f"❌ Alpha Vantage API: {str(e)}")
    
    # Test Google Sheets
    print("\nTesting Google Sheets...")
    try:
        sheets_logger = GoogleSheetsLogger()
        if sheets_logger.is_connected():
            print("✅ Google Sheets: Connected")
            status = sheets_logger.get_connection_status()
            print(f"   Worksheets: {status['worksheets_configured']}")
        else:
            print("❌ Google Sheets: Not connected")
    except Exception as e:
        print(f"❌ Google Sheets: {str(e)}")
    
    # Test Telegram Bot
    print("\nTesting Telegram Bot...")
    try:
        telegram_bot = TelegramAlertsBot()
        if telegram_bot.is_enabled:
            test_result = telegram_bot.test_connection()
            if test_result:
                print("✅ Telegram Bot: Connected")
            else:
                print("❌ Telegram Bot: Connection failed")
        else:
            print("❌ Telegram Bot: Not configured")
    except Exception as e:
        print(f"❌ Telegram Bot: {str(e)}")
    
    print("="*50)


def show_config():
    """Display current configuration."""
    config = Config.get_all_config()
    validation = Config.validate_config()
    
    print("\n" + "="*50)
    print("CONFIGURATION STATUS")
    print("="*50)
    print(f"Status: {'✅ Valid' if validation['valid'] else '❌ Invalid'}")
    
    if validation['missing_configs']:
        print(f"Missing: {', '.join(validation['missing_configs'])}")
    
    if validation['warnings']:
        print("Warnings:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    
    print(f"\nCapital: ₹{config['initial_capital']:,.2f}")
    print(f"Risk per Trade: {config['risk_per_trade']:.2%}")
    print(f"Max Positions: {config['max_positions']}")
    print(f"Stocks Monitored: {len(config['nifty_stocks'])}")
    
    print("\nStocks:")
    for stock in config['nifty_stocks']:
        print(f"  - {stock}")
    
    print("\nTechnical Parameters:")
    print(f"  RSI Period: {config['rsi_parameters']['period']}")
    print(f"  RSI Oversold: {config['rsi_parameters']['oversold']}")
    print(f"  RSI Overbought: {config['rsi_parameters']['overbought']}")
    print(f"  Short MA: {config['ma_parameters']['short_ma']}")
    print(f"  Long MA: {config['ma_parameters']['long_ma']}")
    
    print("="*50)


def run_original_backtest():
    """Run backtest with the original strategy for comparison."""
    if not CORE_AVAILABLE:
        print("❌ Core modules not available. Please install dependencies:")
        print("pip install pandas numpy alpha-vantage scikit-learn matplotlib loguru")
        return
    
    logger.info("Running 6-month backtest with ORIGINAL strategy...")
    
    # Initialize components with original strategy
    try:
        data_fetcher = DataFetcher()
        strategy = RSIMACrossoverStrategy()  # Original strategy
        logger.info("Using ORIGINAL RSI+MA strategy for backtesting")
    except Exception as e:
        print(f"❌ Failed to initialize components: {e}")
        return
    
    # Fetch and run backtest (same logic as improved backtest)
    try:
        data = data_fetcher.get_nifty50_data()
        if not data:
            logger.error("No data available for backtesting")
            return
        
        # Filter to 6 months
        from datetime import datetime, timedelta
        latest_date = max(stock_data.index.max() for stock_data in data.values() if not stock_data.empty)
        start_date = latest_date - timedelta(days=180)
        
        six_month_data = {}
        for symbol, stock_data in data.items():
            if not stock_data.empty:
                mask = (stock_data.index >= start_date) & (stock_data.index <= latest_date)
                filtered_data = stock_data[mask]
                if len(filtered_data) > 0:
                    six_month_data[symbol] = filtered_data
        
        # Run backtest
        results = strategy.backtest(six_month_data, start_date=start_date.strftime('%Y-%m-%d'))
        
        # Display results
        print("\n" + "="*60)
        print("ORIGINAL STRATEGY - 6-MONTH BACKTEST RESULTS")
        print("="*60)
        print(f"Period: {start_date.date()} to {latest_date.date()}")
        print(f"Total Return: {results.get('total_return', 0):.2%}")
        print(f"Win Rate: {results.get('win_rate', 0):.2%}")
        print(f"Max Drawdown: {results.get('max_drawdown', 0):.2%}")
        print(f"Total Trades: {results.get('total_trades', 0)}")
        print("="*60)
        print("💡 Compare with: python main.py backtest (improved strategy)")
        
    except Exception as e:
        print(f"❌ Original backtest failed: {e}")


def compare_strategies():
    """Compare original and improved trading strategies."""
    if not CORE_AVAILABLE:
        print("❌ Core modules not available")
        print("Please install dependencies: pip install -r requirements.txt")
        return
    
    try:
        # Import the comparison script functionality
        from test_improved_strategy import run_strategy_comparison
        
        print("🔄 Starting strategy comparison...")
        results = run_strategy_comparison()
        
        if results:
            print("\n✅ Strategy comparison completed successfully!")
            print("📊 Check the output above for detailed performance metrics.")
        else:
            print("\n❌ Strategy comparison failed.")
            
    except ImportError as e:
        print(f"❌ Could not import improved strategy: {e}")
    except Exception as e:
        print(f"❌ Strategy comparison failed: {e}")
        logger.error(f"Strategy comparison error: {e}")


def cleanup_repository():
    """Clean up unnecessary files from the repository."""
    try:
        from cleanup_system import RepositoryCleanup
        
        print("🧹 Starting repository cleanup...")
        print("This will remove cache files, logs, and temporary files.")
        
        # Ask for confirmation
        response = input("\nProceed with cleanup? (y/N): ").strip().lower()
        
        if response in ['y', 'yes']:
            cleanup = RepositoryCleanup(dry_run=False)
            cleanup.run_full_cleanup(include_cache=True, include_logs=True)
        else:
            print("Cleanup cancelled.")
            
    except ImportError as e:
        print(f"❌ Could not import cleanup system: {e}")
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        if CORE_AVAILABLE:
            logger.error(f"Cleanup error: {e}")


def main():
    """Main function with command-line interface."""
    setup_logging()
    
    parser = argparse.ArgumentParser(
        description="Algo-Trading Prototype - RSI + MA Strategy with ML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py run                 # Start the trading engine
  python main.py backtest           # Run strategy backtesting
  python main.py train-ml           # Train ML prediction model
  python main.py scan               # Run single market scan
  python main.py test               # Test integrations
  python main.py config             # Show configuration
  python main.py compare-strategies  # Compare original vs improved strategy
  python main.py cleanup            # Clean up unnecessary files
  python main.py backtest-original  # Run backtest with original strategy
        """
    )
    
    parser.add_argument(
        'command',
        choices=['run', 'backtest', 'train-ml', 'scan', 'test', 'config', 'compare-strategies', 'cleanup', 'backtest-original'],
        help='Command to execute'
    )
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('cache', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    try:
        if args.command == 'run':
            run_engine()
        elif args.command == 'backtest':
            run_backtest()
        elif args.command == 'train-ml':
            train_ml_model()
        elif args.command == 'scan':
            scan_market()
        elif args.command == 'test':
            test_integrations()
        elif args.command == 'config':
            show_config()
        elif args.command == 'compare-strategies':
            compare_strategies()
        elif args.command == 'cleanup':
            cleanup_repository()
        elif args.command == 'backtest-original':
            run_original_backtest()
    
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        print("\nOperation cancelled by user.")
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()