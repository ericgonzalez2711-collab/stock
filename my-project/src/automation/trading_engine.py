"""
Main trading automation engine that orchestrates all components.
Implements auto-triggered scanning, signal generation, and execution coordination.
"""

import pandas as pd
import numpy as np
import time
import schedule
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from loguru import logger
import traceback
import os
import psutil

from ..data.data_fetcher import DataFetcher
from ..data.technical_indicators import TechnicalIndicators
from ..strategies.rsi_ma_strategy import RSIMACrossoverStrategy, Trade
from ..strategies.simple_improved_strategy import SimpleImprovedStrategy
from ..ml.predictive_model import StockPredictiveModel
from ..utils.sheets_logger import GoogleSheetsLogger
from ..utils.telegram_alerts import TelegramAlertsBot
from ..config import Config


class TradingEngine:
    """Main trading automation engine."""
    
    def __init__(self, use_improved_strategy: bool = True):
        """Initialize the trading engine with all components."""
        self.config = Config()
        self.use_improved_strategy = use_improved_strategy
        
        # Initialize components
        self.data_fetcher = None
        self.strategy = None
        self.ml_model = None
        self.sheets_logger = None
        self.telegram_bot = None
        
        # Engine state
        self.is_running = False
        self.last_scan_time = None
        self.scan_count = 0
        self.error_count = 0
        self.api_call_count = 0
        self.start_time = datetime.now()
        
        # Performance tracking
        self.daily_stats = {
            'trades_today': 0,
            'signals_generated': 0,
            'pnl_today': 0,
            'errors_today': 0
        }
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize components
        self._initialize_components()
        
        logger.info("Trading Engine initialized successfully")
    
    def _setup_logging(self):
        """Setup comprehensive logging."""
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Configure loguru
        logger.add(
            Config.LOG_FILE,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
            level=Config.LOG_LEVEL,
            rotation="10 MB",
            retention="30 days",
            compression="zip"
        )
        
        logger.success("Logging system configured")
    
    def _initialize_components(self):
        """Initialize all trading components."""
        try:
            # Data fetcher
            logger.info("Initializing data fetcher...")
            self.data_fetcher = DataFetcher()
            
            # Trading strategy
            if self.use_improved_strategy:
                logger.info("Initializing simple improved trading strategy...")
                self.strategy = SimpleImprovedStrategy(
                    initial_capital=Config.INITIAL_CAPITAL,
                    risk_per_trade=Config.RISK_PER_TRADE * 0.75  # Reduced risk for improved strategy
                )
            else:
                logger.info("Initializing original trading strategy...")
                self.strategy = RSIMACrossoverStrategy(
                    initial_capital=Config.INITIAL_CAPITAL,
                    risk_per_trade=Config.RISK_PER_TRADE
                )
            
            # ML model
            logger.info("Initializing ML model...")
            self.ml_model = StockPredictiveModel(model_type='random_forest')
            
            # Google Sheets logger
            logger.info("Initializing Google Sheets logger...")
            try:
                self.sheets_logger = GoogleSheetsLogger()
                if not self.sheets_logger.is_connected():
                    raise Exception("Failed to connect to Google Sheets")
            except Exception as e:
                logger.warning(f"Google Sheets initialization failed: {e}")
                logger.info("Using mock Google Sheets logger for demonstration...")
                try:
                    from ..utils.mock_sheets_logger import MockGoogleSheetsLogger
                    self.sheets_logger = MockGoogleSheetsLogger()
                    logger.success("Mock Google Sheets logger initialized successfully")
                except Exception as mock_error:
                    logger.error(f"Failed to initialize mock logger: {mock_error}")
                    self.sheets_logger = None
            
            # Telegram bot
            logger.info("Initializing Telegram bot...")
            try:
                self.telegram_bot = TelegramAlertsBot()
            except Exception as e:
                logger.warning(f"Telegram bot initialization failed: {e}")
                self.telegram_bot = None
            
            logger.success("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate system configuration and component status."""
        logger.info("Validating system configuration...")
        
        validation_results = {
            'overall_status': 'OK',
            'errors': [],
            'warnings': [],
            'component_status': {}
        }
        
        # Validate configuration
        config_validation = Config.validate_config()
        if not config_validation['valid']:
            validation_results['errors'].extend(config_validation['missing_configs'])
            validation_results['overall_status'] = 'ERROR'
        
        validation_results['warnings'].extend(config_validation.get('warnings', []))
        
        # Check data fetcher
        if self.data_fetcher:
            validation_results['component_status']['data_fetcher'] = 'OK'
        else:
            validation_results['errors'].append('Data fetcher not initialized')
            validation_results['overall_status'] = 'ERROR'
        
        # Check strategy
        validation_results['component_status']['strategy'] = 'OK' if self.strategy else 'ERROR'
        
        # Check ML model
        validation_results['component_status']['ml_model'] = 'OK' if self.ml_model else 'ERROR'
        
        # Check Google Sheets
        if self.sheets_logger and self.sheets_logger.is_connected():
            validation_results['component_status']['google_sheets'] = 'OK'
        else:
            validation_results['component_status']['google_sheets'] = 'DISABLED'
            validation_results['warnings'].append('Google Sheets integration not available')
        
        # Check Telegram
        if self.telegram_bot and self.telegram_bot.is_enabled:
            validation_results['component_status']['telegram'] = 'OK'
        else:
            validation_results['component_status']['telegram'] = 'DISABLED'
            validation_results['warnings'].append('Telegram alerts not available')
        
        logger.info(f"Configuration validation completed: {validation_results['overall_status']}")
        return validation_results
    
    def train_ml_model(self, retrain: bool = False) -> bool:
        """Train or load the ML model."""
        if not self.ml_model:
            logger.error("ML model not initialized")
            return False
        
        model_path = 'models/trading_model.pkl'
        
        # Check if model exists and doesn't need retraining
        if os.path.exists(model_path) and not retrain:
            try:
                self.ml_model.load_model(model_path)
                logger.success("ML model loaded from disk")
                return True
            except Exception as e:
                logger.warning(f"Failed to load existing model: {e}")
        
        # Train new model
        try:
            logger.info("Training ML model...")
            
            # Fetch training data
            training_data = self.data_fetcher.get_nifty50_data()
            
            if not training_data:
                logger.error("No training data available")
                return False
            
            # Train model
            training_results = self.ml_model.train(training_data)
            
            # Save model
            os.makedirs('models', exist_ok=True)
            self.ml_model.save_model(model_path)
            
            logger.success(f"ML model trained successfully. Accuracy: {training_results['test_accuracy']:.4f}")
            
            # Send Telegram notification
            if self.telegram_bot:
                summary = self.ml_model.get_model_summary()
                self.telegram_bot.send_message_sync(
                    f"🤖 <b>ML Model Training Complete</b>\n\n"
                    f"<b>Accuracy:</b> {summary['test_accuracy']:.2%}\n"
                    f"<b>Features:</b> {summary['num_features']}\n"
                    f"<b>Model Type:</b> {summary['model_type']}"
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to train ML model: {e}")
            
            # Send error alert
            if self.telegram_bot:
                self.telegram_bot.send_error_alert(
                    f"ML model training failed: {str(e)}",
                    error_type='ERROR'
                )
            
            return False
    
    def scan_and_analyze(self) -> Dict[str, Any]:
        """Perform market scan and analysis."""
        logger.info("Starting market scan and analysis...")
        
        scan_results = {
            'timestamp': datetime.now(),
            'symbols_scanned': 0,
            'signals_generated': 0,
            'trades_executed': 0,
            'errors': 0,
            'signals': {},
            'ml_predictions': {}
        }
        
        try:
            # Fetch current market data
            logger.info("Fetching market data...")
            market_data = self.data_fetcher.get_nifty50_data()
            
            if not market_data:
                logger.warning("No market data available")
                return scan_results
            
            scan_results['symbols_scanned'] = len(market_data)
            
            # Process each symbol
            for symbol, data in market_data.items():
                try:
                    logger.debug(f"Analyzing {symbol}...")
                    
                    # Calculate technical indicators
                    data_with_indicators = TechnicalIndicators.calculate_all_indicators(data)
                    
                    # Get current signals
                    current_signals = self.strategy.get_current_signals({symbol: data_with_indicators})
                    
                    if symbol in current_signals:
                        signal_data = current_signals[symbol]
                        scan_results['signals'][symbol] = signal_data
                        
                        # Get ML prediction if model is available
                        ml_prediction = None
                        if self.ml_model and self.ml_model.is_trained:
                            try:
                                ml_prediction = self.ml_model.predict(data_with_indicators)
                                scan_results['ml_predictions'][symbol] = ml_prediction
                            except Exception as e:
                                logger.warning(f"ML prediction failed for {symbol}: {e}")
                        
                        # Check if signal is actionable
                        if signal_data['signal'] in ['BUY', 'SELL']:
                            scan_results['signals_generated'] += 1
                            
                            # Log signal
                            if self.sheets_logger:
                                enhanced_signal = signal_data.copy()
                                enhanced_signal['symbol'] = symbol
                                self.sheets_logger.log_signal(enhanced_signal, ml_prediction)
                            
                            # Send Telegram alert for strong signals
                            if (self.telegram_bot and 
                                signal_data['strength'] in ['STRONG', 'MODERATE']):
                                self.telegram_bot.send_signal_alert(
                                    {**signal_data, 'symbol': symbol},
                                    ml_prediction
                                )
                            
                            logger.info(f"Generated {signal_data['signal']} signal for {symbol} "
                                      f"(Strength: {signal_data['strength']})")
                    
                    self.api_call_count += 1
                    
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
                    scan_results['errors'] += 1
                    self.error_count += 1
            
            # Update daily stats
            self.daily_stats['signals_generated'] += scan_results['signals_generated']
            self.scan_count += 1
            self.last_scan_time = datetime.now()
            
            logger.success(f"Market scan completed. Scanned {scan_results['symbols_scanned']} symbols, "
                         f"generated {scan_results['signals_generated']} signals")
            
        except Exception as e:
            logger.error(f"Market scan failed: {e}")
            scan_results['errors'] += 1
            self.error_count += 1
            
            # Send error alert
            if self.telegram_bot:
                self.telegram_bot.send_error_alert(
                    f"Market scan failed: {str(e)}",
                    error_type='ERROR'
                )
        
        return scan_results
    
    def execute_backtest(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Execute strategy backtesting."""
        logger.info("Starting strategy backtesting...")
        
        try:
            # Calculate date range if not provided
            if not start_date:
                start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
            
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            # Fetch historical data
            logger.info(f"Fetching historical data from {start_date} to {end_date}...")
            historical_data = self.data_fetcher.get_nifty50_data()
            
            if not historical_data:
                logger.error("No historical data available for backtesting")
                return {}
            
            # Run backtest
            logger.info("Running backtest...")
            backtest_results = self.strategy.backtest(
                historical_data, 
                start_date=start_date, 
                end_date=end_date,
                sheets_logger=self.sheets_logger
            )
            
            # Log results to Google Sheets
            if self.sheets_logger and backtest_results:
                performance_metrics = {
                    'total_return': backtest_results['total_return'],
                    'annualized_return': backtest_results['annualized_return'],
                    'sharpe_ratio': backtest_results['sharpe_ratio'],
                    'max_drawdown': backtest_results['max_drawdown'],
                    'win_rate': backtest_results['win_rate'],
                    'total_trades': backtest_results['total_trades']
                }
                self.sheets_logger.update_performance_metrics(performance_metrics)
            
            # Send Telegram summary
            if self.telegram_bot and backtest_results:
                summary_message = f"""
📊 <b>BACKTEST RESULTS</b>

<b>Period:</b> {start_date} to {end_date}
<b>Total Return:</b> {backtest_results['total_return']:.2%}
<b>Annualized Return:</b> {backtest_results['annualized_return']:.2%}
<b>Sharpe Ratio:</b> {backtest_results['sharpe_ratio']:.2f}
<b>Max Drawdown:</b> {backtest_results['max_drawdown']:.2%}
<b>Win Rate:</b> {backtest_results['win_rate']:.2%}
<b>Total Trades:</b> {backtest_results['total_trades']}
"""
                self.telegram_bot.send_message_sync(summary_message)
            
            logger.success(f"Backtesting completed. Total return: {backtest_results.get('total_return', 0):.2%}")
            
            return backtest_results
            
        except Exception as e:
            logger.error(f"Backtesting failed: {e}")
            
            # Send error alert
            if self.telegram_bot:
                self.telegram_bot.send_error_alert(
                    f"Backtesting failed: {str(e)}",
                    error_type='ERROR'
                )
            
            return {}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        uptime = datetime.now() - self.start_time
        
        # Get system resources
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        status = {
            'engine_status': {
                'running': self.is_running,
                'uptime_seconds': uptime.total_seconds(),
                'uptime_string': str(uptime).split('.')[0],  # Remove microseconds
                'last_scan': self.last_scan_time.isoformat() if self.last_scan_time else None,
                'scan_count': self.scan_count,
                'error_count': self.error_count,
                'api_calls': self.api_call_count
            },
            'components': {
                'data_fetcher': self.data_fetcher is not None,
                'strategy': self.strategy is not None,
                'ml_model': self.ml_model is not None and self.ml_model.is_trained,
                'google_sheets': self.sheets_logger is not None and self.sheets_logger.is_connected(),
                'telegram': self.telegram_bot is not None and self.telegram_bot.is_enabled
            },
            'daily_stats': self.daily_stats.copy(),
            'system_resources': {
                'cpu_usage': cpu_usage,
                'memory_usage': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_usage': disk.percent,
                'disk_free_gb': disk.free / (1024**3)
            },
            'configuration': {
                'initial_capital': Config.INITIAL_CAPITAL,
                'risk_per_trade': Config.RISK_PER_TRADE,
                'max_positions': Config.MAX_POSITIONS,
                'stocks_monitored': len(Config.NIFTY_50_STOCKS)
            }
        }
        
        return status
    
    def send_daily_summary(self):
        """Send daily trading summary."""
        try:
            status = self.get_system_status()
            
            # Get trade summary from sheets if available
            trade_summary = {}
            if self.sheets_logger:
                trade_summary = self.sheets_logger.get_trade_summary()
            
            summary_data = {
                'trades_today': self.daily_stats['trades_today'],
                'signals_generated': self.daily_stats['signals_generated'],
                'pnl_today': self.daily_stats['pnl_today'],
                'total_value': status['configuration']['initial_capital'],
                'active_positions': 0,
                'cash_available': status['configuration']['initial_capital'],
                'win_rate': trade_summary.get('win_rate', 0),
                'total_trades': trade_summary.get('total_trades', 0),
                'total_pnl': trade_summary.get('total_pnl', 0),
                'api_calls': self.api_call_count,
                'errors': self.error_count,
                'uptime': f"{status['engine_status']['uptime_string']}"
            }
            
            # Send Telegram summary
            if self.telegram_bot:
                self.telegram_bot.send_daily_summary(summary_data)
            
            # Update Google Sheets dashboard
            if self.sheets_logger:
                self.sheets_logger.create_dashboard_charts()
            
            logger.info("Daily summary sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to send daily summary: {e}")
    
    def start_automated_trading(self):
        """Start the automated trading engine."""
        logger.info("Starting automated trading engine...")
        
        # Validate configuration
        validation = self.validate_configuration()
        if validation['overall_status'] == 'ERROR':
            logger.error("Cannot start trading engine due to configuration errors")
            return False
        
        # Train ML model if needed
        if not self.ml_model.is_trained:
            logger.info("Training ML model...")
            self.train_ml_model()
        
        # Schedule tasks
        self._schedule_tasks()
        
        self.is_running = True
        
        # Send startup notification
        if self.telegram_bot:
            self.telegram_bot.send_message_sync(
                "🚀 <b>TRADING ENGINE STARTED</b>\n\n"
                f"<b>Status:</b> {validation['overall_status']}\n"
                f"<b>Warnings:</b> {len(validation['warnings'])}\n"
                f"<b>Stocks Monitored:</b> {len(Config.NIFTY_50_STOCKS)}\n"
                f"<b>Capital:</b> ₹{Config.INITIAL_CAPITAL:,.2f}\n\n"
                "Automated trading is now active! 📈"
            )
        
        logger.success("Automated trading engine started successfully")
        
        try:
            # Main trading loop
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
            self.stop_automated_trading()
        except Exception as e:
            logger.error(f"Trading engine error: {e}")
            
            if self.telegram_bot:
                self.telegram_bot.send_error_alert(
                    f"Trading engine crashed: {str(e)}",
                    error_type='CRITICAL',
                    include_traceback=True
                )
            
            raise
    
    def _schedule_tasks(self):
        """Schedule automated tasks."""
        # Market scan during trading hours (9:15 AM to 3:30 PM IST)
        schedule.every().monday.at("09:15").do(self.scan_and_analyze)
        schedule.every().monday.at("11:00").do(self.scan_and_analyze)
        schedule.every().monday.at("13:00").do(self.scan_and_analyze)
        schedule.every().monday.at("15:00").do(self.scan_and_analyze)
        
        schedule.every().tuesday.at("09:15").do(self.scan_and_analyze)
        schedule.every().tuesday.at("11:00").do(self.scan_and_analyze)
        schedule.every().tuesday.at("13:00").do(self.scan_and_analyze)
        schedule.every().tuesday.at("15:00").do(self.scan_and_analyze)
        
        schedule.every().wednesday.at("09:15").do(self.scan_and_analyze)
        schedule.every().wednesday.at("11:00").do(self.scan_and_analyze)
        schedule.every().wednesday.at("13:00").do(self.scan_and_analyze)
        schedule.every().wednesday.at("15:00").do(self.scan_and_analyze)
        
        schedule.every().thursday.at("09:15").do(self.scan_and_analyze)
        schedule.every().thursday.at("11:00").do(self.scan_and_analyze)
        schedule.every().thursday.at("13:00").do(self.scan_and_analyze)
        schedule.every().thursday.at("15:00").do(self.scan_and_analyze)
        
        schedule.every().friday.at("09:15").do(self.scan_and_analyze)
        schedule.every().friday.at("11:00").do(self.scan_and_analyze)
        schedule.every().friday.at("13:00").do(self.scan_and_analyze)
        schedule.every().friday.at("15:00").do(self.scan_and_analyze)
        
        # Daily summary at market close
        schedule.every().monday.at("16:00").do(self.send_daily_summary)
        schedule.every().tuesday.at("16:00").do(self.send_daily_summary)
        schedule.every().wednesday.at("16:00").do(self.send_daily_summary)
        schedule.every().thursday.at("16:00").do(self.send_daily_summary)
        schedule.every().friday.at("16:00").do(self.send_daily_summary)
        
        # Weekly backtest
        schedule.every().saturday.at("10:00").do(self.execute_backtest)
        
        # System status check
        schedule.every().hour.do(self._periodic_health_check)
        
        logger.info("Automated tasks scheduled successfully")
    
    def _periodic_health_check(self):
        """Perform periodic health check."""
        try:
            status = self.get_system_status()
            
            # Check for high error rate
            if self.error_count > 10:
                logger.warning(f"High error count detected: {self.error_count}")
                
                if self.telegram_bot:
                    self.telegram_bot.send_error_alert(
                        f"High error count detected: {self.error_count} errors",
                        error_type='WARNING'
                    )
            
            # Check system resources
            if status['system_resources']['cpu_usage'] > 90:
                logger.warning(f"High CPU usage: {status['system_resources']['cpu_usage']:.1f}%")
            
            if status['system_resources']['memory_usage'] > 90:
                logger.warning(f"High memory usage: {status['system_resources']['memory_usage']:.1f}%")
            
            if status['system_resources']['disk_usage'] > 90:
                logger.warning(f"Low disk space: {status['system_resources']['disk_usage']:.1f}% used")
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    def stop_automated_trading(self):
        """Stop the automated trading engine."""
        logger.info("Stopping automated trading engine...")
        
        self.is_running = False
        
        # Clear scheduled tasks
        schedule.clear()
        
        # Send shutdown notification
        if self.telegram_bot:
            uptime = datetime.now() - self.start_time
            self.telegram_bot.send_message_sync(
                "🛑 <b>TRADING ENGINE STOPPED</b>\n\n"
                f"<b>Uptime:</b> {str(uptime).split('.')[0]}\n"
                f"<b>Scans Completed:</b> {self.scan_count}\n"
                f"<b>Errors:</b> {self.error_count}\n"
                f"<b>API Calls:</b> {self.api_call_count}\n\n"
                "Trading engine has been shut down. 📴"
            )
        
        logger.success("Automated trading engine stopped successfully")
    
    def run_single_scan(self) -> Dict[str, Any]:
        """Run a single market scan (for testing/manual execution)."""
        logger.info("Running single market scan...")
        return self.scan_and_analyze()
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        try:
            # Get backtest results
            backtest_results = self.execute_backtest()
            
            # Get system status
            system_status = self.get_system_status()
            
            # Get trade summary
            trade_summary = {}
            if self.sheets_logger:
                trade_summary = self.sheets_logger.get_trade_summary()
            
            # Get ML model summary
            ml_summary = {}
            if self.ml_model and self.ml_model.is_trained:
                ml_summary = self.ml_model.get_model_summary()
            
            report = {
                'generated_at': datetime.now().isoformat(),
                'backtest_results': backtest_results,
                'system_status': system_status,
                'trade_summary': trade_summary,
                'ml_model_summary': ml_summary,
                'configuration': Config.get_all_config()
            }
            
            logger.success("Performance report generated successfully")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            return {}


# Convenience function to run the trading engine
def run_trading_engine():
    """Run the trading engine."""
    engine = TradingEngine()
    
    try:
        engine.start_automated_trading()
    except KeyboardInterrupt:
        logger.info("Shutting down trading engine...")
        engine.stop_automated_trading()
    except Exception as e:
        logger.error(f"Trading engine failed: {e}")
        engine.stop_automated_trading()
        raise


if __name__ == "__main__":
    run_trading_engine()