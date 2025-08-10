"""
Mock Google Sheets logger for testing when real credentials aren't available.
This allows the system to function and demonstrate trade logging without requiring Google Sheets setup.
"""

from typing import Dict, Any
from datetime import datetime
from loguru import logger
import json
import os
from pathlib import Path


class MockGoogleSheetsLogger:
    """Mock implementation of Google Sheets logger that saves data locally."""
    
    def __init__(self, mock_data_dir: str = "mock_sheets_data"):
        """Initialize mock logger with local file storage."""
        self.mock_data_dir = Path(mock_data_dir)
        self.mock_data_dir.mkdir(exist_ok=True)
        
        # Initialize mock data files
        self.trade_log_file = self.mock_data_dir / "trade_log.json"
        self.portfolio_summary_file = self.mock_data_dir / "portfolio_summary.json"
        self.performance_metrics_file = self.mock_data_dir / "performance_metrics.json"
        self.signal_log_file = self.mock_data_dir / "signal_log.json"
        
        # Initialize empty files if they don't exist
        for file_path in [self.trade_log_file, self.portfolio_summary_file, 
                         self.performance_metrics_file, self.signal_log_file]:
            if not file_path.exists():
                with open(file_path, 'w') as f:
                    json.dump([], f)
        
        logger.info(f"Mock Google Sheets logger initialized. Data will be saved to: {self.mock_data_dir}")
    
    def is_connected(self) -> bool:
        """Mock connection status - always returns True."""
        return True
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Mock connection status."""
        return {
            'connected': True,
            'type': 'mock',
            'data_directory': str(self.mock_data_dir),
            'last_update': datetime.now().isoformat()
        }
    
    def log_trade(self, trade_data: Dict[str, Any]) -> bool:
        """Log trade data to local JSON file."""
        try:
            # Convert datetime objects to strings for JSON serialization
            processed_data = self._serialize_data(trade_data)
            processed_data['logged_at'] = datetime.now().isoformat()
            
            # Load existing data
            with open(self.trade_log_file, 'r') as f:
                trades = json.load(f)
            
            # Add new trade
            trades.append(processed_data)
            
            # Save updated data
            with open(self.trade_log_file, 'w') as f:
                json.dump(trades, f, indent=2)
            
            logger.info(f"Mock: Logged trade {trade_data.get('action', 'UNKNOWN')} for {trade_data.get('symbol', 'UNKNOWN')}")
            return True
            
        except Exception as e:
            logger.error(f"Mock: Failed to log trade: {e}")
            return False
    
    def log_signal(self, signal_data: Dict[str, Any], ml_prediction: Dict[str, Any] = None) -> bool:
        """Log signal data to local JSON file."""
        try:
            # Combine signal and ML prediction data
            processed_data = self._serialize_data(signal_data)
            if ml_prediction:
                processed_data['ml_prediction'] = self._serialize_data(ml_prediction)
            processed_data['logged_at'] = datetime.now().isoformat()
            
            # Load existing data
            with open(self.signal_log_file, 'r') as f:
                signals = json.load(f)
            
            # Add new signal
            signals.append(processed_data)
            
            # Save updated data
            with open(self.signal_log_file, 'w') as f:
                json.dump(signals, f, indent=2)
            
            logger.info(f"Mock: Logged signal for {signal_data.get('symbol', 'UNKNOWN')}")
            return True
            
        except Exception as e:
            logger.error(f"Mock: Failed to log signal: {e}")
            return False
    
    def update_portfolio_summary(self, summary_data: Dict[str, Any]) -> bool:
        """Update portfolio summary in local JSON file."""
        try:
            processed_data = self._serialize_data(summary_data)
            processed_data['updated_at'] = datetime.now().isoformat()
            
            # Load existing data
            with open(self.portfolio_summary_file, 'r') as f:
                summaries = json.load(f)
            
            # Add new summary
            summaries.append(processed_data)
            
            # Save updated data
            with open(self.portfolio_summary_file, 'w') as f:
                json.dump(summaries, f, indent=2)
            
            logger.info(f"Mock: Updated portfolio summary - Total Capital: ${summary_data.get('total_capital', 0):,.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Mock: Failed to update portfolio summary: {e}")
            return False
    
    def update_performance_metrics(self, metrics_data: Dict[str, Any]) -> bool:
        """Update performance metrics in local JSON file."""
        try:
            processed_data = self._serialize_data(metrics_data)
            processed_data['updated_at'] = datetime.now().isoformat()
            
            # Save metrics data
            with open(self.performance_metrics_file, 'w') as f:
                json.dump(processed_data, f, indent=2)
            
            logger.info(f"Mock: Updated performance metrics - Total Return: {metrics_data.get('total_return', 0):.2%}")
            return True
            
        except Exception as e:
            logger.error(f"Mock: Failed to update performance metrics: {e}")
            return False
    
    def get_trade_summary(self) -> Dict[str, Any]:
        """Get trade summary from local data."""
        try:
            with open(self.trade_log_file, 'r') as f:
                trades = json.load(f)
            
            if not trades:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'total_pnl': 0,
                    'win_rate': 0
                }
            
            total_trades = len(trades)
            winning_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
            losing_trades = sum(1 for trade in trades if trade.get('pnl', 0) < 0)
            total_pnl = sum(trade.get('pnl', 0) for trade in trades)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'total_pnl': total_pnl,
                'win_rate': win_rate
            }
            
        except Exception as e:
            logger.error(f"Mock: Failed to get trade summary: {e}")
            return {}
    
    def create_dashboard_charts(self) -> bool:
        """Mock dashboard chart creation."""
        logger.info("Mock: Dashboard charts created (simulated)")
        return True
    
    def backup_data(self) -> bool:
        """Mock data backup."""
        logger.info("Mock: Data backup completed (simulated)")
        return True
    
    def _serialize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert data types for JSON serialization."""
        serialized = {}
        for key, value in data.items():
            if isinstance(value, datetime):
                serialized[key] = value.isoformat()
            elif hasattr(value, 'item'):  # numpy types
                serialized[key] = value.item()
            else:
                serialized[key] = value
        return serialized
    
    def print_summary(self):
        """Print a summary of all logged data."""
        print("\n" + "=" * 60)
        print("MOCK GOOGLE SHEETS DATA SUMMARY")
        print("=" * 60)
        
        try:
            # Trade log summary
            with open(self.trade_log_file, 'r') as f:
                trades = json.load(f)
            print(f"📊 Trade Log: {len(trades)} trades recorded")
            
            if trades:
                buy_trades = sum(1 for t in trades if t.get('action') == 'BUY')
                sell_trades = sum(1 for t in trades if t.get('action') == 'SELL')
                print(f"   - Buy trades: {buy_trades}")
                print(f"   - Sell trades: {sell_trades}")
            
            # Portfolio summary
            with open(self.portfolio_summary_file, 'r') as f:
                summaries = json.load(f)
            print(f"💰 Portfolio Summaries: {len(summaries)} entries")
            
            # Performance metrics
            if self.performance_metrics_file.exists():
                with open(self.performance_metrics_file, 'r') as f:
                    metrics = json.load(f)
                print(f"📈 Performance Metrics: Updated")
                if metrics:
                    print(f"   - Total Return: {metrics.get('total_return', 0):.2%}")
                    print(f"   - Total Trades: {metrics.get('total_trades', 0)}")
                    print(f"   - Win Rate: {metrics.get('win_rate', 0):.1%}")
            
            # Signal log
            with open(self.signal_log_file, 'r') as f:
                signals = json.load(f)
            print(f"🎯 Signal Log: {len(signals)} signals recorded")
            
            print(f"\n📁 All data saved to: {self.mock_data_dir}")
            print("=" * 60)
            
        except Exception as e:
            logger.error(f"Failed to print summary: {e}")