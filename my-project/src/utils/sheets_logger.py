"""
Google Sheets automation module for logging trades and tracking P&L.
Implements automatic trade logging, summary P&L, and win ratio tracking in separate tabs.
"""

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from loguru import logger
import time
import os

try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    logger.warning("gspread not installed. Google Sheets features will be disabled.")
    GSPREAD_AVAILABLE = False

from ..config import Config


class GoogleSheetsLogger:
    """Handles automatic logging of trades and analytics to Google Sheets."""
    
    def __init__(self, credentials_file: str = None, spreadsheet_id: str = None):
        """
        Initialize Google Sheets logger.
        
        Args:
            credentials_file: Path to Google service account credentials JSON file
            spreadsheet_id: Google Sheets spreadsheet ID
        """
        self.credentials_file = credentials_file or Config.GOOGLE_SHEETS_CREDENTIALS_FILE
        self.spreadsheet_id = spreadsheet_id or Config.GOOGLE_SHEETS_SPREADSHEET_ID
        
        self.client = None
        self.spreadsheet = None
        self.worksheets = {}
        
        if not GSPREAD_AVAILABLE:
            logger.warning("Google Sheets disabled - gspread not available")
            return
        
        # Initialize connection
        self._initialize_connection()
        
        # Define worksheet structures
        self.worksheet_configs = {
            'Trade_Log': {
                'headers': [
                    'Timestamp', 'Symbol', 'Action', 'Quantity', 'Entry_Price', 
                    'Exit_Price', 'Entry_Date', 'Exit_Date', 'Exit_Reason',
                    'PnL', 'PnL_Percent', 'Stop_Loss', 'Take_Profit', 
                    'RSI', 'SMA_20', 'SMA_50', 'Signal_Strength', 'Trade_Duration'
                ]
            },
            'Portfolio_Summary': {
                'headers': [
                    'Date', 'Total_Capital', 'Cash', 'Positions_Value', 
                    'Total_PnL', 'Daily_Return', 'Cumulative_Return',
                    'Active_Positions', 'Max_Drawdown', 'Win_Rate'
                ]
            },
            'Signal_Log': {
                'headers': [
                    'Timestamp', 'Symbol', 'Signal_Type', 'Signal_Strength',
                    'Price', 'RSI', 'SMA_20', 'SMA_50', 'MACD', 'Volume_Ratio',
                    'ML_Prediction', 'ML_Confidence', 'Action_Taken'
                ]
            },
            'Performance_Metrics': {
                'headers': [
                    'Metric', 'Value', 'Period', 'Last_Updated'
                ]
            }
        }
        
        # Initialize worksheets only if connected
        if self.is_connected():
            self._setup_worksheets()
            logger.info("Google Sheets logger initialized successfully")
    
    def _initialize_connection(self):
        """Initialize connection to Google Sheets using updated authentication."""
        try:
            if not self.credentials_file or not self.spreadsheet_id:
                logger.warning("Google Sheets credentials or spreadsheet ID not configured")
                return
            
            if not os.path.exists(self.credentials_file):
                logger.warning(f"Credentials file not found: {self.credentials_file}")
                return
            
            # Define the scope - updated scopes
            scopes = [
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive'
            ]
            
            # Load credentials using the new google-auth library
            credentials = Credentials.from_service_account_file(
                self.credentials_file, 
                scopes=scopes
            )
            
            # Initialize client
            self.client = gspread.authorize(credentials)
            
            # Test connection by opening spreadsheet
            self.spreadsheet = self.client.open_by_key(self.spreadsheet_id)
            
            logger.success("Connected to Google Sheets successfully")
            
        except FileNotFoundError:
            logger.error(f"Credentials file not found: {self.credentials_file}")
            self.client = None
            self.spreadsheet = None
        except Exception as e:
            logger.error(f"Failed to connect to Google Sheets: {e}")
            self.client = None
            self.spreadsheet = None
    
    def _setup_worksheets(self):
        """Setup required worksheets with headers."""
        if not self.spreadsheet:
            return
        
        try:
            for sheet_name, config in self.worksheet_configs.items():
                try:
                    # Try to get existing worksheet
                    worksheet = self.spreadsheet.worksheet(sheet_name)
                    self.worksheets[sheet_name] = worksheet
                    
                    # Check if headers exist, if not add them
                    if not worksheet.row_values(1):
                        worksheet.append_row(config['headers'])
                        logger.info(f"Added headers to existing worksheet: {sheet_name}")
                    
                except gspread.WorksheetNotFound:
                    # Create new worksheet
                    worksheet = self.spreadsheet.add_worksheet(
                        title=sheet_name, 
                        rows=1000, 
                        cols=len(config['headers'])
                    )
                    worksheet.append_row(config['headers'])
                    self.worksheets[sheet_name] = worksheet
                    logger.success(f"Created new worksheet: {sheet_name}")
                
                # Apply formatting to headers
                self._format_headers(worksheet, len(config['headers']))
                
                time.sleep(1)  # Rate limiting
            
        except Exception as e:
            logger.error(f"Error setting up worksheets: {e}")
    
    def _format_headers(self, worksheet, num_cols: int):
        """Apply formatting to worksheet headers."""
        try:
            # Format header row
            worksheet.format('1:1', {
                'backgroundColor': {'red': 0.8, 'green': 0.8, 'blue': 0.8},
                'textFormat': {'bold': True},
                'horizontalAlignment': 'CENTER'
            })
            
            # Auto-resize columns
            worksheet.format(f'A1:{chr(64 + num_cols)}1', {
                'wrapStrategy': 'WRAP'
            })
            
        except Exception as e:
            logger.warning(f"Could not apply formatting to headers: {e}")
    
    def log_trade(self, trade_data: Dict[str, Any], signal_data: Dict[str, Any] = None):
        """
        Log a completed trade to the Trade_Log worksheet.
        
        Args:
            trade_data: Dictionary with trade information
            signal_data: Dictionary with signal information (optional)
        """
        if not self.spreadsheet or 'Trade_Log' not in self.worksheets:
            logger.warning("Cannot log trade - Google Sheets not configured")
            return
        
        try:
            worksheet = self.worksheets['Trade_Log']
            
            # Prepare trade row data with proper timestamp handling
            trade_duration = None
            if trade_data.get('exit_timestamp') and trade_data.get('entry_timestamp'):
                try:
                    if isinstance(trade_data['entry_timestamp'], datetime):
                        entry_time = trade_data['entry_timestamp']
                    else:
                        entry_time = datetime.fromisoformat(str(trade_data['entry_timestamp']))
                    
                    if isinstance(trade_data['exit_timestamp'], datetime):
                        exit_time = trade_data['exit_timestamp']
                    else:
                        exit_time = datetime.fromisoformat(str(trade_data['exit_timestamp']))
                    
                    trade_duration = (exit_time - entry_time).days
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not calculate trade duration: {e}")
                    trade_duration = 0
            
            # Calculate PnL percentage
            pnl_percent = None
            if trade_data.get('pnl') and trade_data.get('entry_price') and trade_data.get('quantity'):
                cost_basis = trade_data['entry_price'] * trade_data['quantity']
                pnl_percent = (trade_data['pnl'] / cost_basis) * 100 if cost_basis > 0 else 0
            
            # Format timestamps safely
            def format_timestamp(ts, fmt='%Y-%m-%d'):
                if not ts:
                    return ''
                try:
                    if isinstance(ts, datetime):
                        return ts.strftime(fmt)
                    elif isinstance(ts, str):
                        return datetime.fromisoformat(ts).strftime(fmt)
                    else:
                        return str(ts)
                except (ValueError, TypeError):
                    return str(ts) if ts else ''
            
            row_data = [
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                trade_data.get('symbol', ''),
                trade_data.get('action', ''),
                trade_data.get('quantity', 0),
                trade_data.get('entry_price', 0),
                trade_data.get('exit_price', 0),
                format_timestamp(trade_data.get('entry_timestamp')),
                format_timestamp(trade_data.get('exit_timestamp')),
                trade_data.get('exit_reason', ''),
                trade_data.get('pnl', 0),
                pnl_percent or 0,
                trade_data.get('stop_loss', 0),
                trade_data.get('take_profit', 0),
                signal_data.get('rsi', 0) if signal_data else 0,
                signal_data.get('sma_20', 0) if signal_data else 0,
                signal_data.get('sma_50', 0) if signal_data else 0,
                signal_data.get('strength', 'NONE') if signal_data else 'NONE',
                trade_duration or 0
            ]
            
            worksheet.append_row(row_data)
            logger.success(f"Logged trade for {trade_data.get('symbol')} to Google Sheets")
            
        except Exception as e:
            logger.error(f"Failed to log trade to Google Sheets: {e}")
    
    def log_signal(self, signal_data: Dict[str, Any], ml_prediction: Dict[str, Any] = None):
        """
        Log trading signals to the Signal_Log worksheet.
        
        Args:
            signal_data: Dictionary with signal information
            ml_prediction: Dictionary with ML prediction data (optional)
        """
        if not self.spreadsheet or 'Signal_Log' not in self.worksheets:
            logger.warning("Cannot log signal - Google Sheets not configured")
            return
        
        try:
            worksheet = self.worksheets['Signal_Log']
            
            row_data = [
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                signal_data.get('symbol', ''),
                signal_data.get('signal', 'NONE'),
                signal_data.get('strength', 'NONE'),
                signal_data.get('price', 0),
                signal_data.get('rsi', 0),
                signal_data.get('sma_20', 0),
                signal_data.get('sma_50', 0),
                signal_data.get('macd', 0),
                signal_data.get('volume_ratio', 1),
                ml_prediction.get('prediction', 'N/A') if ml_prediction else 'N/A',
                ml_prediction.get('confidence', 0) if ml_prediction else 0,
                signal_data.get('action_taken', 'NONE')
            ]
            
            worksheet.append_row(row_data)
            logger.debug(f"Logged signal for {signal_data.get('symbol')} to Google Sheets")
            
        except Exception as e:
            logger.error(f"Failed to log signal to Google Sheets: {e}")
    
    def update_portfolio_summary(self, portfolio_data: Dict[str, Any]):
        """
        Update portfolio summary in the Portfolio_Summary worksheet.
        
        Args:
            portfolio_data: Dictionary with portfolio information
        """
        if not self.spreadsheet or 'Portfolio_Summary' not in self.worksheets:
            logger.warning("Cannot update portfolio summary - Google Sheets not configured")
            return
        
        try:
            worksheet = self.worksheets['Portfolio_Summary']
            
            row_data = [
                datetime.now().strftime('%Y-%m-%d'),
                portfolio_data.get('total_capital', 0),
                portfolio_data.get('cash', 0),
                portfolio_data.get('positions_value', 0),
                portfolio_data.get('total_pnl', 0),
                portfolio_data.get('daily_return', 0),
                portfolio_data.get('cumulative_return', 0),
                portfolio_data.get('active_positions', 0),
                portfolio_data.get('max_drawdown', 0),
                portfolio_data.get('win_rate', 0)
            ]
            
            worksheet.append_row(row_data)
            logger.success("Updated portfolio summary in Google Sheets")
            
        except Exception as e:
            logger.error(f"Failed to update portfolio summary: {e}")
    
    def update_performance_metrics(self, metrics: Dict[str, Any]):
        """
        Update performance metrics in the Performance_Metrics worksheet.
        
        Args:
            metrics: Dictionary with performance metrics
        """
        if not self.spreadsheet or 'Performance_Metrics' not in self.worksheets:
            logger.warning("Cannot update performance metrics - Google Sheets not configured")
            return
        
        try:
            worksheet = self.worksheets['Performance_Metrics']
            
            # Clear existing data (keep headers)
            worksheet.clear()
            worksheet.append_row(self.worksheet_configs['Performance_Metrics']['headers'])
            
            # Add metrics
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, dict):
                    # Handle nested metrics
                    for sub_metric, sub_value in metric_value.items():
                        row_data = [
                            f"{metric_name}_{sub_metric}",
                            sub_value,
                            "Overall",
                            timestamp
                        ]
                        worksheet.append_row(row_data)
                else:
                    row_data = [
                        metric_name,
                        metric_value,
                        "Overall",
                        timestamp
                    ]
                    worksheet.append_row(row_data)
            
            logger.success("Updated performance metrics in Google Sheets")
            
        except Exception as e:
            logger.error(f"Failed to update performance metrics: {e}")
    
    def get_trade_summary(self) -> Dict[str, Any]:
        """
        Get trade summary from Google Sheets.
        
        Returns:
            Dictionary with trade summary statistics
        """
        if not self.spreadsheet or 'Trade_Log' not in self.worksheets:
            logger.warning("Cannot get trade summary - Google Sheets not configured")
            return {}
        
        try:
            worksheet = self.worksheets['Trade_Log']
            
            # Get all data
            data = worksheet.get_all_records()
            
            if not data:
                return {'total_trades': 0, 'win_rate': 0, 'total_pnl': 0}
            
            df = pd.DataFrame(data)
            
            # Calculate summary statistics
            total_trades = len(df)
            winning_trades = len(df[df['PnL'] > 0])
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            total_pnl = df['PnL'].sum() if 'PnL' in df.columns else 0
            avg_pnl = df['PnL'].mean() if 'PnL' in df.columns else 0
            
            summary = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': total_trades - winning_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl,
                'best_trade': df['PnL'].max() if 'PnL' in df.columns else 0,
                'worst_trade': df['PnL'].min() if 'PnL' in df.columns else 0
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get trade summary: {e}")
            return {}
    
    def create_dashboard_charts(self):
        """Create dashboard charts in Google Sheets."""
        if not self.spreadsheet:
            logger.warning("Cannot create dashboard - Google Sheets not configured")
            return
        
        try:
            # Create or get dashboard worksheet
            try:
                dashboard = self.spreadsheet.worksheet('Dashboard')
            except gspread.WorksheetNotFound:
                dashboard = self.spreadsheet.add_worksheet(title='Dashboard', rows=30, cols=10)
            
            # Add summary information
            trade_summary = self.get_trade_summary()
            
            # Create summary table
            summary_data = [
                ['Trading Performance Summary', ''],
                ['Total Trades', trade_summary.get('total_trades', 0)],
                ['Winning Trades', trade_summary.get('winning_trades', 0)],
                ['Losing Trades', trade_summary.get('losing_trades', 0)],
                ['Win Rate (%)', f"{trade_summary.get('win_rate', 0):.2f}"],
                ['Total P&L', f"${trade_summary.get('total_pnl', 0):.2f}"],
                ['Average P&L', f"${trade_summary.get('avg_pnl', 0):.2f}"],
                ['Best Trade', f"${trade_summary.get('best_trade', 0):.2f}"],
                ['Worst Trade', f"${trade_summary.get('worst_trade', 0):.2f}"],
                ['Last Updated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
            ]
            
            # Clear and update dashboard
            dashboard.clear()
            
            for i, row in enumerate(summary_data, 1):
                dashboard.update(f'A{i}:B{i}', [row])
            
            # Format summary table
            dashboard.format('A1:B1', {
                'backgroundColor': {'red': 0.2, 'green': 0.6, 'blue': 0.2},
                'textFormat': {'bold': True, 'foregroundColor': {'red': 1, 'green': 1, 'blue': 1}},
                'horizontalAlignment': 'CENTER'
            })
            
            logger.success("Created dashboard in Google Sheets")
            
        except Exception as e:
            logger.error(f"Failed to create dashboard: {e}")
    
    def backup_data(self) -> Dict[str, pd.DataFrame]:
        """
        Backup all data from Google Sheets to local DataFrames.
        
        Returns:
            Dictionary of worksheet_name -> DataFrame
        """
        backup_data = {}
        
        if not self.spreadsheet:
            logger.warning("Cannot backup data - Google Sheets not configured")
            return backup_data
        
        try:
            for sheet_name, worksheet in self.worksheets.items():
                try:
                    data = worksheet.get_all_records()
                    if data:
                        backup_data[sheet_name] = pd.DataFrame(data)
                        logger.info(f"Backed up {len(data)} records from {sheet_name}")
                    else:
                        backup_data[sheet_name] = pd.DataFrame()
                        
                except Exception as e:
                    logger.error(f"Failed to backup {sheet_name}: {e}")
                    backup_data[sheet_name] = pd.DataFrame()
                
                time.sleep(1)  # Rate limiting
            
            logger.success(f"Backed up data from {len(backup_data)} worksheets")
            
        except Exception as e:
            logger.error(f"Failed to backup data: {e}")
        
        return backup_data
    
    def is_connected(self) -> bool:
        """Check if connected to Google Sheets."""
        return self.client is not None and self.spreadsheet is not None
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get connection status and configuration info."""
        return {
            'connected': self.is_connected(),
            'spreadsheet_id': self.spreadsheet_id,
            'worksheets_configured': len(self.worksheets),
            'expected_worksheets': len(self.worksheet_configs)
        }