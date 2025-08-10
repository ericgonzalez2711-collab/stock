"""
Telegram bot integration for sending trading signal alerts and error notifications.
Implements automated alerts for buy/sell signals, trade executions, and system errors.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger
import traceback

try:
    from telegram import Bot
    from telegram.error import TelegramError
    TELEGRAM_AVAILABLE = True
except ImportError:
    logger.warning("python-telegram-bot not installed. Telegram features will be disabled.")
    TELEGRAM_AVAILABLE = False

from ..config import Config


class TelegramAlertsBot:
    """Handles Telegram alerts for trading signals and notifications."""
    
    def __init__(self, bot_token: str = None, chat_id: str = None):
        """
        Initialize Telegram alerts bot.
        
        Args:
            bot_token: Telegram bot token
            chat_id: Telegram chat ID to send messages to
        """
        self.bot_token = bot_token or Config.TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or Config.TELEGRAM_CHAT_ID
        
        self.bot = None
        self.is_enabled = False
        
        if not TELEGRAM_AVAILABLE:
            logger.warning("Telegram bot disabled - python-telegram-bot not available")
            return
        
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram bot disabled - token or chat_id not configured")
            return
        
        try:
            self.bot = Bot(token=self.bot_token)
            self.is_enabled = True
            logger.success("Telegram bot initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}")
            self.is_enabled = False
    
    async def send_message(self, message: str, parse_mode: str = 'HTML') -> bool:
        """
        Send a message via Telegram.
        
        Args:
            message: Message text to send
            parse_mode: Message formatting ('HTML' or 'Markdown')
            
        Returns:
            True if message sent successfully, False otherwise
        """
        if not self.is_enabled:
            logger.debug("Telegram not enabled, skipping message")
            return False
        
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode
            )
            logger.debug("Telegram message sent successfully")
            return True
            
        except TelegramError as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending Telegram message: {e}")
            return False
    
    def send_message_sync(self, message: str, parse_mode: str = 'HTML') -> bool:
        """
        Send a message synchronously (wrapper for async method).
        
        Args:
            message: Message text to send
            parse_mode: Message formatting ('HTML' or 'Markdown')
            
        Returns:
            True if message sent successfully, False otherwise
        """
        try:
            # Create new event loop if none exists
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            return loop.run_until_complete(self.send_message(message, parse_mode))
        except Exception as e:
            logger.error(f"Error in synchronous message send: {e}")
            return False
    
    def send_signal_alert(self, signal_data: Dict[str, Any], ml_prediction: Dict[str, Any] = None) -> bool:
        """
        Send trading signal alert.
        
        Args:
            signal_data: Dictionary with signal information
            ml_prediction: Dictionary with ML prediction data (optional)
            
        Returns:
            True if alert sent successfully, False otherwise
        """
        if not self.is_enabled:
            return False
        
        try:
            symbol = signal_data.get('symbol', 'Unknown')
            signal_type = signal_data.get('signal', 'NONE')
            strength = signal_data.get('strength', 'NONE')
            price = signal_data.get('price', 0)
            rsi = signal_data.get('rsi', 0)
            
            # Create signal emoji
            if signal_type == 'BUY':
                emoji = '🟢'
                action_emoji = '📈'
            elif signal_type == 'SELL':
                emoji = '🔴'
                action_emoji = '📉'
            else:
                emoji = '🟡'
                action_emoji = '➡️'
            
            # Create strength indicator
            strength_indicators = {
                'STRONG': '🔥🔥🔥',
                'MODERATE': '🔥🔥',
                'WEAK': '🔥',
                'NONE': '⚪'
            }
            strength_indicator = strength_indicators.get(strength, '⚪')
            
            # Build message
            message = f"""
{emoji} <b>TRADING SIGNAL ALERT</b> {action_emoji}

<b>Symbol:</b> {symbol}
<b>Signal:</b> {signal_type}
<b>Strength:</b> {strength} {strength_indicator}
<b>Price:</b> ₹{price:.2f}
<b>RSI:</b> {rsi:.2f}

<b>Technical Analysis:</b>
• SMA 20: ₹{signal_data.get('sma_20', 0):.2f}
• SMA 50: ₹{signal_data.get('sma_50', 0):.2f}
• Volume Ratio: {signal_data.get('volume_ratio', 1):.2f}x
"""
            
            # Add ML prediction if available
            if ml_prediction:
                ml_direction = "📈 UP" if ml_prediction.get('prediction') == 1 else "📉 DOWN"
                confidence = ml_prediction.get('confidence', 0) * 100
                
                message += f"""
<b>🤖 ML Prediction:</b> {ml_direction}
<b>Confidence:</b> {confidence:.1f}%
"""
            
            message += f"""
<b>⏰ Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

<i>This is an automated trading signal. Please verify before taking action.</i>
"""
            
            return self.send_message_sync(message)
            
        except Exception as e:
            logger.error(f"Failed to send signal alert: {e}")
            return False
    
    def send_trade_alert(self, trade_data: Dict[str, Any], trade_type: str = 'EXECUTION') -> bool:
        """
        Send trade execution alert.
        
        Args:
            trade_data: Dictionary with trade information
            trade_type: Type of trade alert ('EXECUTION', 'CLOSED', 'STOP_LOSS', 'TAKE_PROFIT')
            
        Returns:
            True if alert sent successfully, False otherwise
        """
        if not self.is_enabled:
            return False
        
        try:
            symbol = trade_data.get('symbol', 'Unknown')
            action = trade_data.get('action', 'Unknown')
            quantity = trade_data.get('quantity', 0)
            price = trade_data.get('entry_price') or trade_data.get('exit_price', 0)
            
            # Create trade type emoji and title
            type_config = {
                'EXECUTION': {'emoji': '⚡', 'title': 'TRADE EXECUTED', 'color': '🟢'},
                'CLOSED': {'emoji': '✅', 'title': 'TRADE CLOSED', 'color': '🔵'},
                'STOP_LOSS': {'emoji': '🛑', 'title': 'STOP LOSS HIT', 'color': '🔴'},
                'TAKE_PROFIT': {'emoji': '🎯', 'title': 'TAKE PROFIT HIT', 'color': '🟢'}
            }
            
            config = type_config.get(trade_type, type_config['EXECUTION'])
            
            message = f"""
{config['color']} <b>{config['title']}</b> {config['emoji']}

<b>Symbol:</b> {symbol}
<b>Action:</b> {action}
<b>Quantity:</b> {quantity} shares
<b>Price:</b> ₹{price:.2f}
<b>Value:</b> ₹{quantity * price:,.2f}
"""
            
            # Add P&L information for closed trades
            if trade_type in ['CLOSED', 'STOP_LOSS', 'TAKE_PROFIT'] and 'pnl' in trade_data:
                pnl = trade_data['pnl']
                pnl_emoji = '💰' if pnl > 0 else '💸' if pnl < 0 else '💫'
                pnl_color = 'green' if pnl > 0 else 'red' if pnl < 0 else 'blue'
                
                entry_price = trade_data.get('entry_price', 0)
                exit_price = trade_data.get('exit_price', 0)
                
                message += f"""
<b>Entry Price:</b> ₹{entry_price:.2f}
<b>Exit Price:</b> ₹{exit_price:.2f}
<b>P&L:</b> ₹{pnl:.2f} {pnl_emoji}
<b>Exit Reason:</b> {trade_data.get('exit_reason', 'Manual')}
"""
            
            # Add stop loss and take profit for new trades
            elif trade_type == 'EXECUTION':
                if 'stop_loss' in trade_data and trade_data['stop_loss']:
                    message += f"<b>Stop Loss:</b> ₹{trade_data['stop_loss']:.2f}\n"
                if 'take_profit' in trade_data and trade_data['take_profit']:
                    message += f"<b>Take Profit:</b> ₹{trade_data['take_profit']:.2f}\n"
            
            message += f"""
<b>⏰ Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            return self.send_message_sync(message)
            
        except Exception as e:
            logger.error(f"Failed to send trade alert: {e}")
            return False
    
    def send_portfolio_update(self, portfolio_data: Dict[str, Any]) -> bool:
        """
        Send portfolio performance update.
        
        Args:
            portfolio_data: Dictionary with portfolio information
            
        Returns:
            True if alert sent successfully, False otherwise
        """
        if not self.is_enabled:
            return False
        
        try:
            total_value = portfolio_data.get('total_capital', 0)
            cash = portfolio_data.get('cash', 0)
            positions_value = portfolio_data.get('positions_value', 0)
            total_pnl = portfolio_data.get('total_pnl', 0)
            daily_return = portfolio_data.get('daily_return', 0) * 100
            
            # Determine overall performance emoji
            if total_pnl > 0:
                performance_emoji = '📈💰'
            elif total_pnl < 0:
                performance_emoji = '📉💸'
            else:
                performance_emoji = '📊💫'
            
            message = f"""
{performance_emoji} <b>PORTFOLIO UPDATE</b>

<b>💼 Total Value:</b> ₹{total_value:,.2f}
<b>💵 Cash:</b> ₹{cash:,.2f}
<b>📊 Positions:</b> ₹{positions_value:,.2f}
<b>💎 Total P&L:</b> ₹{total_pnl:,.2f}
<b>📅 Daily Return:</b> {daily_return:.2f}%

<b>🔢 Active Positions:</b> {portfolio_data.get('active_positions', 0)}
<b>🎯 Win Rate:</b> {portfolio_data.get('win_rate', 0):.1f}%
<b>📉 Max Drawdown:</b> {portfolio_data.get('max_drawdown', 0):.2%}

<b>⏰ Updated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            return self.send_message_sync(message)
            
        except Exception as e:
            logger.error(f"Failed to send portfolio update: {e}")
            return False
    
    def send_error_alert(self, error_message: str, error_type: str = 'ERROR', 
                        include_traceback: bool = False) -> bool:
        """
        Send error/warning alert.
        
        Args:
            error_message: Error message to send
            error_type: Type of error ('ERROR', 'WARNING', 'CRITICAL')
            include_traceback: Whether to include full traceback
            
        Returns:
            True if alert sent successfully, False otherwise
        """
        if not self.is_enabled:
            return False
        
        try:
            # Error type configuration
            error_config = {
                'CRITICAL': {'emoji': '🚨', 'color': '🔴'},
                'ERROR': {'emoji': '❌', 'color': '🟠'},
                'WARNING': {'emoji': '⚠️', 'color': '🟡'},
                'INFO': {'emoji': 'ℹ️', 'color': '🔵'}
            }
            
            config = error_config.get(error_type, error_config['ERROR'])
            
            message = f"""
{config['color']} <b>{error_type} ALERT</b> {config['emoji']}

<b>Message:</b> {error_message}

<b>⏰ Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            # Add traceback if requested and available
            if include_traceback:
                tb = traceback.format_exc()
                if tb and tb != 'NoneType: None\n':
                    # Truncate traceback if too long
                    if len(tb) > 1000:
                        tb = tb[:1000] + '...\n[Truncated]'
                    
                    message += f"""
<b>🔍 Traceback:</b>
<pre>{tb}</pre>
"""
            
            return self.send_message_sync(message)
            
        except Exception as e:
            logger.error(f"Failed to send error alert: {e}")
            return False
    
    def send_daily_summary(self, summary_data: Dict[str, Any]) -> bool:
        """
        Send daily trading summary.
        
        Args:
            summary_data: Dictionary with daily summary information
            
        Returns:
            True if alert sent successfully, False otherwise
        """
        if not self.is_enabled:
            return False
        
        try:
            trades_today = summary_data.get('trades_today', 0)
            signals_generated = summary_data.get('signals_generated', 0)
            pnl_today = summary_data.get('pnl_today', 0)
            
            # Determine day performance
            if pnl_today > 0:
                day_emoji = '🎉📈'
                day_desc = 'PROFITABLE DAY'
            elif pnl_today < 0:
                day_emoji = '😔📉'
                day_desc = 'LOSS DAY'
            else:
                day_emoji = '😐📊'
                day_desc = 'NEUTRAL DAY'
            
            message = f"""
{day_emoji} <b>DAILY SUMMARY - {day_desc}</b>

<b>📅 Date:</b> {datetime.now().strftime('%Y-%m-%d')}

<b>📊 Today's Activity:</b>
• Trades Executed: {trades_today}
• Signals Generated: {signals_generated}
• Daily P&L: ₹{pnl_today:.2f}

<b>💼 Portfolio Status:</b>
• Total Value: ₹{summary_data.get('total_value', 0):,.2f}
• Active Positions: {summary_data.get('active_positions', 0)}
• Cash Available: ₹{summary_data.get('cash_available', 0):,.2f}

<b>📈 Performance Metrics:</b>
• Win Rate: {summary_data.get('win_rate', 0):.1f}%
• Total Trades: {summary_data.get('total_trades', 0)}
• Total P&L: ₹{summary_data.get('total_pnl', 0):,.2f}

<b>🤖 System Status:</b>
• API Calls: {summary_data.get('api_calls', 0)}
• Errors: {summary_data.get('errors', 0)}
• Uptime: {summary_data.get('uptime', '100')}%

<i>Keep trading smart! 🚀</i>
"""
            
            return self.send_message_sync(message)
            
        except Exception as e:
            logger.error(f"Failed to send daily summary: {e}")
            return False
    
    def send_system_status(self, status_data: Dict[str, Any]) -> bool:
        """
        Send system status update.
        
        Args:
            status_data: Dictionary with system status information
            
        Returns:
            True if alert sent successfully, False otherwise
        """
        if not self.is_enabled:
            return False
        
        try:
            message = f"""
🖥️ <b>SYSTEM STATUS UPDATE</b>

<b>🔄 Services:</b>
• Data Fetcher: {'✅' if status_data.get('data_fetcher_ok') else '❌'}
• Trading Engine: {'✅' if status_data.get('trading_engine_ok') else '❌'}
• ML Model: {'✅' if status_data.get('ml_model_ok') else '❌'}
• Google Sheets: {'✅' if status_data.get('sheets_ok') else '❌'}

<b>📡 Connectivity:</b>
• Alpha Vantage API: {'✅' if status_data.get('api_connected') else '❌'}
• Internet: {'✅' if status_data.get('internet_ok') else '❌'}

<b>💾 Resources:</b>
• CPU Usage: {status_data.get('cpu_usage', 0):.1f}%
• Memory Usage: {status_data.get('memory_usage', 0):.1f}%
• Disk Space: {status_data.get('disk_space', 0):.1f}% used

<b>⏰ Last Updated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            return self.send_message_sync(message)
            
        except Exception as e:
            logger.error(f"Failed to send system status: {e}")
            return False
    
    def test_connection(self) -> bool:
        """
        Test Telegram bot connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not self.is_enabled:
            logger.info("Telegram bot is not enabled")
            return False
        
        test_message = f"""
🤖 <b>TELEGRAM BOT TEST</b>

Connection test successful! ✅

<b>Bot Token:</b> {self.bot_token[:10]}...
<b>Chat ID:</b> {self.chat_id}
<b>Test Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

The algo-trading bot is ready to send alerts! 🚀
"""
        
        success = self.send_message_sync(test_message)
        
        if success:
            logger.success("Telegram connection test passed")
        else:
            logger.error("Telegram connection test failed")
        
        return success
    
    def is_enabled_status(self) -> Dict[str, Any]:
        """
        Get Telegram bot status information.
        
        Returns:
            Dictionary with status information
        """
        return {
            'enabled': self.is_enabled,
            'telegram_available': TELEGRAM_AVAILABLE,
            'bot_token_configured': bool(self.bot_token),
            'chat_id_configured': bool(self.chat_id),
            'bot_initialized': self.bot is not None
        }