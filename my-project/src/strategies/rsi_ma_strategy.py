"""
RSI + Moving Average Crossover Trading Strategy.
Implements buy signals when RSI < 30 and 20-DMA crosses above 50-DMA.
Includes comprehensive backtesting functionality.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns

from ..data.technical_indicators import TechnicalIndicators
from ..config import Config


class Trade:
    """Represents a single trade."""
    
    def __init__(self, symbol: str, action: str, quantity: int, price: float, 
                 timestamp: datetime, stop_loss: float = None, take_profit: float = None):
        self.symbol = symbol
        self.action = action  # 'BUY' or 'SELL'
        self.quantity = quantity
        self.price = price
        self.timestamp = timestamp
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.pnl = 0.0
        self.exit_price = None
        self.exit_timestamp = None
        self.exit_reason = None
        
    def close_trade(self, exit_price: float, exit_timestamp: datetime, reason: str):
        """Close the trade and calculate P&L."""
        self.exit_price = exit_price
        self.exit_timestamp = exit_timestamp
        self.exit_reason = reason
        
        if self.action == 'BUY':
            self.pnl = (exit_price - self.price) * self.quantity
        else:  # SELL
            self.pnl = (self.price - exit_price) * self.quantity
    
    def to_dict(self) -> Dict:
        """Convert trade to dictionary for logging."""
        return {
            'symbol': self.symbol,
            'action': self.action,
            'quantity': self.quantity,
            'entry_price': self.price,
            'entry_timestamp': self.timestamp,
            'exit_price': self.exit_price,
            'exit_timestamp': self.exit_timestamp,
            'exit_reason': self.exit_reason,
            'pnl': self.pnl,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit
        }


class RSIMACrossoverStrategy:
    """RSI + Moving Average Crossover Trading Strategy."""
    
    def __init__(self, initial_capital: float = 100000, risk_per_trade: float = 0.02):
        """
        Initialize the trading strategy.
        
        Args:
            initial_capital: Starting capital
            risk_per_trade: Risk percentage per trade (0.02 = 2%)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.trades = []
        self.positions = {}  # Current positions by symbol
        self.trade_log = []
        
        # Strategy parameters
        self.rsi_oversold = Config.RSI_OVERSOLD
        self.rsi_overbought = Config.RSI_OVERBOUGHT
        self.short_ma = Config.SHORT_MA
        self.long_ma = Config.LONG_MA
        
        logger.info(f"RSI MA Crossover Strategy initialized with ${initial_capital:,.2f} capital")
    
    def calculate_position_size(self, price: float, atr: float) -> int:
        """
        Calculate position size based on risk management.
        
        Args:
            price: Current stock price
            atr: Average True Range for volatility
            
        Returns:
            Number of shares to trade
        """
        # Risk amount per trade
        risk_amount = self.current_capital * self.risk_per_trade
        
        # Use ATR for stop loss calculation (2 * ATR below entry price)
        stop_loss_distance = max(2 * atr, price * 0.05)  # At least 5% stop loss
        
        # Calculate position size
        if stop_loss_distance > 0:
            position_size = int(risk_amount / stop_loss_distance)
        else:
            position_size = int(risk_amount / (price * 0.05))  # Default 5% risk
        
        # Ensure we can afford the position
        max_affordable = int(self.current_capital * 0.2 / price)  # Max 20% of capital per stock
        position_size = min(position_size, max_affordable)
        
        return max(position_size, 1)  # At least 1 share
    
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Generate buy/sell signals based on RSI and MA crossover strategy.
        
        Args:
            data: DataFrame with OHLCV data and technical indicators
            symbol: Stock symbol
            
        Returns:
            DataFrame with signals added
        """
        if data.empty:
            logger.warning(f"No data provided for signal generation for {symbol}")
            return data
        
        # Ensure we have all required indicators
        if not all(col in data.columns for col in ['rsi', 'sma_20', 'sma_50']):
            logger.error(f"Missing required indicators for {symbol}")
            return data
        
        signals_df = data.copy()
        signals_df['signal'] = 0
        signals_df['position'] = 0
        signals_df['signal_strength'] = 'NONE'
        
        # Generate buy signals - more practical approach
        # Either RSI oversold with bullish MA setup OR MA crossover with reasonable RSI
        ma_bullish = signals_df['sma_20'] > signals_df['sma_50']
        ma_crossover = (signals_df['sma_20'] > signals_df['sma_50']) & (signals_df['sma_20'].shift(1) <= signals_df['sma_50'].shift(1))
        rsi_oversold = signals_df['rsi'] < self.rsi_oversold
        rsi_reasonable = signals_df['rsi'] < 50  # Less restrictive RSI condition
        
        buy_condition = (
            (rsi_oversold & ma_bullish) |  # RSI oversold in bullish trend
            (ma_crossover & rsi_reasonable)  # MA crossover with reasonable RSI
        )
        
        # Generate sell signals - more balanced approach
        ma_bearish = signals_df['sma_20'] < signals_df['sma_50']
        ma_crossover_down = (signals_df['sma_20'] < signals_df['sma_50']) & (signals_df['sma_20'].shift(1) >= signals_df['sma_50'].shift(1))
        rsi_overbought = signals_df['rsi'] > self.rsi_overbought
        rsi_high = signals_df['rsi'] > 65  # Less restrictive sell condition
        
        sell_condition = (
            (rsi_overbought) |  # RSI overbought
            (ma_crossover_down) |  # MA crossover down
            (rsi_high & ma_bearish)  # High RSI in bearish trend
        )
        
        # Set signals
        signals_df.loc[buy_condition, 'signal'] = 1
        signals_df.loc[sell_condition, 'signal'] = -1
        
        # Debug: Log signal statistics
        total_buy_signals = buy_condition.sum()
        total_sell_signals = sell_condition.sum()
        logger.info(f"{symbol}: Generated {total_buy_signals} buy signals and {total_sell_signals} sell signals out of {len(signals_df)} periods")
        
        # Calculate signal strength
        for idx in signals_df.index:
            if signals_df.loc[idx, 'signal'] != 0:
                rsi_val = signals_df.loc[idx, 'rsi']
                macd_hist = signals_df.loc[idx, 'histogram'] if 'histogram' in signals_df.columns else 0
                vol_ratio = signals_df.loc[idx, 'volume_ratio'] if 'volume_ratio' in signals_df.columns else 1
                
                signals_df.loc[idx, 'signal_strength'] = TechnicalIndicators.get_signal_strength(
                    rsi_val, macd_hist, vol_ratio
                )
        
        # Generate position changes
        signals_df['position'] = signals_df['signal'].replace(to_replace=0, method='ffill').fillna(0)
        
        return signals_df
    
    def backtest(self, data_dict: Dict[str, pd.DataFrame], start_date: str = None, 
                end_date: str = None, sheets_logger=None) -> Dict:
        """
        Perform backtesting on historical data.
        
        Args:
            data_dict: Dictionary of symbol -> DataFrame with OHLCV and indicators
            start_date: Start date for backtesting (YYYY-MM-DD)
            end_date: End date for backtesting (YYYY-MM-DD)
            sheets_logger: Optional Google Sheets logger for trade logging
            
        Returns:
            Dictionary with backtest results
        """
        logger.info("Starting backtesting...")
        
        # Store sheets logger reference
        self.sheets_logger = sheets_logger
        
        # Reset strategy state
        self.current_capital = self.initial_capital
        self.trades = []
        self.positions = {}
        self.trade_log = []
        
        # Prepare data
        all_dates = set()
        processed_data = {}
        
        for symbol, data in data_dict.items():
            if data.empty:
                continue
                
            # Filter by date range if specified
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]
            
            # Calculate indicators if not present
            if 'rsi' not in data.columns:
                data = TechnicalIndicators.calculate_all_indicators(data)
            
            # Generate signals
            data_with_signals = self.generate_signals(data, symbol)
            processed_data[symbol] = data_with_signals
            all_dates.update(data.index)
            
            # Debug: Check signal generation
            signal_summary = data_with_signals['signal'].value_counts()
            logger.info(f"{symbol} signal distribution: {signal_summary.to_dict()}")
        
        # Sort dates for chronological processing
        sorted_dates = sorted(list(all_dates))
        
        # Track portfolio value over time
        portfolio_history = []
        
        # Process each trading day
        for current_date in sorted_dates:
            daily_portfolio_value = self.current_capital
            
            # Process each symbol
            for symbol, data in processed_data.items():
                if current_date not in data.index:
                    continue
                
                current_row = data.loc[current_date]
                self._process_trading_signals(symbol, current_row, current_date)
                
                # Add position values to portfolio
                if symbol in self.positions:
                    position_value = self.positions[symbol]['quantity'] * current_row['close']
                    daily_portfolio_value += position_value
            
            portfolio_history.append({
                'date': current_date,
                'portfolio_value': daily_portfolio_value,
                'cash': self.current_capital,
                'positions_value': daily_portfolio_value - self.current_capital
            })
        
        # Calculate final results
        results = self._calculate_backtest_results(portfolio_history)
        
        # Log portfolio summary to Google Sheets if available
        if hasattr(self, 'sheets_logger') and self.sheets_logger and portfolio_history:
            try:
                # Log the final portfolio summary
                final_summary = {
                    'date': portfolio_history[-1]['date'],
                    'total_capital': portfolio_history[-1]['portfolio_value'],
                    'cash': portfolio_history[-1]['cash'],
                    'positions_value': portfolio_history[-1]['positions_value'],
                    'total_pnl': portfolio_history[-1]['portfolio_value'] - self.initial_capital,
                    'cumulative_return': results.get('total_return', 0),
                    'active_positions': len(self.positions),
                    'max_drawdown': results.get('max_drawdown', 0),
                    'win_rate': results.get('win_rate', 0)
                }
                self.sheets_logger.update_portfolio_summary(final_summary)
                logger.info("Portfolio summary logged to Google Sheets")
            except Exception as e:
                logger.warning(f"Failed to log portfolio summary to Google Sheets: {e}")
        
        logger.success(f"Backtesting completed. Total return: {results['total_return']:.2%}")
        
        # Show mock data summary if using mock logger
        if (hasattr(self, 'sheets_logger') and self.sheets_logger and 
            hasattr(self.sheets_logger, 'print_summary')):
            self.sheets_logger.print_summary()
        
        return results
    
    def _process_trading_signals(self, symbol: str, row: pd.Series, date: datetime):
        """Process trading signals for a specific symbol and date."""
        signal = row['signal']
        price = row['close']
        atr = row.get('atr', price * 0.02)  # Default 2% if ATR not available
        
        # Check for buy signal
        if signal == 1 and symbol not in self.positions:
            quantity = self.calculate_position_size(price, atr)
            cost = quantity * price
            
            if cost <= self.current_capital * 0.95:  # Keep 5% cash buffer
                # Calculate stop loss and take profit
                stop_loss = price - (2 * atr)
                take_profit = price + (3 * atr)  # 1.5:1 risk-reward ratio
                
                # Create trade
                trade = Trade(symbol, 'BUY', quantity, price, date, stop_loss, take_profit)
                self.trades.append(trade)
                
                # Update position
                self.positions[symbol] = {
                    'quantity': quantity,
                    'entry_price': price,
                    'entry_date': date,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'trade': trade
                }
                
                # Update capital
                self.current_capital -= cost
                
                # Log to Google Sheets if available
                if hasattr(self, 'sheets_logger') and self.sheets_logger:
                    try:
                        trade_data = {
                            'timestamp': date,
                            'symbol': symbol,
                            'action': 'BUY',
                            'quantity': quantity,
                            'entry_price': price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'rsi': row.get('rsi', 0),
                            'sma_20': row.get('sma_20', 0),
                            'sma_50': row.get('sma_50', 0),
                            'signal_strength': row.get('signal_strength', 'UNKNOWN')
                        }
                        self.sheets_logger.log_trade(trade_data)
                    except Exception as e:
                        logger.warning(f"Failed to log trade to Google Sheets: {e}")
                
                logger.info(f"BUY: {quantity} shares of {symbol} at ${price:.2f}")
        
        # Check for sell signal or stop loss/take profit
        elif symbol in self.positions:
            position = self.positions[symbol]
            should_sell = False
            exit_reason = None
            
            # Check sell signal
            if signal == -1:
                should_sell = True
                exit_reason = "SELL_SIGNAL"
            
            # Check stop loss
            elif price <= position['stop_loss']:
                should_sell = True
                exit_reason = "STOP_LOSS"
            
            # Check take profit
            elif price >= position['take_profit']:
                should_sell = True
                exit_reason = "TAKE_PROFIT"
            
            if should_sell:
                quantity = position['quantity']
                proceeds = quantity * price
                
                # Close the trade
                trade = position['trade']
                trade.close_trade(price, date, exit_reason)
                
                # Update capital
                self.current_capital += proceeds
                
                # Log completed trade to Google Sheets if available
                if hasattr(self, 'sheets_logger') and self.sheets_logger:
                    try:
                        completed_trade_data = {
                            'timestamp': date,
                            'symbol': symbol,
                            'action': 'SELL',
                            'quantity': quantity,
                            'entry_price': trade.entry_price,
                            'exit_price': price,
                            'entry_date': trade.entry_date,
                            'exit_date': date,
                            'exit_reason': exit_reason,
                            'pnl': trade.pnl,
                            'pnl_percent': trade.pnl_percent,
                            'stop_loss': trade.stop_loss,
                            'take_profit': trade.take_profit,
                            'trade_duration': (date - trade.entry_date).days if hasattr(trade, 'entry_date') else 0,
                            'rsi': row.get('rsi', 0),
                            'sma_20': row.get('sma_20', 0),
                            'sma_50': row.get('sma_50', 0),
                            'signal_strength': row.get('signal_strength', 'UNKNOWN')
                        }
                        self.sheets_logger.log_trade(completed_trade_data)
                    except Exception as e:
                        logger.warning(f"Failed to log completed trade to Google Sheets: {e}")
                
                # Remove position
                del self.positions[symbol]
                
                logger.info(f"SELL: {quantity} shares of {symbol} at ${price:.2f} ({exit_reason})")
    
    def _calculate_backtest_results(self, portfolio_history: List[Dict]) -> Dict:
        """Calculate comprehensive backtest results."""
        if not portfolio_history:
            return {}
        
        # Convert to DataFrame for easier analysis
        portfolio_df = pd.DataFrame(portfolio_history)
        portfolio_df.set_index('date', inplace=True)
        
        # Basic metrics
        initial_value = portfolio_df['portfolio_value'].iloc[0]
        final_value = portfolio_df['portfolio_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate daily returns
        portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
        daily_returns = portfolio_df['daily_return'].dropna()
        
        # Risk metrics
        volatility = daily_returns.std() * np.sqrt(252)  # Annualized
        if daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))
        else:
            sharpe_ratio = 0.0  # No volatility means no Sharpe ratio
        
        # Maximum drawdown
        portfolio_df['peak'] = portfolio_df['portfolio_value'].expanding().max()
        portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['peak']) / portfolio_df['peak']
        max_drawdown = portfolio_df['drawdown'].min()
        
        # Trade analysis
        completed_trades = [trade for trade in self.trades if trade.exit_price is not None]
        winning_trades = [trade for trade in completed_trades if trade.pnl > 0]
        losing_trades = [trade for trade in completed_trades if trade.pnl < 0]
        
        win_rate = len(winning_trades) / len(completed_trades) if completed_trades else 0
        avg_win = np.mean([trade.pnl for trade in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([trade.pnl for trade in losing_trades]) if losing_trades else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Time period
        start_date = portfolio_df.index[0]
        end_date = portfolio_df.index[-1]
        trading_days = len(portfolio_df)
        
        results = {
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'trading_days': trading_days,
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'annualized_return': (1 + total_return) ** (252 / trading_days) - 1,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(completed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'portfolio_history': portfolio_df,
            'trade_details': [trade.to_dict() for trade in completed_trades]
        }
        
        return results
    
    def plot_backtest_results(self, results: Dict, save_path: str = None):
        """Plot backtest results including portfolio value and drawdown."""
        if not results or 'portfolio_history' not in results:
            logger.error("No backtest results to plot")
            return
        
        portfolio_df = results['portfolio_history']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Backtest Results: RSI + MA Crossover Strategy', fontsize=16)
        
        # Portfolio value over time
        axes[0, 0].plot(portfolio_df.index, portfolio_df['portfolio_value'], linewidth=2)
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Drawdown
        axes[0, 1].fill_between(portfolio_df.index, portfolio_df['drawdown'], 0, 
                               color='red', alpha=0.3)
        axes[0, 1].plot(portfolio_df.index, portfolio_df['drawdown'], color='red', linewidth=2)
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Daily returns distribution
        daily_returns = portfolio_df['daily_return'].dropna()
        axes[1, 0].hist(daily_returns, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(daily_returns.mean(), color='red', linestyle='--', 
                          label=f'Mean: {daily_returns.mean():.4f}')
        axes[1, 0].set_title('Daily Returns Distribution')
        axes[1, 0].set_xlabel('Daily Return')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Monthly returns heatmap
        monthly_returns = portfolio_df['portfolio_value'].resample('M').last().pct_change().dropna()
        monthly_returns.index = monthly_returns.index.strftime('%Y-%m')
        
        if len(monthly_returns) > 1:
            # Create a simple bar plot for monthly returns
            axes[1, 1].bar(range(len(monthly_returns)), monthly_returns.values)
            axes[1, 1].set_title('Monthly Returns')
            axes[1, 1].set_ylabel('Monthly Return (%)')
            axes[1, 1].set_xticks(range(len(monthly_returns)))
            axes[1, 1].set_xticklabels(monthly_returns.index, rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Backtest plot saved to {save_path}")
        
        plt.show()
    
    def get_current_signals(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Get current trading signals for all symbols.
        
        Args:
            data_dict: Dictionary of symbol -> DataFrame with latest data
            
        Returns:
            Dictionary with current signals for each symbol
        """
        current_signals = {}
        
        for symbol, data in data_dict.items():
            if data.empty:
                continue
            
            # Ensure indicators are calculated
            if 'rsi' not in data.columns:
                data = TechnicalIndicators.calculate_all_indicators(data)
            
            # Get latest data point
            latest = data.iloc[-1]
            
            # Determine signal
            signal_type = 'NONE'
            signal_strength = 'NONE'
            
            # Check buy conditions
            if (latest['rsi'] < self.rsi_oversold and 
                latest['sma_20'] > latest['sma_50']):
                signal_type = 'BUY'
                signal_strength = TechnicalIndicators.get_signal_strength(
                    latest['rsi'], 
                    latest.get('histogram', 0),
                    latest.get('volume_ratio', 1)
                )
            
            # Check sell conditions
            elif (latest['rsi'] > self.rsi_overbought or 
                  latest['sma_20'] < latest['sma_50']):
                signal_type = 'SELL'
                signal_strength = TechnicalIndicators.get_signal_strength(
                    latest['rsi'], 
                    latest.get('histogram', 0),
                    latest.get('volume_ratio', 1)
                )
            
            current_signals[symbol] = {
                'signal': signal_type,
                'strength': signal_strength,
                'rsi': latest['rsi'],
                'sma_20': latest['sma_20'],
                'sma_50': latest['sma_50'],
                'price': latest['close'],
                'volume_ratio': latest.get('volume_ratio', 1),
                'timestamp': latest.name
            }
        
        return current_signals