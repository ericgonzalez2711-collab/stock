"""
Improved RSI + Moving Average Crossover Trading Strategy.
Enhanced version with better risk management, signal filtering, and market regime detection.
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
from .rsi_ma_strategy import Trade, RSIMACrossoverStrategy


class ImprovedRSIMACrossoverStrategy(RSIMACrossoverStrategy):
    """Enhanced RSI + MA crossover strategy with improved risk management."""
    
    def __init__(self, initial_capital: float = 100000, risk_per_trade: float = 0.015):
        """
        Initialize improved strategy with better default risk management.
        
        Args:
            initial_capital: Starting capital
            risk_per_trade: Risk per trade (reduced from 2% to 1.5%)
        """
        super().__init__(initial_capital, risk_per_trade)
        
        # Enhanced parameters
        self.rsi_oversold = 25  # More restrictive (was 30)
        self.rsi_overbought = 75  # More restrictive (was 70)
        self.volume_threshold = 1.2  # Require 20% above average volume
        self.trend_strength_threshold = 0.6  # Market regime filter
        self.max_correlation = 0.7  # Portfolio diversification
        
        # Trailing stop parameters
        self.trailing_stop_pct = 0.15  # 15% trailing stop
        self.breakeven_buffer = 0.02  # Move stop to breakeven after 2% profit
        
        logger.info("Improved RSI+MA strategy initialized with enhanced risk management")
    
    def _calculate_market_regime(self, data: pd.DataFrame) -> str:
        """
        Determine current market regime (bullish, bearish, sideways).
        
        Args:
            data: DataFrame with OHLCV and indicators
            
        Returns:
            Market regime: 'BULLISH', 'BEARISH', or 'SIDEWAYS'
        """
        if len(data) < 50:
            return 'SIDEWAYS'
        
        # Use multiple timeframe analysis
        sma_20 = data['sma_20'].iloc[-1]
        sma_50 = data['sma_50'].iloc[-1]
        sma_200 = data['close'].rolling(200).mean().iloc[-1] if len(data) >= 200 else sma_50
        
        # Price position relative to moving averages
        current_price = data['close'].iloc[-1]
        
        # Trend strength calculation
        price_above_sma20 = current_price > sma_20
        sma20_above_sma50 = sma_20 > sma_50
        sma50_above_sma200 = sma_50 > sma_200
        
        # Calculate trend momentum
        price_momentum = data['close'].pct_change(20).iloc[-1]
        
        # Determine regime
        bullish_signals = sum([price_above_sma20, sma20_above_sma50, sma50_above_sma200, price_momentum > 0])
        
        if bullish_signals >= 3:
            return 'BULLISH'
        elif bullish_signals <= 1:
            return 'BEARISH'
        else:
            return 'SIDEWAYS'
    
    def _check_volume_confirmation(self, data: pd.DataFrame, lookback: int = 20) -> bool:
        """
        Check if current volume supports the signal.
        
        Args:
            data: DataFrame with volume data
            lookback: Period for average volume calculation
            
        Returns:
            True if volume is above threshold
        """
        if 'volume' not in data.columns or len(data) < lookback:
            return True  # Default to True if no volume data
        
        current_volume = data['volume'].iloc[-1]
        avg_volume = data['volume'].rolling(lookback).mean().iloc[-1]
        
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        return volume_ratio >= self.volume_threshold
    
    def _calculate_volatility_adjusted_stops(self, price: float, atr: float, 
                                           market_regime: str) -> Tuple[float, float]:
        """
        Calculate dynamic stop loss and take profit based on volatility and market regime.
        
        Args:
            price: Entry price
            atr: Average True Range
            market_regime: Current market regime
            
        Returns:
            Tuple of (stop_loss, take_profit)
        """
        # Base multipliers
        stop_multiplier = 2.0
        profit_multiplier = 3.0
        
        # Adjust based on market regime
        if market_regime == 'BULLISH':
            stop_multiplier = 1.8  # Tighter stops in trending markets
            profit_multiplier = 4.0  # Higher targets
        elif market_regime == 'BEARISH':
            stop_multiplier = 1.5  # Very tight stops in bear markets
            profit_multiplier = 2.5  # Conservative targets
        else:  # SIDEWAYS
            stop_multiplier = 2.5  # Wider stops in choppy markets
            profit_multiplier = 3.0
        
        # Calculate levels
        stop_distance = max(atr * stop_multiplier, price * 0.03)  # Min 3% stop
        profit_distance = atr * profit_multiplier
        
        stop_loss = price - stop_distance
        take_profit = price + profit_distance
        
        return stop_loss, take_profit
    
    def _update_trailing_stops(self, symbol: str, current_price: float):
        """
        Update trailing stops for existing positions.
        
        Args:
            symbol: Stock symbol
            current_price: Current market price
        """
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        trade = position['trade']
        entry_price = position['entry_price']
        
        # Calculate current profit percentage
        profit_pct = (current_price - entry_price) / entry_price
        
        # Move to breakeven after 2% profit
        if profit_pct >= self.breakeven_buffer:
            new_stop = max(position['stop_loss'], entry_price * 1.001)  # Breakeven + small buffer
        else:
            # Calculate trailing stop
            trailing_stop = current_price * (1 - self.trailing_stop_pct)
            new_stop = max(position['stop_loss'], trailing_stop)
        
        # Update stop loss
        if new_stop > position['stop_loss']:
            position['stop_loss'] = new_stop
            trade.stop_loss = new_stop
            logger.info(f"Updated trailing stop for {symbol}: ${new_stop:.2f}")
    
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Generate improved buy/sell signals with enhanced filtering.
        
        Args:
            data: DataFrame with OHLCV data and technical indicators
            symbol: Stock symbol
            
        Returns:
            DataFrame with enhanced signals
        """
        if data.empty:
            logger.warning(f"No data provided for signal generation for {symbol}")
            return data
        
        # Ensure we have all required indicators
        required_indicators = ['rsi', 'sma_20', 'sma_50', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'atr']
        if not all(col in data.columns for col in required_indicators):
            logger.error(f"Missing required indicators for {symbol}")
            return data
        
        signals_df = data.copy()
        signals_df['signal'] = 0
        signals_df['position'] = 0
        signals_df['signal_strength'] = 'NONE'
        signals_df['market_regime'] = 'SIDEWAYS'
        
        # Calculate additional indicators for filtering
        signals_df['price_change_5d'] = signals_df['close'].pct_change(5)
        signals_df['price_change_20d'] = signals_df['close'].pct_change(20)
        signals_df['volume_ratio'] = signals_df['volume'] / signals_df['volume'].rolling(20).mean() if 'volume' in signals_df.columns else 1
        
        # Market regime analysis
        for idx in range(50, len(signals_df)):  # Start after enough data for indicators
            market_regime = self._calculate_market_regime(signals_df.iloc[:idx+1])
            signals_df.iloc[idx, signals_df.columns.get_loc('market_regime')] = market_regime
        
        # Enhanced buy conditions
        for idx in range(50, len(signals_df)):
            row = signals_df.iloc[idx]
            prev_row = signals_df.iloc[idx-1]
            
            # Basic conditions
            rsi_oversold = row['rsi'] < self.rsi_oversold
            ma_bullish = row['sma_20'] > row['sma_50']
            ma_crossover = (row['sma_20'] > row['sma_50']) and (prev_row['sma_20'] <= prev_row['sma_50'])
            
            # Enhanced filters
            volume_confirmation = row['volume_ratio'] >= self.volume_threshold if 'volume' in signals_df.columns else True
            macd_bullish = row['macd'] > row['macd_signal']
            price_above_bb_middle = row['close'] > (row['bb_upper'] + row['bb_lower']) / 2
            not_in_strong_downtrend = row['price_change_20d'] > -0.15  # Not in severe downtrend
            
            # Market regime filter - only trade in favorable conditions
            favorable_regime = row['market_regime'] in ['BULLISH', 'SIDEWAYS']
            
            # Momentum confirmation
            recent_momentum_positive = row['price_change_5d'] > -0.05  # Not falling too fast
            
            # Combined buy condition (much more restrictive)
            buy_condition = (
                rsi_oversold and 
                ma_bullish and 
                volume_confirmation and
                macd_bullish and
                not_in_strong_downtrend and
                favorable_regime and
                recent_momentum_positive
            )
            
            # MA crossover buy (alternative entry)
            crossover_buy = (
                ma_crossover and 
                row['rsi'] < 40 and  # More restrictive RSI for crossover
                volume_confirmation and
                favorable_regime and
                row['price_change_5d'] > 0  # Recent momentum
            )
            
            if buy_condition or crossover_buy:
                signals_df.iloc[idx, signals_df.columns.get_loc('signal')] = 1
                
                # Calculate signal strength
                strength_score = 0
                strength_score += 1 if rsi_oversold else 0
                strength_score += 1 if ma_crossover else 0
                strength_score += 1 if volume_confirmation else 0
                strength_score += 1 if macd_bullish else 0
                strength_score += 1 if row['market_regime'] == 'BULLISH' else 0
                
                if strength_score >= 4:
                    signals_df.iloc[idx, signals_df.columns.get_loc('signal_strength')] = 'STRONG'
                elif strength_score >= 3:
                    signals_df.iloc[idx, signals_df.columns.get_loc('signal_strength')] = 'MODERATE'
                else:
                    signals_df.iloc[idx, signals_df.columns.get_loc('signal_strength')] = 'WEAK'
            
            # Enhanced sell conditions
            rsi_overbought = row['rsi'] > self.rsi_overbought
            ma_bearish = row['sma_20'] < row['sma_50']
            ma_crossover_down = (row['sma_20'] < row['sma_50']) and (prev_row['sma_20'] >= prev_row['sma_50'])
            macd_bearish = row['macd'] < row['macd_signal']
            strong_downtrend = row['price_change_5d'] < -0.05
            
            sell_condition = (
                rsi_overbought or 
                ma_crossover_down or 
                (macd_bearish and row['rsi'] > 60) or
                strong_downtrend
            )
            
            if sell_condition:
                signals_df.iloc[idx, signals_df.columns.get_loc('signal')] = -1
        
        # Generate position changes
        signals_df['position'] = signals_df['signal'].replace(to_replace=0, method='ffill').fillna(0)
        
        # Log signal statistics
        total_buy_signals = (signals_df['signal'] == 1).sum()
        total_sell_signals = (signals_df['signal'] == -1).sum()
        logger.info(f"{symbol}: Generated {total_buy_signals} buy signals and {total_sell_signals} sell signals (Improved Strategy)")
        
        return signals_df
    
    def _process_trading_signals(self, symbol: str, row: pd.Series, date):
        """
        Enhanced signal processing with improved risk management.
        """
        signal = row.get('signal', 0)
        price = row['close']
        atr = row.get('atr', price * 0.02)
        market_regime = row.get('market_regime', 'SIDEWAYS')
        
        # Update trailing stops for existing positions
        if symbol in self.positions:
            self._update_trailing_stops(symbol, price)
        
        # Process buy signal
        if signal == 1 and symbol not in self.positions and len(self.positions) < self.max_positions:
            # Additional validation before entry
            volume_ok = self._check_volume_confirmation(pd.DataFrame([row]))
            
            # Check portfolio correlation (avoid overconcentration)
            if self._check_portfolio_diversification(symbol):
                # Calculate position size with enhanced risk management
                position_size = self._calculate_enhanced_position_size(price, atr, market_regime)
                cost = position_size * price
                
                if cost <= self.current_capital:
                    # Calculate dynamic stops
                    stop_loss, take_profit = self._calculate_volatility_adjusted_stops(
                        price, atr, market_regime
                    )
                    
                    # Create trade
                    trade = Trade(symbol, 'BUY', position_size, price, date, stop_loss, take_profit)
                    self.trades.append(trade)
                    
                    # Update position
                    self.positions[symbol] = {
                        'quantity': position_size,
                        'entry_price': price,
                        'entry_date': date,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'trade': trade,
                        'highest_price': price,  # For trailing stops
                        'market_regime': market_regime
                    }
                    
                    # Update capital
                    self.current_capital -= cost
                    
                    logger.info(f"BUY: {position_size} shares of {symbol} at ${price:.2f} (Regime: {market_regime})")
        
        # Process sell signal or risk management exits
        elif symbol in self.positions:
            position = self.positions[symbol]
            should_sell = False
            exit_reason = None
            
            # Update highest price for trailing stops
            if price > position['highest_price']:
                position['highest_price'] = price
            
            # Check various exit conditions
            if signal == -1:
                should_sell = True
                exit_reason = "SELL_SIGNAL"
            elif price <= position['stop_loss']:
                should_sell = True
                exit_reason = "STOP_LOSS"
            elif price >= position['take_profit']:
                should_sell = True
                exit_reason = "TAKE_PROFIT"
            
            # Additional risk management exits
            elif self._check_emergency_exit(symbol, row, position):
                should_sell = True
                exit_reason = "RISK_MANAGEMENT"
            
            if should_sell:
                quantity = position['quantity']
                proceeds = quantity * price
                
                # Close the trade
                trade = position['trade']
                trade.close_trade(price, date, exit_reason)
                
                # Update capital
                self.current_capital += proceeds
                
                # Remove position
                del self.positions[symbol]
                
                logger.info(f"SELL: {quantity} shares of {symbol} at ${price:.2f} ({exit_reason})")
    
    def _calculate_enhanced_position_size(self, price: float, atr: float, market_regime: str) -> int:
        """
        Calculate position size with regime-based adjustments.
        
        Args:
            price: Current stock price
            atr: Average True Range
            market_regime: Current market regime
            
        Returns:
            Position size in shares
        """
        # Adjust risk based on market regime
        regime_risk_multiplier = {
            'BULLISH': 1.0,    # Normal risk
            'SIDEWAYS': 0.8,   # Reduced risk
            'BEARISH': 0.5     # Much reduced risk
        }
        
        adjusted_risk = self.risk_per_trade * regime_risk_multiplier.get(market_regime, 0.8)
        risk_amount = self.current_capital * adjusted_risk
        
        # Use ATR for stop loss calculation
        stop_loss_distance = max(2 * atr, price * 0.025)  # Min 2.5% stop loss
        
        # Calculate position size
        if stop_loss_distance > 0:
            position_size = int(risk_amount / stop_loss_distance)
        else:
            position_size = int(risk_amount / (price * 0.025))
        
        # Portfolio limits
        max_position_value = self.current_capital * 0.15  # Max 15% per position
        max_affordable = int(max_position_value / price)
        position_size = min(position_size, max_affordable)
        
        return max(position_size, 1)
    
    def _check_portfolio_diversification(self, new_symbol: str) -> bool:
        """
        Check if adding new position maintains portfolio diversification.
        
        Args:
            new_symbol: Symbol to potentially add
            
        Returns:
            True if position can be added without over-concentration
        """
        # Simple sector diversification (basic implementation)
        # In practice, you'd have sector mapping
        if len(self.positions) >= 3:  # Limit concurrent positions
            return False
        
        return True
    
    def _check_emergency_exit(self, symbol: str, row: pd.Series, position: Dict) -> bool:
        """
        Check for emergency exit conditions.
        
        Args:
            symbol: Stock symbol
            row: Current market data
            position: Current position details
            
        Returns:
            True if emergency exit is needed
        """
        current_price = row['close']
        entry_price = position['entry_price']
        
        # Large gap down (more than 5%)
        if current_price < entry_price * 0.95:
            return True
        
        # RSI extremely overbought (> 85)
        if row.get('rsi', 50) > 85:
            return True
        
        # Market regime changed to strongly bearish
        if row.get('market_regime') == 'BEARISH' and current_price < entry_price:
            return True
        
        return False
    
    def backtest_with_comparison(self, data_dict: Dict[str, pd.DataFrame], 
                               start_date: str = None, end_date: str = None) -> Dict:
        """
        Run backtest and compare with original strategy.
        
        Args:
            data_dict: Dictionary of symbol -> DataFrame with OHLCV and indicators
            start_date: Start date for backtesting
            end_date: End date for backtesting
            
        Returns:
            Dictionary with improved backtest results and comparison
        """
        logger.info("Running improved strategy backtest...")
        
        # Run improved strategy backtest
        improved_results = self.backtest(data_dict, start_date, end_date)
        
        # Run original strategy for comparison
        logger.info("Running original strategy for comparison...")
        original_strategy = RSIMACrossoverStrategy(self.initial_capital, 0.02)
        original_results = original_strategy.backtest(data_dict, start_date, end_date)
        
        # Calculate improvements
        improvement_metrics = {
            'return_improvement': improved_results['total_return'] - original_results['total_return'],
            'sharpe_improvement': improved_results['sharpe_ratio'] - original_results['sharpe_ratio'],
            'drawdown_improvement': original_results['max_drawdown'] - improved_results['max_drawdown'],
            'win_rate_improvement': improved_results['win_rate'] - original_results['win_rate'],
            'trade_count_change': improved_results['total_trades'] - original_results['total_trades']
        }
        
        # Combine results
        comparison_results = {
            'improved_strategy': improved_results,
            'original_strategy': original_results,
            'improvements': improvement_metrics,
            'recommendation': self._generate_strategy_recommendation(improvement_metrics)
        }
        
        logger.success("Strategy comparison completed")
        return comparison_results
    
    def _generate_strategy_recommendation(self, improvements: Dict) -> str:
        """Generate recommendation based on strategy comparison."""
        recommendations = []
        
        if improvements['return_improvement'] > 0.05:
            recommendations.append("✅ Significant return improvement")
        elif improvements['return_improvement'] > 0:
            recommendations.append("✅ Modest return improvement")
        else:
            recommendations.append("⚠️ Lower returns - consider further optimization")
        
        if improvements['sharpe_improvement'] > 0.2:
            recommendations.append("✅ Much better risk-adjusted returns")
        elif improvements['sharpe_improvement'] > 0:
            recommendations.append("✅ Better risk-adjusted returns")
        
        if improvements['drawdown_improvement'] > 0.02:
            recommendations.append("✅ Significantly reduced drawdown")
        
        if improvements['win_rate_improvement'] > 0.1:
            recommendations.append("✅ Much higher win rate")
        elif improvements['win_rate_improvement'] > 0:
            recommendations.append("✅ Higher win rate")
        
        if improvements['trade_count_change'] < 0:
            recommendations.append("✅ More selective trading (fewer false signals)")
        
        return " | ".join(recommendations) if recommendations else "Strategy needs further optimization"