#!/usr/bin/env python3
"""
Quick setup and test script for the improved trading strategy.
Can run with minimal dependencies to demonstrate improvements.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

print("🚀 Algo-Trading Strategy Improvement Test")
print("=" * 50)

# Check if we can run the full comparison
try:
    from src.config import Config
    from src.data.technical_indicators import TechnicalIndicators
    BASIC_MODULES_AVAILABLE = True
    print("✅ Basic modules available")
except ImportError as e:
    print(f"⚠️ Some modules missing: {e}")
    BASIC_MODULES_AVAILABLE = False

def create_sample_trading_data():
    """Create sample data to demonstrate strategy improvements."""
    print("\n📊 Creating sample trading scenario...")
    
    # Generate 180 days of sample data
    dates = pd.date_range(start='2023-06-01', end='2023-12-01', freq='D')
    
    # Simulate RELIANCE stock data with realistic patterns
    np.random.seed(42)
    base_price = 2500
    
    # Create a realistic price series with trends and volatility
    returns = []
    for i in range(len(dates)):
        # Add some market cycles
        cycle_component = 0.001 * np.sin(i * 2 * np.pi / 60)  # 60-day cycle
        trend_component = 0.0002 if i < 90 else -0.0001  # Trend change
        noise = np.random.normal(0, 0.015)  # 1.5% daily volatility
        
        daily_return = cycle_component + trend_component + noise
        returns.append(daily_return)
    
    # Calculate prices
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_price = close * (1 + np.random.normal(0, 0.005))
        volume = int(1000000 * (1 + np.random.normal(0, 0.3)))
        
        data.append({
            'date': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': max(volume, 100000)
        })
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    
    # Calculate basic technical indicators manually
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    
    # RSI calculation
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR calculation
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(14).mean()
    
    # Volume ratio
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    print(f"✅ Created {len(df)} days of sample data")
    return df

def simulate_original_strategy(data):
    """Simulate original strategy performance."""
    print("\n📊 Simulating Original Strategy...")
    
    capital = 100000
    positions = {}
    trades = []
    
    # Original strategy conditions (relaxed)
    for i in range(50, len(data)):
        row = data.iloc[i]
        prev_row = data.iloc[i-1]
        
        # Original buy conditions (too relaxed)
        rsi_condition = row['rsi'] < 30 or row['rsi'] < 50  # Too permissive
        ma_condition = row['sma_20'] > row['sma_50']
        ma_crossover = (row['sma_20'] > row['sma_50']) and (prev_row['sma_20'] <= prev_row['sma_50'])
        
        buy_signal = (rsi_condition and ma_condition) or ma_crossover
        
        # Original sell conditions
        sell_signal = row['rsi'] > 70 or row['sma_20'] < row['sma_50']
        
        # Execute trades
        if buy_signal and len(positions) == 0:
            position_size = int(capital * 0.2 / row['close'])  # 20% of capital
            cost = position_size * row['close']
            if cost <= capital:
                positions['RELIANCE'] = {
                    'quantity': position_size,
                    'entry_price': row['close'],
                    'entry_date': row.name
                }
                capital -= cost
                trades.append(('BUY', row.name, row['close'], position_size))
        
        elif sell_signal and 'RELIANCE' in positions:
            pos = positions['RELIANCE']
            proceeds = pos['quantity'] * row['close']
            pnl = proceeds - (pos['quantity'] * pos['entry_price'])
            capital += proceeds
            trades.append(('SELL', row.name, row['close'], pos['quantity'], pnl))
            del positions['RELIANCE']
    
    # Calculate final performance
    final_value = capital
    if 'RELIANCE' in positions:
        final_value += positions['RELIANCE']['quantity'] * data['close'].iloc[-1]
    
    total_return = (final_value - 100000) / 100000
    num_trades = len([t for t in trades if t[0] == 'BUY'])
    
    print(f"  💰 Final Value: ₹{final_value:,.0f}")
    print(f"  📈 Total Return: {total_return:.2%}")
    print(f"  🔄 Number of Trades: {num_trades}")
    
    return {
        'final_value': final_value,
        'total_return': total_return,
        'num_trades': num_trades,
        'trades': trades
    }

def simulate_improved_strategy(data):
    """Simulate improved strategy performance."""
    print("\n🎯 Simulating Improved Strategy...")
    
    capital = 100000
    positions = {}
    trades = []
    
    # Calculate market regime
    def get_market_regime(idx):
        if idx < 50:
            return 'SIDEWAYS'
        
        current_data = data.iloc[:idx+1]
        current_price = current_data['close'].iloc[-1]
        sma_20 = current_data['sma_20'].iloc[-1]
        sma_50 = current_data['sma_50'].iloc[-1]
        
        price_momentum = current_data['close'].pct_change(20).iloc[-1]
        
        bullish_signals = sum([
            current_price > sma_20,
            sma_20 > sma_50,
            price_momentum > 0
        ])
        
        if bullish_signals >= 2:
            return 'BULLISH'
        elif bullish_signals <= 1:
            return 'BEARISH'
        else:
            return 'SIDEWAYS'
    
    # Improved strategy conditions (much more restrictive)
    for i in range(50, len(data)):
        row = data.iloc[i]
        prev_row = data.iloc[i-1]
        
        market_regime = get_market_regime(i)
        
        # Improved buy conditions (much more restrictive)
        rsi_oversold = row['rsi'] < 25  # More restrictive
        ma_bullish = row['sma_20'] > row['sma_50']
        volume_ok = row['volume_ratio'] > 1.2  # Volume confirmation
        not_in_downtrend = data['close'].pct_change(20).iloc[i] > -0.15
        favorable_regime = market_regime in ['BULLISH', 'SIDEWAYS']
        
        # Much more restrictive buy condition
        buy_signal = (
            rsi_oversold and 
            ma_bullish and 
            volume_ok and 
            not_in_downtrend and 
            favorable_regime
        )
        
        # Improved sell conditions
        rsi_overbought = row['rsi'] > 75  # More restrictive
        ma_crossover_down = (row['sma_20'] < row['sma_50']) and (prev_row['sma_20'] >= prev_row['sma_50'])
        strong_downtrend = data['close'].pct_change(5).iloc[i] < -0.05
        
        sell_signal = rsi_overbought or ma_crossover_down or strong_downtrend
        
        # Risk-adjusted position sizing
        risk_multiplier = {'BULLISH': 1.0, 'SIDEWAYS': 0.8, 'BEARISH': 0.5}
        adjusted_risk = 0.015 * risk_multiplier.get(market_regime, 0.8)  # 1.5% base risk
        
        # Execute trades
        if buy_signal and len(positions) == 0:
            risk_amount = capital * adjusted_risk
            stop_distance = max(row['atr'] * 2, row['close'] * 0.03)  # 3% min stop
            position_size = int(risk_amount / stop_distance)
            
            # Limit position size
            max_position_value = capital * 0.15  # Max 15% per position
            max_shares = int(max_position_value / row['close'])
            position_size = min(position_size, max_shares)
            
            cost = position_size * row['close']
            if cost <= capital and position_size > 0:
                stop_loss = row['close'] - stop_distance
                take_profit = row['close'] + (row['atr'] * 3)
                
                positions['RELIANCE'] = {
                    'quantity': position_size,
                    'entry_price': row['close'],
                    'entry_date': row.name,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'highest_price': row['close']
                }
                capital -= cost
                trades.append(('BUY', row.name, row['close'], position_size, market_regime))
        
        elif len(positions) > 0 and 'RELIANCE' in positions:
            pos = positions['RELIANCE']
            
            # Update trailing stop
            if row['close'] > pos['highest_price']:
                pos['highest_price'] = row['close']
                # Update trailing stop (15% trailing)
                new_stop = max(pos['stop_loss'], row['close'] * 0.85)
                pos['stop_loss'] = new_stop
            
            # Check exit conditions
            should_sell = (
                sell_signal or 
                row['close'] <= pos['stop_loss'] or 
                row['close'] >= pos['take_profit']
            )
            
            if should_sell:
                proceeds = pos['quantity'] * row['close']
                pnl = proceeds - (pos['quantity'] * pos['entry_price'])
                capital += proceeds
                
                exit_reason = 'SIGNAL' if sell_signal else ('STOP' if row['close'] <= pos['stop_loss'] else 'PROFIT')
                trades.append(('SELL', row.name, row['close'], pos['quantity'], pnl, exit_reason))
                del positions['RELIANCE']
    
    # Calculate final performance
    final_value = capital
    if 'RELIANCE' in positions:
        final_value += positions['RELIANCE']['quantity'] * data['close'].iloc[-1]
    
    total_return = (final_value - 100000) / 100000
    num_trades = len([t for t in trades if t[0] == 'BUY'])
    
    print(f"  💰 Final Value: ₹{final_value:,.0f}")
    print(f"  📈 Total Return: {total_return:.2%}")
    print(f"  🔄 Number of Trades: {num_trades}")
    
    return {
        'final_value': final_value,
        'total_return': total_return,
        'num_trades': num_trades,
        'trades': trades
    }

def main():
    """Run the strategy comparison demonstration."""
    try:
        # Create sample data
        data = create_sample_trading_data()
        
        # Run both strategies
        original_results = simulate_original_strategy(data)
        improved_results = simulate_improved_strategy(data)
        
        # Show comparison
        print("\n" + "=" * 60)
        print("📈 STRATEGY PERFORMANCE COMPARISON")
        print("=" * 60)
        
        print(f"{'Metric':<25} {'Original':<15} {'Improved':<15} {'Change'}")
        print("-" * 65)
        
        # Return comparison
        orig_return = original_results['total_return']
        imp_return = improved_results['total_return']
        return_change = imp_return - orig_return
        
        print(f"{'Total Return':<25} {orig_return:>14.2%} {imp_return:>14.2%} {return_change:>+7.2%}")
        
        # Trade count comparison
        orig_trades = original_results['num_trades']
        imp_trades = improved_results['num_trades']
        trade_change = imp_trades - orig_trades
        
        print(f"{'Number of Trades':<25} {orig_trades:>14d} {imp_trades:>14d} {trade_change:>+7d}")
        
        # Final value comparison
        orig_value = original_results['final_value']
        imp_value = improved_results['final_value']
        value_change = imp_value - orig_value
        
        print(f"{'Final Portfolio Value':<25} ₹{orig_value:>13,.0f} ₹{imp_value:>13,.0f} ₹{value_change:>+6,.0f}")
        
        # Strategy recommendation
        print("\n" + "=" * 60)
        print("🎯 RECOMMENDATION")
        print("=" * 60)
        
        if return_change > 0.02:
            print("🏆 HIGHLY RECOMMENDED: Use Improved Strategy")
            print(f"   📈 {return_change:.2%} better returns")
            print(f"   🎯 {trade_change:+d} trade efficiency change")
        elif return_change > 0:
            print("✅ RECOMMENDED: Use Improved Strategy")
            print(f"   📈 {return_change:.2%} better returns")
        else:
            print("⚠️ Further optimization needed")
        
        print(f"\n💡 Key Improvements:")
        print(f"   • More selective entry conditions (RSI < 25 vs < 30)")
        print(f"   • Market regime filtering (avoid bear markets)")
        print(f"   • Volume confirmation (20% above average)")
        print(f"   • Trailing stop losses (protect profits)")
        print(f"   • Dynamic position sizing (adjust for volatility)")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def create_sample_trading_data():
    """Create sample data for strategy testing."""
    print("📊 Generating sample market data...")
    
    # Create 6 months of daily data
    dates = pd.date_range(start='2023-06-01', end='2023-12-01', freq='D')
    np.random.seed(42)  # For reproducible results
    
    # Base price for RELIANCE
    base_price = 2500
    
    # Generate realistic price movements
    data = []
    current_price = base_price
    
    for i, date in enumerate(dates):
        # Market cycles and trends
        trend = 0.0003 if i < 60 else (-0.0002 if i < 120 else 0.0001)
        volatility = 0.015 + 0.005 * np.sin(i / 20)  # Variable volatility
        daily_return = np.random.normal(trend, volatility)
        
        current_price *= (1 + daily_return)
        
        # Generate OHLC
        daily_range = current_price * np.random.uniform(0.01, 0.03)
        high = current_price + np.random.uniform(0, daily_range)
        low = current_price - np.random.uniform(0, daily_range)
        open_price = current_price + np.random.uniform(-daily_range/2, daily_range/2)
        
        # Generate volume
        base_volume = 1000000
        volume_multiplier = 1 + np.random.normal(0, 0.3)
        volume = int(base_volume * max(volume_multiplier, 0.3))
        
        data.append({
            'date': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': current_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    
    # Calculate technical indicators
    print("📊 Calculating technical indicators...")
    
    # Moving averages
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = abs(df['high'] - df['close'].shift(1))
    df['low_close'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['tr'].rolling(14).mean()
    
    # Volume ratio
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # Clean up temporary columns
    df.drop(['high_low', 'high_close', 'low_close', 'tr'], axis=1, inplace=True)
    
    print(f"✅ Generated {len(df)} days of data with technical indicators")
    return df

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 Strategy improvement demonstration completed!")
        print("\n💡 Next Steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Set up API keys in .env file")
        print("3. Run: python main.py compare-strategies")
        print("4. Implement improved strategy in live trading")
    else:
        print("\n❌ Demonstration failed. Check error messages above.")