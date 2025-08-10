#!/usr/bin/env python3
"""
Test script to compare original and improved trading strategies.
Demonstrates the performance improvements and provides detailed analysis.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    from src.strategies.rsi_ma_strategy import RSIMACrossoverStrategy
    from src.strategies.improved_rsi_ma_strategy import ImprovedRSIMACrossoverStrategy
    from src.data.data_fetcher import DataFetcher
    from src.data.technical_indicators import TechnicalIndicators
    from src.config import Config
    from loguru import logger
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)


def generate_sample_data(symbol: str, days: int = 180) -> pd.DataFrame:
    """
    Generate realistic sample stock data for testing when API is not available.
    
    Args:
        symbol: Stock symbol
        days: Number of days of data to generate
        
    Returns:
        DataFrame with OHLCV data
    """
    logger.info(f"Generating sample data for {symbol} ({days} days)")
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Base parameters for realistic stock data
    base_price = 2500 if 'RELIANCE' in symbol else (3500 if 'TCS' in symbol else 1500)
    volatility = 0.02  # 2% daily volatility
    
    # Generate price data with realistic patterns
    np.random.seed(42 + hash(symbol) % 1000)  # Consistent but different for each symbol
    
    # Generate returns with some trend and mean reversion
    returns = np.random.normal(0.0005, volatility, len(dates))  # Slight positive drift
    
    # Add some realistic patterns
    for i in range(1, len(returns)):
        # Add some momentum and mean reversion
        momentum = returns[i-1] * 0.1  # 10% momentum
        mean_reversion = -returns[max(0, i-5):i].mean() * 0.05  # 5% mean reversion
        returns[i] += momentum + mean_reversion
    
    # Calculate prices
    prices = base_price * np.cumprod(1 + returns)
    
    # Generate OHLC from close prices
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Generate realistic OHLC
        daily_range = close * np.random.uniform(0.01, 0.04)  # 1-4% daily range
        high = close + np.random.uniform(0, daily_range)
        low = close - np.random.uniform(0, daily_range)
        open_price = close + np.random.uniform(-daily_range/2, daily_range/2)
        
        # Generate volume
        base_volume = 1000000
        volume = int(base_volume * np.random.uniform(0.5, 2.0))
        
        data.append({
            'date': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    
    # Add technical indicators
    df = TechnicalIndicators.calculate_all_indicators(df)
    
    logger.success(f"Generated {len(df)} days of sample data for {symbol}")
    return df


def run_strategy_comparison():
    """Run comparison between original and improved strategies."""
    logger.info("Starting strategy comparison test...")
    
    # Generate sample data for testing
    symbols = ['RELIANCE.BSE', 'TCS.BSE', 'INFY.BSE']
    data_dict = {}
    
    for symbol in symbols:
        try:
            # Try to use real data first
            data_fetcher = DataFetcher()
            data = data_fetcher.get_daily_data(symbol)
            if data.empty:
                raise Exception("No data from API")
            logger.info(f"Using real data for {symbol}")
        except Exception as e:
            logger.warning(f"Using sample data for {symbol}: {e}")
            data = generate_sample_data(symbol, 180)
        
        data_dict[symbol] = data
    
    # Test period (last 6 months)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    
    print("\n" + "="*60)
    print("🚀 ALGO-TRADING STRATEGY COMPARISON")
    print("="*60)
    
    # Run original strategy
    print("\n📊 Testing Original Strategy...")
    original_strategy = RSIMACrossoverStrategy(initial_capital=100000, risk_per_trade=0.02)
    original_results = original_strategy.backtest(data_dict, start_date, end_date)
    
    # Run improved strategy
    print("\n🎯 Testing Improved Strategy...")
    improved_strategy = ImprovedRSIMACrossoverStrategy(initial_capital=100000, risk_per_trade=0.015)
    improved_results = improved_strategy.backtest(data_dict, start_date, end_date)
    
    # Display comparison
    print("\n" + "="*60)
    print("📈 STRATEGY PERFORMANCE COMPARISON")
    print("="*60)
    
    metrics = [
        ('Total Return', 'total_return', '%'),
        ('Annualized Return', 'annualized_return', '%'),
        ('Sharpe Ratio', 'sharpe_ratio', ''),
        ('Max Drawdown', 'max_drawdown', '%'),
        ('Win Rate', 'win_rate', '%'),
        ('Total Trades', 'total_trades', ''),
        ('Avg Win', 'avg_win', '₹'),
        ('Avg Loss', 'avg_loss', '₹'),
        ('Profit Factor', 'profit_factor', '')
    ]
    
    print(f"{'Metric':<20} {'Original':<15} {'Improved':<15} {'Change':<15}")
    print("-" * 65)
    
    for metric_name, metric_key, unit in metrics:
        orig_val = original_results.get(metric_key, 0)
        imp_val = improved_results.get(metric_key, 0)
        
        if unit == '%':
            orig_str = f"{orig_val:.2%}"
            imp_str = f"{imp_val:.2%}"
            change = imp_val - orig_val
            change_str = f"{change:+.2%}"
        elif unit == '₹':
            orig_str = f"₹{orig_val:,.0f}"
            imp_str = f"₹{imp_val:,.0f}"
            change = imp_val - orig_val
            change_str = f"₹{change:+,.0f}"
        else:
            orig_str = f"{orig_val:.2f}"
            imp_str = f"{imp_val:.2f}"
            change = imp_val - orig_val
            change_str = f"{change:+.2f}"
        
        # Color coding for improvements
        if metric_key in ['total_return', 'annualized_return', 'sharpe_ratio', 'win_rate', 'profit_factor']:
            indicator = "🟢" if change > 0 else "🔴" if change < 0 else "🟡"
        elif metric_key in ['max_drawdown']:
            indicator = "🟢" if change < 0 else "🔴" if change > 0 else "🟡"  # Negative change is good for drawdown
        else:
            indicator = "🟡"
        
        print(f"{metric_name:<20} {orig_str:<15} {imp_str:<15} {change_str:<15} {indicator}")
    
    # Summary recommendation
    print("\n" + "="*60)
    print("🎯 STRATEGY RECOMMENDATION")
    print("="*60)
    
    return_improvement = improved_results['total_return'] - original_results['total_return']
    sharpe_improvement = improved_results['sharpe_ratio'] - original_results['sharpe_ratio']
    drawdown_improvement = original_results['max_drawdown'] - improved_results['max_drawdown']
    
    if return_improvement > 0.02 and sharpe_improvement > 0.1:
        recommendation = "🏆 HIGHLY RECOMMENDED: Use Improved Strategy"
        reason = "Significant improvements in both returns and risk-adjusted performance"
    elif return_improvement > 0 and drawdown_improvement > 0:
        recommendation = "✅ RECOMMENDED: Use Improved Strategy"
        reason = "Better returns with reduced risk"
    elif sharpe_improvement > 0:
        recommendation = "✅ RECOMMENDED: Use Improved Strategy"
        reason = "Better risk-adjusted returns"
    else:
        recommendation = "⚠️ FURTHER OPTIMIZATION NEEDED"
        reason = "Strategy improvements are marginal"
    
    print(f"\n{recommendation}")
    print(f"Reason: {reason}")
    
    # Key improvements summary
    print(f"\n📊 Key Improvements:")
    print(f"• Return Improvement: {return_improvement:+.2%}")
    print(f"• Risk Reduction: {drawdown_improvement:+.2%} drawdown improvement")
    print(f"• Sharpe Improvement: {sharpe_improvement:+.2f}")
    print(f"• Trade Efficiency: {improved_results['total_trades'] - original_results['total_trades']:+d} trades")
    
    return {
        'original': original_results,
        'improved': improved_results,
        'recommendation': recommendation
    }


def main():
    """Main function to run strategy testing."""
    try:
        results = run_strategy_comparison()
        
        print("\n" + "="*60)
        print("✅ STRATEGY COMPARISON COMPLETED SUCCESSFULLY")
        print("="*60)
        
        # Save results for further analysis
        if results['improved']['total_return'] > results['original']['total_return']:
            print("\n🎉 The improved strategy shows better performance!")
            print("💡 Consider implementing the enhanced version for live trading.")
        
    except Exception as e:
        logger.error(f"Strategy comparison failed: {e}")
        print(f"\n❌ Test failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    main()