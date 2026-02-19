import pandas as pd
import numpy as np
from datetime import datetime

def validate_tickers(tickers):
    """Validate ticker symbols"""
    if not tickers:
        return []
    return [t.upper().strip() for t in tickers if t.strip()]

def format_percentage(value, decimals=2):
    """Format value as percentage"""
    return f"{value * 100:.{decimals}f}%"

def format_currency(value, decimals=2):
    """Format value as currency"""
    return f"${value:,.{decimals}f}"

def calculate_returns(prices, method='simple'):
    """Calculate returns from prices"""
    if method == 'simple':
        return prices.pct_change()
    elif method == 'log':
        return np.log(prices / prices.shift(1))
    else:
        raise ValueError("method must be 'simple' or 'log'")

def annualize_returns(returns, periods_per_year=252):
    """Annualize returns"""
    return returns * periods_per_year

def annualize_volatility(returns, periods_per_year=252):
    """Annualize volatility"""
    return returns.std() * np.sqrt(periods_per_year)

def save_results(data, filename, output_dir='data/processed'):
    """Save results to CSV"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    data.to_csv(filepath)
    print(f"✓ Saved to {filepath}")
    return filepath

def load_data(filename, data_dir='data/processed'):
    """Load data from CSV"""
    filepath = os.path.join(data_dir, filename)
    try:
        data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print(f"✓ Loaded from {filepath}")
        return data
    except FileNotFoundError:
        print(f"✗ File not found: {filepath}")
        return pd.DataFrame()

def print_portfolio_summary(weights, tickers):
    """Print formatted portfolio weights"""
    print("\n" + "="*50)
    print("PORTFOLIO ALLOCATION")
    print("="*50)
    for ticker, weight in zip(tickers, weights):
        bar = "█" * int(weight * 50)
        print(f"{ticker:6s} {format_percentage(weight, 1):>7s} {bar}")
    print("="*50)

# Test function
if __name__ == "__main__":
    print("Testing utils...")
    
    tickers = validate_tickers(['aapl', ' msft ', 'TSLA'])
    print(f"Validated tickers: {tickers}")
    
    print(f"Percentage: {format_percentage(0.1525)}")
    print(f"Currency: {format_currency(123456.789)}")
    
    print("\n✓ Test complete")