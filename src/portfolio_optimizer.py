import numpy as np
import pandas as pd
from scipy.optimize import minimize
import cvxpy as cp

class PortfolioOptimizer:
    def __init__(self, returns, expected_returns=None):
        """
        Initialize optimizer
        returns: DataFrame of historical returns
        expected_returns: Series of expected returns (optional)
        """
        self.returns = returns
        self.expected_returns = expected_returns if expected_returns is not None else returns.mean()
        self.cov_matrix = returns.cov()
        self.num_assets = len(returns.columns)
        self.tickers = returns.columns.tolist()
        
    def portfolio_stats(self, weights, risk_free_rate=0.02):
        """Calculate portfolio statistics"""
        port_return = np.dot(weights, self.expected_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0
        return port_return, port_vol, sharpe
    
    def markowitz_optimization(self, risk_free_rate=0.02):
        """Classic Markowitz mean-variance optimization (Max Sharpe)"""
        print("\nOptimizing portfolio (Max Sharpe Ratio)...")
        
        def neg_sharpe(weights):
            return -self.portfolio_stats(weights, risk_free_rate)[2]
        
        # Constraints
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        
        # Initial guess (equal weight)
        init_guess = np.array([1/self.num_assets] * self.num_assets)
        
        # Optimize
        result = minimize(
            neg_sharpe, 
            init_guess, 
            method='SLSQP',
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            print("Optimization successful")
        else:
            print("Optimization warning:", result.message)
        
        return result.x
    
    def minimum_variance(self):
        """Minimum variance portfolio"""
        print("\nOptimizing portfolio (Minimum Variance)...")
        
        try:
            w = cp.Variable(self.num_assets)
            risk = cp.quad_form(w, self.cov_matrix.values)
            
            constraints = [
                cp.sum(w) == 1,
                w >= 0
            ]
            
            prob = cp.Problem(cp.Minimize(risk), constraints)
            prob.solve()
            
            if prob.status == 'optimal':
                print("Optimization successful")
                return w.value
            else:
                print("Falling back to equal weights")
                return np.array([1/self.num_assets] * self.num_assets)
        except:
            print("Error in optimization, using equal weights")
            return np.array([1/self.num_assets] * self.num_assets)
    
    def risk_parity(self):
        """Risk parity optimization"""
        print("\nOptimizing portfolio (Risk Parity)...")
        
        def risk_contribution(weights):
            port_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            marginal_contrib = np.dot(self.cov_matrix, weights)
            contrib = weights * marginal_contrib / port_vol if port_vol > 0 else weights * 0
            return contrib
        
        def risk_parity_objective(weights):
            contrib = risk_contribution(weights)
            target = np.sum(contrib) / self.num_assets
            return np.sum((contrib - target)**2)
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.01, 1) for _ in range(self.num_assets))
        init_guess = np.array([1/self.num_assets] * self.num_assets)
        
        result = minimize(
            risk_parity_objective, 
            init_guess, 
            method='SLSQP',
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            print("Optimization successful")
        else:
            print("Optimization warning:", result.message)
        
        return result.x
    
    def black_litterman(self, market_caps=None):
        """Simplified Black-Litterman model"""
        print("\nOptimizing portfolio (Black-Litterman)...")
        
        if market_caps is None:
            # Equal market caps if not provided
            market_caps = pd.Series([1] * self.num_assets, index=self.tickers)
        
        # Market equilibrium returns (simplified)
        market_weights = market_caps / market_caps.sum()
        risk_aversion = 2.5
        pi = risk_aversion * np.dot(self.cov_matrix, market_weights)
        
        # Use equilibrium returns (no views for simplicity)
        print("Using market equilibrium weights")
        return market_weights.values
    
    def get_efficient_frontier(self, n_points=30):
        """Calculate efficient frontier"""
        print("\nCalculating efficient frontier...")
        
        target_returns = np.linspace(
            self.expected_returns.min(), 
            self.expected_returns.max(), 
            n_points
        )
        
        efficient_portfolios = []
        
        for i, target in enumerate(target_returns):
            try:
                w = cp.Variable(self.num_assets)
                risk = cp.quad_form(w, self.cov_matrix.values)
                ret = w @ self.expected_returns.values
                
                constraints = [
                    cp.sum(w) == 1,
                    w >= 0,
                    ret >= target
                ]
                
                prob = cp.Problem(cp.Minimize(risk), constraints)
                prob.solve(verbose=False)
                
                if prob.status == 'optimal' and w.value is not None:
                    port_return = np.dot(w.value, self.expected_returns)
                    port_vol = np.sqrt(np.dot(w.value.T, np.dot(self.cov_matrix, w.value)))
                    efficient_portfolios.append({
                        'return': port_return,
                        'volatility': port_vol,
                        'weights': w.value
                    })
            except:
                continue
        
        print(f"Calculated {len(efficient_portfolios)} frontier points")
        return pd.DataFrame(efficient_portfolios)
    
    def calculate_portfolio_metrics(self, weights, risk_free_rate=0.02):
        """Calculate comprehensive portfolio performance metrics"""
        port_return = np.dot(weights, self.expected_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0
        
        # Portfolio return series
        returns_series = self.returns.dot(weights)
        
        # Downside metrics
        downside_returns = returns_series[returns_series < 0]
        downside_vol = np.sqrt((downside_returns**2).mean()) if len(downside_returns) > 0 else port_vol
        sortino = (port_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
        
        # VaR and CVaR
        var_95 = np.percentile(returns_series.dropna(), 5)
        cvar_95 = returns_series[returns_series <= var_95].mean() if len(returns_series[returns_series <= var_95]) > 0 else var_95
        
        return {
            'Expected Return': port_return,
            'Volatility': port_vol,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'VaR (95%)': var_95,
            'CVaR (95%)': cvar_95,
            'Max Drawdown': self._calculate_max_drawdown(returns_series)
        }
    
    def _calculate_max_drawdown(self, returns_series):
        """Calculate maximum drawdown"""
        cumulative = (1 + returns_series).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

# Test function
if __name__ == "__main__":
    print("Testing PortfolioOptimizer...")
    
    # Create sample returns
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    returns = pd.DataFrame({
        'AAPL': np.random.randn(len(dates)) * 0.02,
        'MSFT': np.random.randn(len(dates)) * 0.02,
        'TSLA': np.random.randn(len(dates)) * 0.03
    }, index=dates)
    
    optimizer = PortfolioOptimizer(returns)
    
    # Test optimization methods
    weights_sharpe = optimizer.markowitz_optimization()
    print(f"\nMax Sharpe weights: {weights_sharpe}")
    
    weights_minvar = optimizer.minimum_variance()
    print(f"Min Variance weights: {weights_minvar}")
    
    # Test metrics
    metrics = optimizer.calculate_portfolio_metrics(weights_sharpe)
    print(f"\nPortfolio metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nTest complete")
