import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

class ReturnPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        
    def create_features(self, prices_df):
        """Create technical features for ML model"""
        print("\nCreating features...")
        returns = prices_df.pct_change()
        features = pd.DataFrame(index=prices_df.index)
        
        for ticker in prices_df.columns:
            # Historical returns
            features[f'{ticker}_return_1d'] = returns[ticker]
            features[f'{ticker}_return_5d'] = prices_df[ticker].pct_change(5)
            features[f'{ticker}_return_20d'] = prices_df[ticker].pct_change(20)
            
            # Volatility
            features[f'{ticker}_vol_20d'] = returns[ticker].rolling(20).std()
            
            # Moving average ratio
            ma_50 = prices_df[ticker].rolling(50).mean()
            features[f'{ticker}_ma_ratio'] = prices_df[ticker] / ma_50
            
            # RSI-like indicator
            delta = returns[ticker]
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss.replace(0, 1)
            features[f'{ticker}_rsi'] = 100 - (100 / (1 + rs))
        
        features = features.dropna()
        print(f"Created {features.shape[1]} features")
        return features
    
    def train_model(self, X, y, test_size=0.2):
        """Train Random Forest model"""
        if len(X) < 50:
            print("Warning: Limited data for training")
        
        print(f"\nTraining model with {len(X)} samples...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        self.feature_names = X.columns.tolist()
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        print(f"Train R²: {train_score:.4f}")
        print(f"Test R²: {test_score:.4f}")
        
        return train_score, test_score
    
    def predict_returns(self, X):
        """Predict expected returns"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return predictions
    
    def save_model(self, path='models/saved_models/'):
        """Save trained model"""
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.model, f'{path}rf_model.pkl')
        joblib.dump(self.scaler, f'{path}scaler.pkl')
        joblib.dump(self.feature_names, f'{path}feature_names.pkl')
        print(f"Model saved to {path}")
    
    def load_model(self, path='models/saved_models/'):
        """Load trained model"""
        self.model = joblib.load(f'{path}rf_model.pkl')
        self.scaler = joblib.load(f'{path}scaler.pkl')
        self.feature_names = joblib.load(f'{path}feature_names.pkl')
        print(f"Model loaded from {path}")


# Test function
if __name__ == "__main__":
    print("Testing ReturnPredictor...")
    
    # Sample price data
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    prices = pd.DataFrame({
        'AAPL': np.random.randn(len(dates)).cumsum() + 100,
        'MSFT': np.random.randn(len(dates)).cumsum() + 100
    }, index=dates)
    
    predictor = ReturnPredictor()
    features = predictor.create_features(prices)
    
    # Target: 5-day forward return
    returns = prices.pct_change(5).shift(-5)
    y = returns.loc[features.index].dropna()
    X = features.loc[y.index]
    
    # TRAIN each ticker separately
    for ticker in prices.columns:
        print(f"\n--- Training model for {ticker} ---")
        X_ticker = X[[col for col in X.columns if col.startswith(f'{ticker}_')]]
        y_ticker = y[ticker]
        predictor.train_model(X_ticker, y_ticker)
    
    print("Test complete")
