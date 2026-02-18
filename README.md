# 🌱 Smart ESG Portfolio Optimizer

An end-to-end portfolio optimization system combining alternative data (ESG news sentiment), machine learning predictions, and modern portfolio theory.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Features

### 📰 Alternative Data
- Scrapes ESG-related news from Google News RSS feeds
- NLP sentiment analysis using TextBlob
- TF-IDF keyword scoring for Environmental, Social, and Governance topics
- Aggregate sentiment metrics by ticker

### 🤖 Machine Learning
- Random Forest models for return prediction
- Technical indicators (momentum, volatility, RSI, MA ratios)
- Feature engineering from price data
- Cross-validated forecasts

### 💼 Portfolio Optimization
- **Markowitz Mean-Variance** (Maximum Sharpe Ratio)
- **Minimum Variance** Portfolio
- **Risk Parity** Allocation
- **Black-Litterman** Model
- Efficient Frontier visualization

### 📊 Performance Analytics
- Sharpe and Sortino ratios
- Value at Risk (VaR) and Conditional VaR
- Maximum drawdown analysis
- Historical backtesting
- Comparison vs. equal-weight benchmark

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Windows 10/11
- Git (optional, for version control)

### Installation

1. **Clone or download this repository**
```cmd
cd %USERPROFILE%\Documents
git clone https://github.com/yourusername/esg-portfolio-optimizer.git
cd esg-portfolio-optimizer
```

2. **Create and activate virtual environment**
```cmd
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies**
```cmd
pip install -r requirements.txt
```

4. **Run the app**
```cmd
streamlit run streamlit_app.py
```

5. **Open your browser**
- The app will automatically open at `http://localhost:8501`
- If not, manually navigate to that URL

## 📖 Usage

1. **Select Assets**: Choose 5-15 stocks from the sidebar
2. **Set Date Range**: Define your historical analysis period
3. **Choose Optimization Method**: Select from multiple strategies
4. **Configure Settings**: Adjust risk-free rate and advanced options
5. **Run Analysis**: Click the "🚀 Run Analysis" button
6. **Explore Results**: Navigate through the 5 result tabs

## 📁 Project Structure
```
esg-portfolio-optimizer/
│
├── data/
│   ├── raw/                    # Raw scraped data
│   └── processed/              # Cleaned data
│
├── src/
│   ├── data_collection.py      # Data scraping & collection
│   ├── sentiment_analysis.py   # NLP sentiment analysis
│   ├── ml_models.py           # ML prediction models
│   ├── portfolio_optimizer.py  # Optimization algorithms
│   └── utils.py               # Helper functions
│
├── models/
│   └── saved_models/          # Trained ML models
│
├── .streamlit/
│   └── config.toml            # Streamlit configuration
│
├── streamlit_app.py           # Main dashboard application
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## 🛠️ Technologies

- **Python 3.8+**: Core programming language
- **Streamlit**: Interactive web dashboard
- **scikit-learn**: Machine learning models
- **CVXPY**: Convex optimization
- **yfinance**: Market data retrieval
- **Plotly**: Interactive visualizations
- **Pandas & NumPy**: Data manipulation
- **TextBlob**: NLP sentiment analysis

## 📊 Sample Output

The app generates:
- Interactive price charts
- Correlation heatmaps
- Sentiment analysis visualizations
- ML prediction comparisons
- Efficient frontier plots
- Portfolio allocation pie charts
- Backtest performance graphs
- Downloadable CSV/TXT reports

## 🔧 Customization

### Adding More Tickers
Edit `streamlit_app.py` line ~90:
```python
all_tickers = ['AAPL', 'MSFT', ..., 'YOUR_TICKER']
```

### Changing ML Model
Edit `src/ml_models.py` and replace `RandomForestRegressor` with your preferred model.

### Adding New Optimization Methods
Extend `src/portfolio_optimizer.py` with new optimization functions.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

MIT License - see LICENSE file for details

## 🐛 Known Issues

- News scraping may be rate-limited by Google
- ML predictions require sufficient historical data (100+ days)
- Some optimization methods may fail with highly correlated assets

## 🔮 Future Enhancements

- [ ] Real-time data updates
- [ ] Additional ML models (LSTM, XGBoost)
- [ ] Advanced backtesting engine
- [ ] REST API deployment
- [ ] Database integration (PostgreSQL)
- [ ] Multi-period rebalancing simulation
- [ ] Factor model integration

## 📧 Contact

For questions or feedback:
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

## 🙏 Acknowledgments

- Data provided by Yahoo Finance and Google News
- Built with Streamlit's amazing framework
- Inspired by modern portfolio theory and quantitative finance research

---

**⭐ If you find this project useful, please give it a star on GitHub!**"# esg-portfolio-optimizer" 
