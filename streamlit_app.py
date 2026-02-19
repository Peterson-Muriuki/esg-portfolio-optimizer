import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

# -------------------------------
# Add src to path safely
# -------------------------------
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# -------------------------------
# Safe imports
# -------------------------------
modules_loaded = True
try:
    from data_collection import DataCollector
    from sentiment_analysis import SentimentAnalyzer
    from ml_models import ReturnPredictor
    from portfolio_optimizer import PortfolioOptimizer
    import utils
except Exception as e:
    st.warning(f"⚠️ Some modules failed to load: {e}")
    modules_loaded = False

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="ESG Portfolio Optimizer", layout="wide", page_icon="📊")

# -------------------------------
# Preview Section (Always Runs)
# -------------------------------
st.markdown("## 🌱 Smart ESG Portfolio Optimizer")
st.info("👈 Configure your portfolio in the sidebar and click **Run Analysis** to begin")

sample_tickers = ['AAPL', 'MSFT', 'TSLA']
sample_data = pd.DataFrame({s: np.random.randn(100).cumsum() + 100 for s in sample_tickers})
fig = go.Figure()
for col in sample_data.columns:
    fig.add_trace(go.Scatter(x=sample_data.index, y=sample_data[col], name=col, mode='lines'))
fig.update_layout(title="Portfolio Performance (Sample)", xaxis_title="Days", yaxis_title="Value", height=300)
st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Sidebar Configuration
# -------------------------------
st.sidebar.header("⚙️ Configuration")
all_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NEE', 'JNJ', 'PG']
tickers = st.sidebar.multiselect("Select Stocks", options=all_tickers, default=all_tickers[:5])

run_analysis = st.sidebar.button("🚀 Run Analysis")

# -------------------------------
# Main Analysis
# -------------------------------
if run_analysis:
    st.info("Running analysis...")
    if not modules_loaded:
        st.warning("⚠️ Some advanced modules unavailable, using sample data only")
        prices = sample_data
        st.write(prices.head())
    else:
        try:
            collector = DataCollector(tickers)
            prices = collector.get_stock_data(datetime.now() - timedelta(days=730), datetime.now())
            st.success("Data collected successfully")
        except Exception as e:
            st.error(f"⚠️ Failed to collect market data: {e}")
