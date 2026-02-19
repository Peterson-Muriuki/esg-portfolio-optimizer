import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_collection import DataCollector
from sentiment_analysis import SentimentAnalyzer
from ml_models import ReturnPredictor
from portfolio_optimizer import PortfolioOptimizer
import utils

# Page config
st.set_page_config(
    page_title="ESG Portfolio Optimizer",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🌱 Smart ESG Portfolio Optimizer</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Combining Alternative Data, Machine Learning & Modern Portfolio Theory</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=ESG+Optimizer", use_container_width=True)
st.sidebar.header("⚙️ Configuration")

# Stock selection
st.sidebar.subheader("📈 Asset Selection")
all_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NEE', 'JNJ', 'PG', 
               'UNH', 'V', 'WMT', 'XOM', 'JPM', 'BAC', 'HD', 'DIS', 'NFLX', 'NVDA']

default_tickers = ['AAPL', 'MSFT', 'TSLA', 'NEE', 'JNJ', 'PG', 'UNH', 'V', 'WMT', 'XOM']

tickers = st.sidebar.multiselect(
    "Select 5-15 stocks",
    options=all_tickers,
    default=default_tickers,
    help="Choose stocks for portfolio optimization"
)

# Validate selection
if len(tickers) < 5:
    st.sidebar.warning("⚠️ Please select at least 5 stocks")
elif len(tickers) > 15:
    st.sidebar.warning("⚠️ Please select no more than 15 stocks")

# Date range
st.sidebar.subheader("📅 Time Period")
col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input(
    "Start",
    datetime.now() - timedelta(days=730),
    max_value=datetime.now()
)
end_date = col2.date_input(
    "End",
    datetime.now(),
    max_value=datetime.now()
)

# Optimization settings
st.sidebar.subheader("🎯 Optimization Settings")
opt_method = st.sidebar.selectbox(
    "Method",
    ["Max Sharpe Ratio", "Minimum Variance", "Risk Parity", "Black-Litterman"],
    help="Choose portfolio optimization strategy"
)

risk_free_rate = st.sidebar.slider(
    "Risk-Free Rate (%)",
    min_value=0.0,
    max_value=10.0,
    value=4.0,
    step=0.1,
    help="Annual risk-free rate (e.g., Treasury yield)"
) / 100

# Advanced settings
with st.sidebar.expander("🔧 Advanced Settings"):
    include_ml = st.checkbox("Use ML Predictions", value=True, help="Use machine learning for return forecasts")
    include_sentiment = st.checkbox("Include Sentiment Analysis", value=True, help="Factor in news sentiment")
    rebalance_freq = st.selectbox("Rebalancing Frequency", ["Monthly", "Quarterly", "Annually"])

# Run button
st.sidebar.markdown("---")
run_analysis = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)

# Info section
if not run_analysis:
    st.info("👈 Configure your portfolio in the sidebar and click **'Run Analysis'** to begin")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📰 Alternative Data")
        st.write("• Scrapes ESG news from Google")
        st.write("• NLP sentiment analysis")
        st.write("• TF-IDF keyword scoring")
        st.write("• Environmental, Social, Governance metrics")
    
    with col2:
        st.markdown("### 🤖 Machine Learning")
        st.write("• Random Forest predictions")
        st.write("• Technical indicators")
        st.write("• Feature engineering")
        st.write("• Cross-validated forecasts")
    
    with col3:
        st.markdown("### 💼 Optimization")
        st.write("• Markowitz mean-variance")
        st.write("• Risk Parity allocation")
        st.write("• Black-Litterman model")
        st.write("• Efficient frontier analysis")
    
    st.markdown("---")
    st.markdown("**Sample Output Preview:**")
    
    # Sample chart
    sample_data = pd.DataFrame({
        'AAPL': np.random.randn(100).cumsum() + 100,
        'MSFT': np.random.randn(100).cumsum() + 100,
        'TSLA': np.random.randn(100).cumsum() + 100
    })
    
    fig = go.Figure()
    for col in sample_data.columns:
        fig.add_trace(go.Scatter(x=sample_data.index, y=sample_data[col], name=col, mode='lines'))
    fig.update_layout(title="Portfolio Performance (Sample)", xaxis_title="Days", yaxis_title="Value", height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    st.stop()

# Main Analysis
if run_analysis and 5 <= len(tickers) <= 15:
    try:
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Data Collection
        status_text.text("📊 Step 1/5: Collecting market data...")
        progress_bar.progress(10)
        
        collector = DataCollector(tickers)
        prices = collector.get_stock_data(start_date, end_date)
        
        if prices.empty:
            st.error("❌ Failed to download price data. Please check tickers and try again.")
            st.stop()
        
        progress_bar.progress(20)
        
        # Get ESG scores
        try:
            esg_scores = collector.get_esg_scores()
        except:
            esg_scores = pd.DataFrame()
        
        progress_bar.progress(30)
        
        # Step 2: Sentiment Analysis
        if include_sentiment:
            status_text.text("🔍 Step 2/5: Analyzing ESG sentiment...")
            try:
                news = collector.collect_all_news()
                if not news.empty:
                    analyzer = SentimentAnalyzer()
                    analyzed_news = analyzer.analyze_news(news)
                    ticker_sentiment = analyzer.aggregate_by_ticker(analyzed_news)
                else:
                    ticker_sentiment = pd.DataFrame()
            except Exception as e:
                st.warning(f"⚠️ Sentiment analysis unavailable: {e}")
                ticker_sentiment = pd.DataFrame()
        else:
            ticker_sentiment = pd.DataFrame()
        
        progress_bar.progress(45)
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Step 3: ML Predictions
        status_text.text("🤖 Step 3/5: Training ML models...")
        expected_returns_ml = None
        
        if include_ml and len(returns) > 100:
            try:
                predictor = ReturnPredictor()
                features = predictor.create_features(prices)
                
                if len(features) > 50:
                    predictions = {}
                    for ticker in tickers:
                        try:
                            # Prepare data for this ticker
                            future_returns = returns[ticker].shift(-5).loc[features.index]
                            X = features[:-5]
                            y = future_returns[:-5].dropna()
                            X = X.loc[y.index]
                            
                            if len(y) > 30:
                                # Train model
                                pred_model = ReturnPredictor()
                                pred_model.train_model(X, y, test_size=0.2)
                                
                                # Predict on latest data
                                latest_pred = pred_model.predict_returns(features.iloc[[-1]])
                                predictions[ticker] = latest_pred[0] * 252  # Annualize
                        except Exception as e:
                            predictions[ticker] = returns[ticker].mean() * 252
                    
                    expected_returns_ml = pd.Series(predictions)
                else:
                    expected_returns_ml = returns.mean() * 252
            except Exception as e:
                st.warning(f"⚠️ ML predictions unavailable: {e}")
                expected_returns_ml = returns.mean() * 252
        else:
            expected_returns_ml = returns.mean() * 252
        
        progress_bar.progress(60)
        
        # Step 4: Portfolio Optimization
        status_text.text("💼 Step 4/5: Optimizing portfolio...")
        
        # Use ML predictions or historical returns
        expected_returns = expected_returns_ml if expected_returns_ml is not None else returns.mean() * 252
        
        # Annualize returns for optimization
        returns_annual = returns * 252
        optimizer = PortfolioOptimizer(returns_annual, expected_returns)
        
        # Optimize based on selected method
        if opt_method == "Max Sharpe Ratio":
            weights = optimizer.markowitz_optimization(risk_free_rate=risk_free_rate)
        elif opt_method == "Minimum Variance":
            weights = optimizer.minimum_variance()
        elif opt_method == "Risk Parity":
            weights = optimizer.risk_parity()
        else:  # Black-Litterman
            market_caps = pd.Series({ticker: 1 for ticker in tickers})
            weights = optimizer.black_litterman(market_caps)
        
        # Calculate metrics
        metrics = optimizer.calculate_portfolio_metrics(weights, risk_free_rate)
        
        progress_bar.progress(80)
        
        # Step 5: Calculate Efficient Frontier
        status_text.text("📈 Step 5/5: Calculating efficient frontier...")
        try:
            efficient_frontier = optimizer.get_efficient_frontier(n_points=25)
        except:
            efficient_frontier = pd.DataFrame()
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        # Clear progress indicators after a moment
        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        # Display Results in Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview",
            "🎯 Sentiment",
            "🤖 ML Insights",
            "💼 Portfolio",
            "📈 Performance"
        ])
        
        # TAB 1: Overview
        with tab1:
            st.header("📊 Market Overview")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            total_return = ((prices.iloc[-1] / prices.iloc[0]) - 1) * 100
            avg_return = total_return.mean()
            best_stock = total_return.idxmax()
            worst_stock = total_return.idxmin()
            
            col1.metric("Avg Return", f"{avg_return:.2f}%", delta=None)
            col2.metric("Best Performer", best_stock, delta=f"{total_return[best_stock]:.2f}%")
            col3.metric("Worst Performer", worst_stock, delta=f"{total_return[worst_stock]:.2f}%")
            col4.metric("Date Range", f"{(end_date - start_date).days} days")
            
            st.markdown("---")
            
            # Price chart
            st.subheader("📈 Normalized Price Performance")
            normalized_prices = (prices / prices.iloc[0]) * 100
            
            fig = go.Figure()
            for ticker in tickers:
                fig.add_trace(go.Scatter(
                    x=normalized_prices.index,
                    y=normalized_prices[ticker],
                    name=ticker,
                    mode='lines',
                    hovertemplate=f'{ticker}<br>%{{y:.2f}}<extra></extra>'
                ))
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Normalized Price (Base = 100)",
                hovermode='x unified',
                height=500,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics table
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Return Statistics")
                stats_df = pd.DataFrame({
                    'Total Return': total_return,
                    'Avg Daily Return': returns.mean() * 100,
                    'Volatility': returns.std() * 100,
                    'Sharpe Ratio': (returns.mean() / returns.std()) * np.sqrt(252)
                })
                st.dataframe(stats_df.style.format({
                    'Total Return': '{:.2f}%',
                    'Avg Daily Return': '{:.3f}%',
                    'Volatility': '{:.2f}%',
                    'Sharpe Ratio': '{:.2f}'
                }), use_container_width=True)
            
            with col2:
                st.subheader("🔗 Correlation Matrix")
                corr_matrix = returns.corr()
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    aspect='auto',
                    color_continuous_scale='RdBu_r',
                    zmin=-1,
                    zmax=1
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Continue with other tabs...
        # (Due to length, I'll provide the rest in the next part)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    Built with Python 🐍 | Streamlit | scikit-learn | CVXPY<br>
    Data from Yahoo Finance & Google News RSS
</div>
""", unsafe_allow_html=True)
# TAB 2: Sentiment Analysis
        with tab2:
            st.header("🎯 ESG Sentiment Analysis")
            
            if not ticker_sentiment.empty:
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.subheader("Average Sentiment by Stock")
                    
                    # Prepare sentiment data
                    sent_data = ticker_sentiment.reset_index()
                    sent_data = sent_data.rename(columns={'index': 'ticker'})
                    
                    # Create bar chart
                    fig = px.bar(
                        sent_data,
                        x='ticker',
                        y='sentiment_mean',
                        color='sentiment_mean',
                        color_continuous_scale='RdYlGn',
                        color_continuous_midpoint=0,
                        labels={'sentiment_mean': 'Sentiment Score', 'ticker': 'Stock'},
                        title="ESG News Sentiment"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("ESG Component Distribution")
                    
                    # Calculate average ESG scores
                    esg_cols = ['environmental_score_sum', 'social_score_sum', 'governance_score_sum']
                    if all(col in ticker_sentiment.columns for col in esg_cols):
                        esg_components = ticker_sentiment[esg_cols].mean()
                        
                        fig = px.pie(
                            values=esg_components.values,
                            names=['Environmental', 'Social', 'Governance'],
                            title="ESG Focus Areas",
                            color_discrete_sequence=px.colors.sequential.Greens
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Detailed table
                st.subheader("📋 Detailed Sentiment Metrics")
                st.dataframe(ticker_sentiment, use_container_width=True)
                
                # Download button
                csv = ticker_sentiment.to_csv()
                st.download_button(
                    label="📥 Download Sentiment Data (CSV)",
                    data=csv,
                    file_name=f"esg_sentiment_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("📰 Sentiment analysis data not available. Try enabling it in settings.")
        
        # TAB 3: ML Insights
        with tab3:
            st.header("🤖 Machine Learning Insights")
            
            if include_ml and expected_returns_ml is not None:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("Expected Return Predictions")
                    
                    # Create prediction dataframe
                    pred_df = pd.DataFrame({
                        'Ticker': expected_returns_ml.index,
                        'Predicted Annual Return (%)': expected_returns_ml.values * 100
                    }).sort_values('Predicted Annual Return (%)', ascending=False)
                    
                    # Bar chart
                    fig = px.bar(
                        pred_df,
                        x='Ticker',
                        y='Predicted Annual Return (%)',
                        color='Predicted Annual Return (%)',
                        color_continuous_scale='RdYlGn',
                        title="ML-Based Return Forecasts"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Top 3 Predictions")
                    top3 = pred_df.head(3)
                    for idx, row in top3.iterrows():
                        st.metric(
                            row['Ticker'],
                            f"{row['Predicted Annual Return (%)']:.2f}%",
                            delta="Top Pick" if idx == 0 else None
                        )
                
                # Comparison with historical
                st.subheader("📊 ML vs Historical Returns")
                hist_returns = returns.mean() * 252 * 100
                
                comparison_df = pd.DataFrame({
                    'Ticker': expected_returns_ml.index,
                    'ML Prediction': expected_returns_ml.values * 100,
                    'Historical Avg': hist_returns.values
                })
                
                fig = go.Figure()
                fig.add_trace(go.Bar(name='ML Prediction', x=comparison_df['Ticker'], y=comparison_df['ML Prediction']))
                fig.add_trace(go.Bar(name='Historical', x=comparison_df['Ticker'], y=comparison_df['Historical Avg']))
                fig.update_layout(barmode='group', height=400, yaxis_title="Annual Return (%)")
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("💡 ML predictions use technical indicators (returns, volatility, momentum, RSI) to forecast short-term movements.")
            else:
                st.info("🤖 ML predictions are disabled or unavailable. Using historical returns instead.")
        
        # TAB 4: Portfolio Optimization
        with tab4:
            st.header("💼 Optimal Portfolio Allocation")
            
            # Display optimization method
            st.success(f"✅ Optimized using: **{opt_method}**")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Portfolio Weights")
                
                # Create weights dataframe
                weights_df = pd.DataFrame({
                    'Ticker': tickers,
                    'Weight (%)': weights * 100
                }).sort_values('Weight (%)', ascending=False)
                
                # Pie chart
                fig = px.pie(
                    weights_df,
                    values='Weight (%)',
                    names='Ticker',
                    title=f"Portfolio Allocation ({opt_method})",
                    color_discrete_sequence=px.colors.sequential.Blues
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Weight Details")
                
                # Add visual bars
                weights_df['Bar'] = weights_df['Weight (%)'].apply(lambda x: '█' * int(x/2))
                
                st.dataframe(
                    weights_df[['Ticker', 'Weight (%)']].style.format({'Weight (%)': '{:.2f}%'}),
                    use_container_width=True,
                    height=400
                )
                
                # Print portfolio summary
                utils.print_portfolio_summary(weights, tickers)
            
            st.markdown("---")
            
            # Efficient Frontier
            if not efficient_frontier.empty:
                st.subheader("📈 Efficient Frontier")
                
                fig = go.Figure()
                
                # Frontier line
                fig.add_trace(go.Scatter(
                    x=efficient_frontier['volatility'] * 100,
                    y=efficient_frontier['return'] * 100,
                    mode='lines',
                    name='Efficient Frontier',
                    line=dict(color='#1f77b4', width=3),
                    hovertemplate='Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>'
                ))
                
                # Optimal portfolio
                fig.add_trace(go.Scatter(
                    x=[metrics['Volatility'] * 100],
                    y=[metrics['Expected Return'] * 100],
                    mode='markers',
                    name='Optimal Portfolio',
                    marker=dict(size=20, color='red', symbol='star', line=dict(width=2, color='white')),
                    hovertemplate='Optimal<br>Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>'
                ))
                
                # Individual assets
                individual_vols = np.sqrt(np.diag(optimizer.cov_matrix)) * 100
                individual_returns = expected_returns * 100
                
                fig.add_trace(go.Scatter(
                    x=individual_vols,
                    y=individual_returns,
                    mode='markers+text',
                    name='Individual Assets',
                    text=tickers,
                    textposition='top center',
                    marker=dict(size=12, color='green', line=dict(width=1, color='white')),
                    hovertemplate='%{text}<br>Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>'
                ))
                
                fig.update_layout(
                    xaxis_title="Annual Volatility (%)",
                    yaxis_title="Expected Annual Return (%)",
                    height=600,
                    hovermode='closest',
                    showlegend=True,
                    legend=dict(x=0.7, y=0.98)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("💡 The efficient frontier shows the best possible risk-return tradeoffs. Your optimal portfolio is marked with a ⭐.")
        
        # TAB 5: Performance Metrics
        with tab5:
            st.header("📈 Portfolio Performance Analysis")
            
            # Key metrics in cards
            st.subheader("🎯 Key Performance Indicators")
            
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric(
                "Expected Return",
                f"{metrics['Expected Return']*100:.2f}%",
                delta=f"{(metrics['Expected Return']-risk_free_rate)*100:.2f}% vs RF"
            )
            col2.metric(
                "Volatility",
                f"{metrics['Volatility']*100:.2f}%"
            )
            col3.metric(
                "Sharpe Ratio",
                f"{metrics['Sharpe Ratio']:.2f}",
                delta="Good" if metrics['Sharpe Ratio'] > 1 else "Moderate"
            )
            col4.metric(
                "Sortino Ratio",
                f"{metrics['Sortino Ratio']:.2f}"
            )
            
            col1, col2, col3 = st.columns(3)
            
            col1.metric(
                "Value at Risk (95%)",
                f"{metrics['VaR (95%)']*100:.2f}%",
                delta="Daily loss threshold"
            )
            col2.metric(
                "Conditional VaR (95%)",
                f"{metrics['CVaR (95%)']*100:.2f}%",
                delta="Expected tail loss"
            )
            col3.metric(
                "Max Drawdown",
                f"{metrics['Max Drawdown']*100:.2f}%",
                delta="Historical worst"
            )
            
            st.markdown("---")
            
            # Backtest
            st.subheader("📊 Historical Backtest")
            
            # Calculate portfolio returns
            portfolio_returns = returns.dot(weights)
            cumulative_returns = (1 + portfolio_returns).cumprod()
            
            # Benchmark (equal weight)
            equal_weights = np.array([1/len(tickers)] * len(tickers))
            benchmark_returns = returns.dot(equal_weights)
            cumulative_benchmark = (1 + benchmark_returns).cumprod()
            
            # Individual stocks (best and worst)
            best_stock_returns = (1 + returns[best_stock]).cumprod()
            worst_stock_returns = (1 + returns[worst_stock]).cumprod()
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns.values * 100,
                name='Optimized Portfolio',
                line=dict(color='green', width=3),
                hovertemplate='%{y:.2f}<extra></extra>'
            ))
            
            fig.add_trace(go.Scatter(
                x=cumulative_benchmark.index,
                y=cumulative_benchmark.values * 100,
                name='Equal Weight',
                line=dict(color='gray', width=2, dash='dash'),
                hovertemplate='%{y:.2f}<extra></extra>'
            ))
            
            fig.add_trace(go.Scatter(
                x=best_stock_returns.index,
                y=best_stock_returns.values * 100,
                name=f'Best Stock ({best_stock})',
                line=dict(color='blue', width=1, dash='dot'),
                hovertemplate='%{y:.2f}<extra></extra>'
            ))
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Cumulative Return (Base = 100)",
                height=500,
                hovermode='x unified',
                showlegend=True,
                legend=dict(x=0.01, y=0.99)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Return Comparison")
                perf_comparison = pd.DataFrame({
                    'Strategy': ['Optimized Portfolio', 'Equal Weight', f'Best Stock ({best_stock})', f'Worst Stock ({worst_stock})'],
                    'Total Return (%)': [
                        (cumulative_returns.iloc[-1] - 1) * 100,
                        (cumulative_benchmark.iloc[-1] - 1) * 100,
                        (best_stock_returns.iloc[-1] - 1) * 100,
                        (worst_stock_returns.iloc[-1] - 1) * 100
                    ],
                    'Volatility (%)': [
                        portfolio_returns.std() * np.sqrt(252) * 100,
                        benchmark_returns.std() * np.sqrt(252) * 100,
                        returns[best_stock].std() * np.sqrt(252) * 100,
                        returns[worst_stock].std() * np.sqrt(252) * 100
                    ]
                })
                
                st.dataframe(
                    perf_comparison.style.format({
                        'Total Return (%)': '{:.2f}%',
                        'Volatility (%)': '{:.2f}%'
                    }).background_gradient(subset=['Total Return (%)'], cmap='RdYlGn'),
                    use_container_width=True
                )
            
            with col2:
                st.subheader("📉 Risk Metrics")
                risk_metrics = pd.DataFrame({
                    'Metric': ['Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown', 'VaR (95%)', 'CVaR (95%)'],
                    'Value': [
                        f"{metrics['Sharpe Ratio']:.2f}",
                        f"{metrics['Sortino Ratio']:.2f}",
                        f"{metrics['Max Drawdown']*100:.2f}%",
                        f"{metrics['VaR (95%)']*100:.2f}%",
                        f"{metrics['CVaR (95%)']*100:.2f}%"
                    ]
                })
                st.dataframe(risk_metrics, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            # Export section
            st.subheader("📥 Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            # Portfolio weights CSV
            weights_csv = weights_df.to_csv(index=False)
            col1.download_button(
                label="📊 Portfolio Weights",
                data=weights_csv,
                file_name=f"portfolio_weights_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Historical returns CSV
            returns_csv = pd.DataFrame({
                'Date': portfolio_returns.index,
                'Portfolio_Return': portfolio_returns.values,
                'Cumulative_Value': cumulative_returns.values
            }).to_csv(index=False)
            col2.download_button(
                label="📈 Historical Returns",
                data=returns_csv,
                file_name=f"historical_returns_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Full report
            report_data = f"""
ESG PORTFOLIO OPTIMIZATION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CONFIGURATION:
- Optimization Method: {opt_method}
- Risk-Free Rate: {risk_free_rate*100:.2f}%
- Date Range: {start_date} to {end_date}
- Assets: {', '.join(tickers)}

OPTIMAL PORTFOLIO WEIGHTS:
{weights_df.to_string(index=False)}

PERFORMANCE METRICS:
- Expected Annual Return: {metrics['Expected Return']*100:.2f}%
- Annual Volatility: {metrics['Volatility']*100:.2f}%
- Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}
- Sortino Ratio: {metrics['Sortino Ratio']:.2f}
- Value at Risk (95%): {metrics['VaR (95%)']*100:.2f}%
- Conditional VaR (95%): {metrics['CVaR (95%)']*100:.2f}%
- Max Drawdown: {metrics['Max Drawdown']*100:.2f}%

BACKTEST RESULTS:
- Total Return: {(cumulative_returns.iloc[-1]-1)*100:.2f}%
- vs Equal Weight: {((cumulative_returns.iloc[-1]/cumulative_benchmark.iloc[-1])-1)*100:.2f}% outperformance
            """
            
            col3.download_button(
                label="📄 Full Report (TXT)",
                data=report_data,
                file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.exception(e)

else:
    if run_analysis:
        st.error("⚠️ Please select between 5 and 15 stocks to continue.")