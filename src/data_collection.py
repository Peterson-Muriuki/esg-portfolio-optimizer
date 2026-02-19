import yfinance as yf
import pandas as pd
import feedparser
import time
from datetime import datetime
from urllib.parse import quote  # for URL encoding


class DataCollector:
    def __init__(self, tickers):
        self.tickers = tickers

    def get_stock_data(self, start_date, end_date):
        """Download stock price data"""
        print(f"Downloading price data for {len(self.tickers)} stocks...")
        try:
            data = yf.download(self.tickers, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                print("No data returned from yfinance.")
                return pd.DataFrame()

            if isinstance(data.columns, pd.MultiIndex):
                prices = data['Adj Close']
            else:
                # Single ticker case
                if 'Adj Close' in data.columns:
                    prices = data[['Adj Close']].rename(columns={'Adj Close': self.tickers[0]})
                else:
                    print("'Adj Close' not found in downloaded data.")
                    return pd.DataFrame()
            
            print("Price data downloaded successfully")
            return prices
        except Exception as e:
            print(f"Error downloading data: {e}")
            return pd.DataFrame()

    def get_esg_scores(self):
        """Get ESG scores from yfinance"""
        print("Fetching ESG scores...")
        esg_data = {}
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                esg_data[ticker] = {
                    'company': info.get('shortName', ticker),
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown')
                }
            except Exception as e:
                print(f"Warning: Could not fetch ESG for {ticker} ({e})")
                esg_data[ticker] = {
                    'company': ticker,
                    'sector': 'Unknown',
                    'industry': 'Unknown'
                }
        print("ESG data fetched")
        return pd.DataFrame(esg_data).T

    def scrape_news_headlines(self, ticker, num_articles=10):
        """Scrape recent news headlines for ESG keywords"""
        print(f"  Fetching news for {ticker}...", end=" ")
        try:
            # URL encode the query to avoid spaces/control characters
            query = quote(f"{ticker} ESG sustainability environmental social governance")
            url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(url)
            
            headlines = []
            for entry in feed.entries[:num_articles]:
                headlines.append({
                    'ticker': ticker,
                    'title': entry.title,
                    'published': getattr(entry, 'published', ''),
                    'link': entry.link
                })
            print(f"({len(headlines)} articles)")
            return pd.DataFrame(headlines)
        except Exception as e:
            print(f"Error: {e}")
            return pd.DataFrame()

    def collect_all_news(self):
        """Collect news for all tickers"""
        print(f"\nCollecting news for {len(self.tickers)} stocks:")
        all_news = []
        for ticker in self.tickers:
            news = self.scrape_news_headlines(ticker)
            if not news.empty:
                all_news.append(news)
            time.sleep(1)  # Be respectful to servers

        if all_news:
            result = pd.concat(all_news, ignore_index=True)
            print(f"\nTotal articles collected: {len(result)}")
            return result
        else:
            print("\nNo news collected")
            return pd.DataFrame()


# Test function
if __name__ == "__main__":
    print("Testing DataCollector...")
    tickers = ['AAPL', 'MSFT', 'TSLA']
    collector = DataCollector(tickers)
    
    # Test price data
    prices = collector.get_stock_data('2023-01-01', '2024-01-01')
    print(f"\nPrice data shape: {prices.shape}")
    
    # Test ESG scores
    esg = collector.get_esg_scores()
    print(f"ESG data shape: {esg.shape}")
    
    # Test news
    news = collector.collect_all_news()
    print(f"News data shape: {news.shape}")
