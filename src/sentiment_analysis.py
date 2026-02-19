import pandas as pd
import numpy as np
from textblob import TextBlob
import re

class SentimentAnalyzer:
    def __init__(self):
        # ESG keywords dictionary
        self.esg_keywords = {
            'environmental': ['climate', 'carbon', 'renewable', 'emission', 'sustainability', 
                              'green', 'pollution', 'waste', 'energy', 'solar', 'wind'],
            'social': ['diversity', 'labor', 'human rights', 'community', 'safety', 
                       'employee', 'workplace', 'discrimination', 'equity', 'inclusion'],
            'governance': ['ethics', 'board', 'transparency', 'compliance', 'leadership',
                           'corruption', 'audit', 'regulation', 'accountability', 'shareholder']
        }
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def calculate_sentiment(self, text):
        """Calculate sentiment polarity (-1 to 1)"""
        text = self.clean_text(text)
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0.0
    
    def calculate_esg_relevance(self, text):
        """Calculate ESG relevance score"""
        text = self.clean_text(text)
        scores = {}
        for category, keywords in self.esg_keywords.items():
            score = sum([text.count(keyword) for keyword in keywords])
            scores[f'{category}_score'] = score
        scores['total_esg_score'] = sum(scores.values())
        return scores
    
    def analyze_news(self, news_df):
        """Analyze all news headlines"""
        if news_df.empty:
            return pd.DataFrame()
        
        print("\nAnalyzing sentiment...")
        # Clean titles
        news_df['cleaned_title'] = news_df['title'].apply(self.clean_text)
        # Sentiment polarity
        news_df['sentiment'] = news_df['cleaned_title'].apply(self.calculate_sentiment)
        # ESG relevance
        esg_scores = news_df['cleaned_title'].apply(self.calculate_esg_relevance)
        esg_df = pd.DataFrame(esg_scores.tolist())
        # Combine
        result = pd.concat([news_df, esg_df], axis=1)
        print("✓ Sentiment analysis complete")
        return result
    
    def aggregate_by_ticker(self, news_df):
        """Aggregate sentiment and ESG scores by ticker"""
        if news_df.empty:
            return pd.DataFrame()
        
        agg_dict = {
            'sentiment': ['mean', 'std', 'min', 'max', 'count'],
            'total_esg_score': 'sum',
            'environmental_score': 'sum',
            'social_score': 'sum',
            'governance_score': 'sum'
        }
        
        ticker_sentiment = news_df.groupby('ticker').agg(agg_dict)
        # Flatten multi-index columns
        ticker_sentiment.columns = ['_'.join(col).strip() for col in ticker_sentiment.columns]
        # Fill NaNs in std with 0
        ticker_sentiment['sentiment_std'] = ticker_sentiment['sentiment_std'].fillna(0.0)
        return ticker_sentiment


# Test function
if __name__ == "__main__":
    print("Testing SentimentAnalyzer...")

    # Example news (replace with your DataCollector output)
    sample_news = pd.DataFrame({
        'ticker': ['AAPL', 'AAPL', 'MSFT', 'TSLA'],
        'title': [
            'Apple announces carbon neutral initiative',
            'Apple faces labor concerns in supply chain',
            'Microsoft invests in renewable energy',
            'Tesla reports zero emissions target'
        ]
    })

    analyzer = SentimentAnalyzer()
    analyzed = analyzer.analyze_news(sample_news)
    
    print("\nAnalyzed news:")
    print(analyzed[['ticker', 'title', 'sentiment', 'total_esg_score']])

    aggregated = analyzer.aggregate_by_ticker(analyzed)
    print("\nAggregated by ticker:")
    print(aggregated)
