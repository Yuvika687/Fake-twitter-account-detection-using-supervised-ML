# feature_engineer_v2.py - ULTIMATE FEATURE EXTRACTOR
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import re

class UltimateFeatureEngineer:
    """Extracts 50+ powerful features for bot detection"""
    
    def __init__(self):
        self.bot_keywords = [
            'crypto', 'bitcoin', 'trader', 'profit', 'forex', 'stock',
            'DM', 'promo', 'follow', 'retweet', 'giveaway', 'winner',
            'discount', 'sale', 'click', 'link', 'http', 'www',
            'bot', 'automated', 'AI', 'algorithm'
        ]
    
    def extract_features(self, df):
        """Extract comprehensive features from raw data"""
        print("="*60)
        print("🔧 EXTRACTING ULTIMATE FEATURES")
        print("="*60)
        
        result = df.copy()
        
        # ==================== 1. TIME-BASED FEATURES ====================
        print("⏰ Extracting time features...")
        result = self._extract_time_features(result)
        
        # ==================== 2. ACTIVITY FEATURES ====================
        print("📈 Extracting activity features...")
        result = self._extract_activity_features(result)
        
        # ==================== 3. NETWORK FEATURES ====================
        print("🕸️ Extracting network features...")
        result = self._extract_network_features(result)
        
        # ==================== 4. PROFILE FEATURES ====================
        print("👤 Extracting profile features...")
        result = self._extract_profile_features(result)
        
        # ==================== 5. COMPOSITE SCORES ====================
        print("📊 Calculating composite scores...")
        result = self._calculate_composite_scores(result)
        
        # ==================== 6. BOT-SPECIFIC PATTERNS ====================
        print("🤖 Detecting bot-specific patterns...")
        result = self._detect_bot_patterns(result)
        
        # Select final features
        feature_columns = self._get_feature_columns(result)
        
        print(f"\n✅ Extracted {len(feature_columns)} features")
        print(f"📋 Feature categories: Time, Activity, Network, Profile, Scores, Patterns")
        
        return result, feature_columns
    
    def _extract_time_features(self, df):
        """Extract time-related features"""
        df = df.copy()
        
        # Account age
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        df['account_age_days'] = (datetime.now(timezone.utc) - df['created_at']).dt.days
        df['account_age_days'] = df['account_age_days'].fillna(365).clip(1, 10000)
        
        # Age categories (YOUR SUGGESTION!)
        df['account_age_category'] = pd.cut(
            df['account_age_days'],
            bins=[0, 7, 30, 90, 180, 365, 1095, float('inf')],
            labels=['week', 'month', 'quarter', 'half_year', 'year', '3_years', 'old']
        )
        
        # Time-based flags
        df['is_new_account'] = (df['account_age_days'] < 7).astype(int)
        df['is_very_new'] = (df['account_age_days'] < 30).astype(int)
        df['is_young'] = (df['account_age_days'] < 180).astype(int)
        df['is_mature'] = (df['account_age_days'] >= 365).astype(int)
        
        return df
    
    def _extract_activity_features(self, df):
        """Extract posting activity features"""
        df = df.copy()
        
        # Basic activity
        df['tweets_per_day'] = df['tweet_count'] / df['account_age_days'].clip(1)
        
        # YOUR SUGGESTED ACTIVITY LEVELS!
        df['activity_safe'] = (df['tweets_per_day'] < 20).astype(int)
        df['activity_risky'] = ((df['tweets_per_day'] >= 20) & (df['tweets_per_day'] < 50)).astype(int)
        df['activity_extreme'] = ((df['tweets_per_day'] >= 50) & (df['tweets_per_day'] < 100)).astype(int)
        df['activity_very_extreme'] = (df['tweets_per_day'] >= 100).astype(int)
        
        # Account age + activity combos (YOUR IDEA!)
        df['new_account_high_activity'] = ((df['is_new_account'] == 1) & (df['tweets_per_day'] > 10)).astype(int)
        df['new_account_extreme_activity'] = ((df['is_new_account'] == 1) & (df['tweets_per_day'] > 50)).astype(int)
        df['young_account_high_activity'] = ((df['is_young'] == 1) & (df['tweets_per_day'] > 30)).astype(int)
        df['mature_account_high_activity'] = ((df['is_mature'] == 1) & (df['tweets_per_day'] > 25)).astype(int)
        
        # Activity consistency
        df['activity_consistency'] = 1 / (1 + df['tweets_per_day'].std() if df['tweet_count'].std() > 0 else 1)
        
        return df
    
    def _extract_network_features(self, df):
        """Extract social network features"""
        df = df.copy()
        
        # Basic ratios
        df['followers_following_ratio'] = np.where(
            df['following_count'] > 0,
            df['followers_count'] / df['following_count'].clip(1),
            10  # If following=0, high ratio
        )
        
        df['following_followers_ratio'] = np.where(
            df['followers_count'] > 0,
            df['following_count'] / df['followers_count'].clip(1),
            10  # If followers=0, high ratio
        )
        
        # Engagement metrics
        df['followers_per_tweet'] = np.where(
            df['tweet_count'] > 0,
            df['followers_count'] / df['tweet_count'].clip(1),
            0
        )
        
        df['tweets_per_follower'] = np.where(
            df['followers_count'] > 0,
            df['tweet_count'] / df['followers_count'].clip(1),
            0
        )
        
        # Network anomalies (YOUR SUGGESTIONS!)
        df['low_followers_high_tweets'] = (
            (df['followers_count'] < 100) & 
            (df['tweet_count'] > 1000)
        ).astype(int)
        
        df['high_following_ratio'] = (
            df['following_count'] > df['followers_count'] * 3
        ).astype(int)
        
        df['egg_account'] = (
            (df['followers_count'] < 10) & 
            (df['tweet_count'] < 5)
        ).astype(int)
        
        df['suspicious_follower_ratio'] = (
            (df['followers_following_ratio'] > 10) | 
            (df['followers_following_ratio'] < 0.1)
        ).astype(int)
        
        return df
    
    def _extract_profile_features(self, df):
        """Extract profile/bio features"""
        df = df.copy()
        
        if 'description' not in df.columns:
            df['description'] = ''
        
        df['description'] = df['description'].fillna('')
        
        # Bio metrics
        df['bio_length'] = df['description'].str.len()
        df['bio_word_count'] = df['description'].str.split().str.len()
        df['has_bio'] = (df['bio_length'] > 0).astype(int)
        
        # Bio complexity
        df['bio_avg_word_length'] = df['description'].apply(
            lambda x: np.mean([len(w) for w in str(x).split()]) if str(x).split() else 0
        )
        
        # Suspicious patterns in bio
        df['bio_has_url'] = df['description'].str.contains(r'http[s]?://|www\.', na=False).astype(int)
        df['bio_has_emoji'] = df['description'].str.contains(
            r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF]', 
            na=False
        ).astype(int)
        
        # Bot keywords in bio (YOUR IDEA!)
        df['bio_bot_keywords'] = df['description'].apply(
            lambda x: sum(1 for kw in self.bot_keywords if kw.lower() in str(x).lower())
        )
        
        # Profile completeness score
        df['profile_completeness'] = (
            df['has_bio'] * 0.4 +
            (df['bio_length'] > 50).astype(int) * 0.3 +
            (df['followers_count'] > 0).astype(int) * 0.2 +
            df['verified'].astype(int) * 0.1
        )
        
        return df
    
    def _calculate_composite_scores(self, df):
        """Calculate intelligent composite scores"""
        df = df.copy()
        
        # 1. BOT RISK SCORE (0-1, higher = more bot-like)
        bot_risk_factors = (
            df['activity_very_extreme'] * 0.25 +
            df['new_account_extreme_activity'] * 0.20 +
            df['low_followers_high_tweets'] * 0.15 +
            df['high_following_ratio'] * 0.10 +
            df['bio_bot_keywords'].clip(0, 5) * 0.10 +
            df['suspicious_follower_ratio'] * 0.10 +
            df['egg_account'] * 0.05 +
            (df['bio_has_url'] & (df['bio_length'] < 100)).astype(int) * 0.05
        )
        df['bot_risk_score'] = bot_risk_factors.clip(0, 1)
        
        # 2. HUMAN CONFIDENCE SCORE (0-1, higher = more human-like)
        human_factors = (
            df['account_age_days'].apply(lambda x: min(x/1095, 1)) * 0.30 +  # 3 years max
            df['profile_completeness'] * 0.25 +
            ((df['followers_following_ratio'] > 0.5) & (df['followers_following_ratio'] < 2)).astype(int) * 0.20 +
            df['activity_safe'] * 0.15 +
            df['verified'].astype(int) * 0.10
        )
        df['human_confidence_score'] = human_factors.clip(0, 1)
        
        # 3. ENGAGEMENT QUALITY SCORE
        df['engagement_quality'] = np.where(
            df['followers_per_tweet'] > 0,
            np.log1p(df['followers_per_tweet']) / np.log1p(100),
            0
        ).clip(0, 1)
        
        # 4. ACCOUNT MATURITY SCORE
        df['account_maturity'] = np.log1p(df['account_age_days']) / np.log1p(1095)  # 3 years
        df['account_maturity'] = df['account_maturity'].clip(0, 1)
        
        return df
    
    def _detect_bot_patterns(self, df):
        """Detect specific bot behavior patterns"""
        df = df.copy()
        
        # Pattern 1: "Spammer" - High tweets, low followers, new account
        df['pattern_spammer'] = (
            (df['is_new_account'] == 1) &
            (df['tweets_per_day'] > 50) &
            (df['followers_per_tweet'] < 0.01)
        ).astype(int)
        
        # Pattern 2: "Follower farmer" - Following thousands, few tweets
        df['pattern_follower_farmer'] = (
            (df['following_count'] > 1000) &
            (df['tweet_count'] < 100) &
            (df['followers_count'] < 500)
        ).astype(int)
        
        # Pattern 3: "Amplifier" - Old account suddenly active
        df['pattern_amplifier'] = (
            (df['account_age_days'] > 365) &
            (df['tweets_per_day'] > 30) &
            (df['bio_bot_keywords'] > 2)
        ).astype(int)
        
        # Pattern 4: "Egg account" - Default/empty profile
        df['pattern_egg'] = (
            (df['has_bio'] == 0) &
            (df['followers_count'] < 20) &
            (df['tweet_count'] < 10)
        ).astype(int)
        
        # Pattern 5: "Suspicious ratios" - Extreme network patterns
        df['pattern_suspicious_ratios'] = (
            (df['followers_following_ratio'] > 50) |
            (df['following_followers_ratio'] > 50)
        ).astype(int)
        
        return df
    
    def _get_feature_columns(self, df):
        """Get final feature columns for model training"""
        # These are our powerful features
        features = [
            # Basic metrics
            'followers_count', 'following_count', 'tweet_count', 'verified',
            'account_age_days',
            
            # Time features
            'is_new_account', 'is_very_new', 'is_young', 'is_mature',
            
            # Activity features (YOUR KEY FEATURES!)
            'tweets_per_day', 'activity_safe', 'activity_risky', 
            'activity_extreme', 'activity_very_extreme',
            'new_account_high_activity', 'new_account_extreme_activity',
            'young_account_high_activity', 'mature_account_high_activity',
            
            # Network features
            'followers_following_ratio', 'following_followers_ratio',
            'followers_per_tweet', 'tweets_per_follower',
            'low_followers_high_tweets', 'high_following_ratio',
            'egg_account', 'suspicious_follower_ratio',
            
            # Profile features
            'bio_length', 'bio_word_count', 'has_bio', 'bio_avg_word_length',
            'bio_has_url', 'bio_has_emoji', 'bio_bot_keywords',
            'profile_completeness',
            
            # Composite scores
            'bot_risk_score', 'human_confidence_score',
            'engagement_quality', 'account_maturity',
            
            # Bot patterns
            'pattern_spammer', 'pattern_follower_farmer',
            'pattern_amplifier', 'pattern_egg', 'pattern_suspicious_ratios'
        ]
        
        # Add one-hot encoded categorical features
        if 'account_age_category' in df.columns:
            age_dummies = pd.get_dummies(df['account_age_category'], prefix='age')
            df = pd.concat([df, age_dummies], axis=1)
            features.extend(age_dummies.columns.tolist())
        
        return [f for f in features if f in df.columns]
    
    def extract_single_profile(self, profile_dict):
        """Extract features for a single Twitter profile"""
        df = pd.DataFrame([profile_dict])
        df_processed, _ = self.extract_features(df)
        return df_processed.iloc[0].to_dict()

# Test the feature engineer
if __name__ == "__main__":
    # Create test data
    test_data = {
        'followers_count': [100, 5000, 50],
        'following_count': [5000, 800, 5000],
        'tweet_count': [50000, 1200, 1000],
        'verified': [False, False, False],
        'description': ['', 'Software engineer', 'Crypto trader $$$'],
        'created_at': ['2023-12-01', '2018-01-01', '2023-01-01'],
        'label': [1, 0, 1]
    }
    
    df_test = pd.DataFrame(test_data)
    engineer = UltimateFeatureEngineer()
    df_features, features = engineer.extract_features(df_test)
    
    print(f"\n🧪 TEST RESULTS:")
    print(f"Input columns: {list(test_data.keys())}")
    print(f"Output features: {len(features)}")
    print(f"Sample features: {features[:10]}...")
    print(f"\nBot risk scores:")
    print(df_features[['bot_risk_score', 'human_confidence_score']].head())