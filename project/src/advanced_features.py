# advanced_features.py
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import re
from typing import Dict, List

class AdvancedBotFeatureEngineer:
    """Advanced feature engineering for bot detection"""
    
    def __init__(self):
        self.feature_cache = {}
    
    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract comprehensive bot detection features"""
        
        # Create a copy
        result_df = df.copy()
        
        # ==================== 1. BASIC ACCOUNT FEATURES ====================
        print("🔧 Extracting basic account features...")
        
        # Account age
        result_df['created_at'] = pd.to_datetime(result_df['created_at'], errors='coerce')
        result_df['account_age_days'] = (datetime.now(timezone.utc) - result_df['created_at']).dt.days
        result_df['account_age_days'] = result_df['account_age_days'].fillna(365).clip(0, 10000)
        
        # Age categories
        result_df['account_age_category'] = pd.cut(
            result_df['account_age_days'],
            bins=[0, 30, 180, 365, 1095, float('inf')],
            labels=['new_0_30', 'young_30_180', 'medium_180_365', 'mature_1_3', 'old_3+']
        )
        
        # ==================== 2. ENGAGEMENT & SOCIAL GRAPH ====================
        print("🔧 Extracting engagement features...")
        
        # Follow ratios
        result_df['followers_following_ratio'] = np.where(
            result_df['following_count'] > 0,
            result_df['followers_count'] / result_df['following_count'],
            10  # High ratio if no following
        )
        
        result_df['following_followers_ratio'] = np.where(
            result_df['followers_count'] > 0,
            result_df['following_count'] / result_df['followers_count'],
            10  # High ratio if no followers
        )
        
        # Engagement rates
        result_df['tweets_per_day'] = np.where(
            result_df['account_age_days'] > 0,
            result_df['tweet_count'] / result_df['account_age_days'],
            0
        )
        
        result_df['followers_per_tweet'] = np.where(
            result_df['tweet_count'] > 0,
            result_df['followers_count'] / result_df['tweet_count'],
            0
        )
        
        # ==================== 3. ACTIVITY PATTERNS ====================
        print("🔧 Extracting activity patterns...")
        
        # Activity intensity
        result_df['activity_level'] = pd.cut(
            result_df['tweets_per_day'],
            bins=[-1, 0.1, 1, 10, 50, 100, float('inf')],
            labels=['inactive', 'low', 'normal', 'high', 'very_high', 'extreme']
        )
        
        # Bot-like activity flags
        result_df['is_extreme_active'] = (result_df['tweets_per_day'] > 100).astype(int)
        result_df['is_very_active'] = (result_df['tweets_per_day'] > 50).astype(int)
        result_df['is_inactive'] = (result_df['tweets_per_day'] < 0.1).astype(int)
        
        # ==================== 4. PROFILE ANALYSIS ====================
        print("🔧 Analyzing profile features...")
        
        if 'description' in result_df.columns:
            result_df['description'] = result_df['description'].fillna('')
            
            # Bio length features
            result_df['bio_length'] = result_df['description'].str.len()
            result_df['bio_word_count'] = result_df['description'].str.split().str.len()
            result_df['has_bio'] = (result_df['bio_length'] > 0).astype(int)
            
            # Bio complexity
            result_df['bio_avg_word_length'] = result_df['description'].apply(
                lambda x: np.mean([len(word) for word in str(x).split()]) if len(str(x).split()) > 0 else 0
            )
            
            # Suspicious patterns in bio
            result_df['bio_has_url'] = result_df['description'].str.contains(r'http[s]?://', na=False).astype(int)
            result_df['bio_has_emoji'] = result_df['description'].str.contains(
                r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', 
                na=False
            ).astype(int)
            
            # Common bot keywords
            bot_keywords = ['crypto', 'bitcoin', 'trader', 'profit', 'forex', 'DM', 'promo', 'follow', 'retweet']
            result_df['bio_bot_keywords'] = result_df['description'].apply(
                lambda x: sum(1 for kw in bot_keywords if kw.lower() in str(x).lower())
            )
        
        else:
            result_df['bio_length'] = 0
            result_df['bio_word_count'] = 0
            result_df['has_bio'] = 0
            result_df['bio_avg_word_length'] = 0
            result_df['bio_has_url'] = 0
            result_df['bio_has_emoji'] = 0
            result_df['bio_bot_keywords'] = 0
        
        # ==================== 5. NETWORK ANOMALIES ====================
        print("🔧 Extracting network anomaly features...")
        
        # Suspicious follower patterns
        result_df['low_followers_high_activity'] = (
            (result_df['followers_count'] < 100) & 
            (result_df['tweet_count'] > 1000)
        ).astype(int)
        
        result_df['high_following_low_followers'] = (
            (result_df['following_count'] > result_df['followers_count'] * 3) & 
            (result_df['followers_count'] > 100)
        ).astype(int)
        
        # Egg account detection
        result_df['is_egg_account'] = (
            (result_df['followers_count'] < 10) & 
            (result_df['tweet_count'] < 5)
        ).astype(int)
        
        # ==================== 6. TEMPORAL FEATURES ====================
        print("🔧 Extracting temporal features...")
        
        # Account maturity score
        result_df['account_maturity_score'] = np.log1p(result_df['account_age_days']) / np.log1p(365*5)
        result_df['account_maturity_score'] = result_df['account_maturity_score'].clip(0, 1)
        
        # Activity density
        result_df['activity_density'] = result_df['tweets_per_day'] / (result_df['account_age_days'] + 1)
        
        # ==================== 7. COMPOSITE BOT SCORES ====================
        print("🔧 Calculating composite scores...")
        
        # Bot likelihood based on multiple signals
        result_df['bot_risk_score'] = (
            result_df['is_extreme_active'] * 0.25 +
            result_df['high_following_low_followers'] * 0.20 +
            result_df['low_followers_high_activity'] * 0.15 +
            result_df['bio_bot_keywords'].clip(0, 3) * 0.10 +
            (result_df['account_age_days'] < 30).astype(int) * 0.15 +
            (result_df['followers_following_ratio'] > 5).astype(int) * 0.15
        )
        
        # Human confidence score
        result_df['human_confidence_score'] = (
            result_df['account_maturity_score'] * 0.30 +
            ((result_df['tweets_per_day'] > 0.5) & (result_df['tweets_per_day'] < 20)).astype(int) * 0.25 +
            ((result_df['followers_following_ratio'] > 0.2) & (result_df['followers_following_ratio'] < 3)).astype(int) * 0.25 +
            result_df['has_bio'] * 0.20
        )
        
        # ==================== 8. FINAL FEATURE SELECTION ====================
        print("🔧 Selecting final features...")
        
        # Define feature columns
        feature_columns = [
            # Basic metrics
            'followers_count', 'following_count', 'tweet_count', 'verified',
            'account_age_days',
            
            # Ratio features
            'followers_following_ratio', 'following_followers_ratio',
            'tweets_per_day', 'followers_per_tweet',
            
            # Activity patterns
            'is_extreme_active', 'is_very_active', 'is_inactive',
            
            # Bio features
            'bio_length', 'bio_word_count', 'has_bio', 'bio_avg_word_length',
            'bio_has_url', 'bio_has_emoji', 'bio_bot_keywords',
            
            # Network anomalies
            'low_followers_high_activity', 'high_following_low_followers',
            'is_egg_account',
            
            # Temporal features
            'account_maturity_score', 'activity_density',
            
            # Composite scores
            'bot_risk_score', 'human_confidence_score'
        ]
        
        # Convert categorical to one-hot if needed
        if 'account_age_category' in result_df.columns:
            age_dummies = pd.get_dummies(result_df['account_age_category'], prefix='age')
            result_df = pd.concat([result_df, age_dummies], axis=1)
            feature_columns.extend(age_dummies.columns.tolist())
        
        if 'activity_level' in result_df.columns:
            activity_dummies = pd.get_dummies(result_df['activity_level'], prefix='activity')
            result_df = pd.concat([result_df, activity_dummies], axis=1)
            feature_columns.extend(activity_dummies.columns.tolist())
        
        # Ensure all features exist
        for col in feature_columns:
            if col not in result_df.columns:
                result_df[col] = 0
        
        # Fill NaN values
        result_df[feature_columns] = result_df[feature_columns].fillna(0)
        
        print(f"✅ Extracted {len(feature_columns)} features")
        
        return result_df, feature_columns
    
    def analyze_single_account(self, profile: Dict) -> Dict:
        """Analyze a single Twitter account"""
        
        df = pd.DataFrame([profile])
        df, features = self.extract_all_features(df)
        
        # Extract feature values
        feature_values = {col: df[col].iloc[0] for col in features}
        
        # Generate insights
        insights = {
            'bot_risk_score': float(df['bot_risk_score'].iloc[0]),
            'human_confidence_score': float(df['human_confidence_score'].iloc[0]),
            
            'account_age_category': 'New (< 30 days)' if df['account_age_days'].iloc[0] < 30 else 
                                   'Young (1-6 months)' if df['account_age_days'].iloc[0] < 180 else 
                                   'Medium (6-12 months)' if df['account_age_days'].iloc[0] < 365 else 
                                   'Mature (1-3 years)' if df['account_age_days'].iloc[0] < 1095 else 
                                   'Old (> 3 years)',
            
            'activity_level': 'Extreme (> 100/day)' if df['is_extreme_active'].iloc[0] == 1 else 
                             'Very High (> 50/day)' if df['is_very_active'].iloc[0] == 1 else 
                             'Normal' if df['tweets_per_day'].iloc[0] > 1 else 
                             'Inactive',
            
            'follow_pattern': 'Mass Follower' if df['high_following_low_followers'].iloc[0] == 1 else 
                             'Balanced' if df['followers_following_ratio'].iloc[0] > 0.2 and df['followers_following_ratio'].iloc[0] < 3 else 
                             'Popular (many followers)',
            
            'engagement_quality': 'Low Engagement' if df['low_followers_high_activity'].iloc[0] == 1 else 
                                 'Good Engagement' if df['followers_per_tweet'].iloc[0] > 0.1 else 
                                 'Average',
            
            'profile_quality': 'Complete' if df['has_bio'].iloc[0] == 1 else 'Incomplete'
        }
        
        # Prediction
        if df['bot_risk_score'].iloc[0] > 0.6:
            prediction = "BOT"
            confidence = "HIGH" if df['bot_risk_score'].iloc[0] > 0.8 else "MEDIUM"
        elif df['human_confidence_score'].iloc[0] > 0.6:
            prediction = "HUMAN"
            confidence = "HIGH" if df['human_confidence_score'].iloc[0] > 0.8 else "MEDIUM"
        else:
            prediction = "UNKNOWN"
            confidence = "LOW"
        
        insights['prediction'] = prediction
        insights['confidence'] = confidence
        
        return feature_values, insights