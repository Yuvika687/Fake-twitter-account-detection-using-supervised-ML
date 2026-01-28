# api.py - UPDATED VERSION
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import joblib
import os
import pandas as pd
from dotenv import load_dotenv
import uvicorn
import sys
import traceback
import json
from datetime import datetime
from twitter_fetcher import TwitterFetcher 
import numpy as np
load_dotenv()

app = FastAPI(
    title="ULTIMATE Bot Detection API",
    description="Advanced bot detection using Ensemble ML + Multiple Datasets",
    version="4.0"
)

# Load the ENHANCED model
try:
    # Try different model paths in order
    MODEL_PATHS = [
        
        # Line ~30 in your api.py - update MODEL_PATHS:

        #os.path.join("..", "models", "clean_bot_detector.pkl"),  # NEW
        os.path.join("..", "models", "ultimate_bot_detector_v3.pkl"),  # OLD

    ]
    
    
    model_data = None
    model_path_used = None
    
    for MODEL_PATH in MODEL_PATHS:
        if os.path.exists(MODEL_PATH):
            print(f"✅ Found model at: {MODEL_PATH}")
            model_data = joblib.load(MODEL_PATH)
            model_path_used = MODEL_PATH
            break
    
    if model_data is None:
        raise FileNotFoundError("No model file found!")
    
    # Extract model components
    if 'models' in model_data:  # Ensemble model
        models = model_data['models']
        model_type = "ENSEMBLE (XGBoost + LightGBM + CatBoost + RF)"
        # Use XGBoost as primary predictor
        model = models.get('xgb', list(models.values())[0])
    else:
        models = {'primary': model_data['model']}
        model = model_data['model']
        model_type = model_data.get('model_type', 'SINGLE_MODEL')
    
    features = model_data['features']
    scaler = model_data.get('scaler', None)
    
    print(f"✅ {model_type} model loaded successfully!")
    print(f"📊 Total features: {len(features)}")
    print(f"🔧 Models available: {list(models.keys())}")
    print(f"📁 Model path: {model_path_used}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    traceback.print_exc()
    # Fallback to dummy model
    class DummyModel:
        def predict_proba(self, X):
            return [[0.4, 0.6]]
    model = DummyModel()
    models = {'dummy': model}
    features = ["followers_count", "following_count", "tweet_count", "verified"]
    scaler = None
    model_type = "DUMMY_FALLBACK"

class TwitterProfile(BaseModel):
    """Input schema for single profile prediction"""
    username: Optional[str] = None
    followers_count: int
    following_count: int
    tweet_count: int
    verified: bool = False
    description: str = ""
    created_at: str = ""
    # Optional fields for direct prediction
    bio_length: Optional[int] = None
    account_age_days: Optional[int] = None
    tweets_per_day: Optional[float] = None

class UsernameList(BaseModel):
    """Input schema for batch prediction"""
    usernames: List[str]

def extract_features_from_profile(profile):
    """
    Extract features from a Twitter profile dictionary
    WITH IMPROVED BOT DETECTION LOGIC
    """
    from datetime import datetime, timezone, timedelta
    
    # Create base feature dict
    feature_dict = {}
    
    # 1. Basic metrics
    feature_dict['followers_count'] = profile.get('followers_count', 0)
    feature_dict['following_count'] = profile.get('following_count', 0)
    feature_dict['tweet_count'] = profile.get('tweet_count', 0)
    feature_dict['verified'] = 1 if profile.get('verified', False) else 0
    
    # 2. Account age with proper date handling
    created_at = profile.get('created_at', '')
    try:
        created_date = pd.to_datetime(created_at)
        now = datetime.now(timezone.utc)
        
        if created_date > now:
            # Future date - suspicious!
            feature_dict['account_age_days'] = 0
            feature_dict['is_future_account'] = 1
        else:
            account_age_days = (now - created_date).days
            feature_dict['account_age_days'] = max(1, account_age_days)
            feature_dict['is_future_account'] = 0
    except:
        feature_dict['account_age_days'] = 365
        feature_dict['is_future_account'] = 0
    
    # 3. IMPROVED: Tweets per day with better categorization
    tweets_per_day = feature_dict['tweet_count'] / max(feature_dict['account_age_days'], 1)
    feature_dict['tweets_per_day'] = tweets_per_day
    
    # ==================== IMPROVED ACTIVITY ANALYSIS ====================
    
    # NEW: Activity levels based on your suggestions
    feature_dict['activity_safe'] = 1 if tweets_per_day < 20 else 0
    feature_dict['activity_risky'] = 1 if 20 <= tweets_per_day < 50 else 0
    feature_dict['activity_extreme'] = 1 if tweets_per_day >= 50 else 0
    feature_dict['activity_very_extreme'] = 1 if tweets_per_day >= 100 else 0
    
    # NEW: Account age based activity suspicion
    if feature_dict['account_age_days'] < 30:
        # Very new account
        feature_dict['new_account_high_activity'] = 1 if tweets_per_day > 10 else 0
        feature_dict['new_account_very_high_activity'] = 1 if tweets_per_day > 20 else 0
    elif feature_dict['account_age_days'] < 180:
        # Young account (1-6 months)
        feature_dict['young_account_high_activity'] = 1 if tweets_per_day > 30 else 0
    else:
        # Mature account
        feature_dict['mature_account_high_activity'] = 1 if tweets_per_day > 25 else 0
    
    # 4. Bio analysis
    description = profile.get('description', '')
    feature_dict['bio_length'] = len(str(description))
    feature_dict['bio_word_count'] = len(str(description).split())
    feature_dict['has_bio'] = 1 if description else 0
    
    # NEW: Bio complexity
    if description:
        words = str(description).split()
        if words:
            feature_dict['bio_avg_word_length'] = sum(len(word) for word in words) / len(words)
        else:
            feature_dict['bio_avg_word_length'] = 0
    else:
        feature_dict['bio_avg_word_length'] = 0
    
    # 5. Ratios and rates
    feature_dict['followers_following_ratio'] = (
        feature_dict['followers_count'] / max(feature_dict['following_count'], 1)
    )
    
    feature_dict['following_followers_ratio'] = (
        feature_dict['following_count'] / max(feature_dict['followers_count'], 1)
    )
    
    # NEW: Engagement metrics
    feature_dict['followers_per_tweet'] = (
        feature_dict['followers_count'] / max(feature_dict['tweet_count'], 1)
    )
    
    feature_dict['tweets_per_follower'] = (
        feature_dict['tweet_count'] / max(feature_dict['followers_count'], 1)
    )
    
    # 6. IMPROVED: Network anomalies
    # NEW: Follower quality based on your suggestion
    feature_dict['low_followers_high_tweets'] = (
        1 if (feature_dict['followers_count'] < 100 and 
              feature_dict['tweet_count'] > 1000) else 0
    )
    
    feature_dict['high_following_ratio'] = (
        1 if (feature_dict['following_count'] > feature_dict['followers_count'] * 3) else 0
    )
    
    feature_dict['low_followers_new_account'] = (
        1 if (feature_dict['followers_count'] < 50 and 
              feature_dict['account_age_days'] < 30) else 0
    )
    
    feature_dict['egg_account'] = (
        1 if (feature_dict['followers_count'] < 10 and 
              feature_dict['tweet_count'] < 5) else 0
    )
    
    # NEW: Suspicious ratio patterns
    feature_dict['very_high_follower_ratio'] = (
        1 if (feature_dict['followers_following_ratio'] > 10) else 0
    )
    
    feature_dict['very_low_follower_ratio'] = (
        1 if (feature_dict['followers_following_ratio'] < 0.1) else 0
    )
    
    # 7. NEW: Account maturity and trust scores
    if feature_dict['account_age_days'] > 0:
        # Maturity score (0-1, higher is better)
        feature_dict['account_maturity_score'] = min(
            np.log1p(feature_dict['account_age_days']) / np.log1p(365*5), 1
        )
    else:
        feature_dict['account_maturity_score'] = 0
    
    # NEW: Profile completeness score
    feature_dict['profile_completeness_score'] = (
        feature_dict['has_bio'] * 0.5 +
        (feature_dict['followers_count'] > 0) * 0.3 +
        (feature_dict['tweet_count'] > 10) * 0.2
    )
    
    # 8. IMPROVED: Composite bot scores
    # Bot risk factors
    bot_risk_factors = 0
    bot_risk_weights = {
        'activity_extreme': 0.30,
        'activity_very_extreme': 0.40,
        'new_account_high_activity': 0.25,
        'new_account_very_high_activity': 0.35,
        'low_followers_high_tweets': 0.20,
        'high_following_ratio': 0.15,
        'low_followers_new_account': 0.25,
        'egg_account': 0.10,
        'is_future_account': 0.50,  # High weight for future dates
        'very_low_follower_ratio': 0.15,
        'activity_risky': 0.10,
    }
    
    for factor, weight in bot_risk_weights.items():
        if factor in feature_dict and feature_dict[factor] == 1:
            bot_risk_factors += weight
    
    feature_dict['bot_risk_score'] = min(bot_risk_factors, 1.0)
    
    # Human confidence factors
    human_confidence_factors = 0
    human_weights = {
        'account_maturity_score': 0.40,
        'profile_completeness_score': 0.30,
        'activity_safe': 0.20,
        'verified': 0.10,
        'bio_length': 0.05 if feature_dict['bio_length'] > 50 else 0,
    }
    
    for factor, weight in human_weights.items():
        if factor == 'bio_length':
            if feature_dict['bio_length'] > 50:
                human_confidence_factors += weight
        elif factor in feature_dict:
            if isinstance(feature_dict[factor], (int, float)):
                human_confidence_factors += feature_dict[factor] * weight
    
    feature_dict['human_confidence_score'] = min(human_confidence_factors, 1.0)
    
    # 9. NEW: Account type classification
    # Determine likely account type
    if feature_dict['bot_risk_score'] > 0.7:
        account_type = "LIKELY_BOT"
    elif feature_dict['human_confidence_score'] > 0.7:
        account_type = "LIKELY_HUMAN"
    elif feature_dict['account_age_days'] < 30 and feature_dict['tweet_count'] > 500:
        account_type = "SUSPICIOUS_NEW"
    elif feature_dict['tweets_per_day'] > 50:
        account_type = "EXTREME_POSTER"
    elif feature_dict['followers_count'] > 10000 and feature_dict['tweets_per_day'] < 10:
        account_type = "INFLUENCER"
    else:
        account_type = "NORMAL"
    
    feature_dict['account_type'] = account_type
    
    # 10. Ensure all required features are present
    for feat in features:
        if feat not in feature_dict:
            if any(x in feat for x in ['count', 'len', 'age', 'day']):
                feature_dict[feat] = 0
            elif any(x in feat for x in ['ratio', 'score', 'rate', 'per']):
                feature_dict[feat] = 0.0
            elif any(x in feat for x in ['is_', 'has_', 'suspicious', 'inactive', 'new_', 'activity_']):
                feature_dict[feat] = 0
            else:
                feature_dict[feat] = 0
    
    return feature_dict
@app.post("/predict")
async def predict_single(profile: TwitterProfile):
    try:
        # Convert profile to dict
        profile_dict = profile.dict()

        # Extract features
        features_dict = extract_features_from_profile(profile_dict)

        # Create DataFrame with correct feature order
        X = pd.DataFrame([features_dict])[features]

        # Apply scaling if available
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X

        # Get predictions from all models in ensemble
        predictions = {}
        probabilities = {}

        for model_name, model_obj in models.items():
            try:
                proba = model_obj.predict_proba(X_scaled)[0, 1]
                pred = "BOT" if proba > 0.5 else "HUMAN"
                predictions[model_name] = pred
                probabilities[model_name] = float(proba)
            except:
                predictions[model_name] = "ERROR"
                probabilities[model_name] = 0.5

        # Ensemble prediction
        ensemble_proba = np.mean(list(probabilities.values()))
        ensemble_pred = "BOT" if ensemble_proba > 0.5 else "HUMAN"

        # Confidence
        confidence = (
            "HIGH" if abs(ensemble_proba - 0.5) > 0.3 else
            "MEDIUM" if abs(ensemble_proba - 0.5) > 0.15 else
            "LOW"
        )

        # Key factors
        key_factors = []
        if features_dict.get('is_extreme_active') == 1:
            key_factors.append("Extremely high activity rate")
        if features_dict.get('high_following_low_followers') == 1:
            key_factors.append("Following many but few followers")
        if features_dict.get('account_age_days') < 30:
            key_factors.append("Very new account")
        if features_dict.get('is_egg_account') == 1:
            key_factors.append("Egg account (low followers & tweets)")

        # BASE RESPONSE
        response_data = {
            "prediction": ensemble_pred,
            "confidence": confidence,
            "probability_fake": float(ensemble_proba),
            "probability_real": float(1 - ensemble_proba),
            "model_type": model_type,
            "model_consensus": predictions,
            "model_probabilities": probabilities,
            "key_factors": key_factors,
            "account_analysis": {
                "account_age_days": features_dict.get('account_age_days'),
                "tweets_per_day": features_dict.get('tweets_per_day'),
                "followers_following_ratio": features_dict.get('followers_following_ratio'),
                "bot_risk_score": features_dict.get('bot_risk_score'),
                "human_confidence_score": features_dict.get('human_confidence_score')
            }
        }

        # -----------------------------------------------
        # ✅ ADD DETAILED ANALYSIS (Correct indentation)
        # -----------------------------------------------
        detailed_analysis = {
            "account_age_category":
                "NEW (<30 days)" if features_dict.get('account_age_days', 0) < 30 else
                "YOUNG (1-6 months)" if features_dict.get('account_age_days', 0) < 180 else
                "MATURE (>6 months)",

            "activity_level":
                "SAFE (<20/day)" if features_dict.get('activity_safe', 0) == 1 else
                "RISKY (20-50/day)" if features_dict.get('activity_risky', 0) == 1 else
                "EXTREME (50-100/day)" if features_dict.get('activity_extreme', 0) == 1 else
                "VERY EXTREME (>100/day)",

            "bot_indicators": [],
            "human_indicators": []
        }

        # Bot indicators
        if features_dict.get('activity_extreme', 0) == 1:
            detailed_analysis["bot_indicators"].append(
                f"Extreme posting: {features_dict.get('tweets_per_day', 0):.1f}/day"
            )

        if features_dict.get('new_account_high_activity', 0) == 1:
            detailed_analysis["bot_indicators"].append("New account with high activity")

        if features_dict.get('low_followers_high_tweets', 0) == 1:
            detailed_analysis["bot_indicators"].append("Low followers but many tweets")

        if features_dict.get('high_following_ratio', 0) == 1:
            detailed_analysis["bot_indicators"].append("Following many more than followers")

        # Human indicators
        if features_dict.get('account_maturity_score', 0) > 0.7:
            detailed_analysis["human_indicators"].append(
                f"Mature account ({features_dict.get('account_age_days')} days old)"
            )

        if features_dict.get('profile_completeness_score', 0) > 0.5:
            detailed_analysis["human_indicators"].append("Complete profile with bio")

        if features_dict.get('activity_safe', 0) == 1:
            detailed_analysis["human_indicators"].append(
                f"Normal posting: {features_dict.get('tweets_per_day', 0):.1f}/day"
            )

        # Attach to response
        response_data["detailed_analysis"] = detailed_analysis

        return response_data

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict-direct")
async def predict_direct(profile: dict):
    """
    Direct prediction from profile dictionary
    """
    try:
        features_dict = extract_features_from_profile(profile)
        X = pd.DataFrame([features_dict])[features]
        
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
        
        # Use primary model
        proba = model.predict_proba(X_scaled)[0, 1]
        pred = "BOT" if proba > 0.5 else "HUMAN"
        
        return {
            "prediction": pred,
            "probability_fake": float(proba),
            "features_used": list(features_dict.keys())
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch")
async def predict_batch(profiles: List[dict]):
    """
    Batch prediction for multiple profiles
    """
    results = []
    
    for profile in profiles:
        try:
            features_dict = extract_features_from_profile(profile)
            X = pd.DataFrame([features_dict])[features]
            
            if scaler is not None:
                X_scaled = scaler.transform(X)
            else:
                X_scaled = X
            
            proba = model.predict_proba(X_scaled)[0, 1]
            pred = "BOT" if proba > 0.5 else "HUMAN"
            
            results.append({
                "username": profile.get('username', 'unknown'),
                "prediction": pred,
                "probability_fake": float(proba),
                "followers": profile.get('followers_count', 0),
                "verified": profile.get('verified', False)
            })
            
        except Exception as e:
            results.append({
                "username": profile.get('username', 'unknown'),
                "error": str(e),
                "prediction": "ERROR"
            })
    
    return {
        "total": len(results),
        "successful": len([r for r in results if 'error' not in r]),
        "results": results
    }
@app.post("/predict-username")
async def predict_by_username(username_data: dict):
    """Predict from username (fetches data from Twitter)"""
    try:
        username = username_data.get("username", "").strip()
        if not username:
            raise HTTPException(status_code=400, detail="Username required")
        
        # Fetch from Twitter
        fetcher = TwitterFetcher()
        profile = fetcher.fetch_user(username)
        
        if "error" in profile:
            raise HTTPException(status_code=404, detail=profile["error"])
        
        # Extract features and predict
        features_dict = extract_features_from_profile(profile)
        X = pd.DataFrame([features_dict])[features]
        
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
        
        # Get predictions
        predictions = {}
        probabilities = {}
        
        for model_name, model_obj in models.items():
            try:
                proba = model_obj.predict_proba(X_scaled)[0, 1]
                pred = "BOT" if proba > 0.5 else "HUMAN"
                predictions[model_name] = pred
                probabilities[model_name] = float(proba)
            except Exception:
                predictions[model_name] = "ERROR"
                probabilities[model_name] = 0.5
        
        # Ensemble prediction
        ensemble_proba = np.mean(list(probabilities.values()))
        ensemble_pred = "BOT" if ensemble_proba > 0.5 else "HUMAN"
        
        # Confidence
        confidence = "HIGH" if abs(ensemble_proba - 0.5) > 0.3 else \
                    "MEDIUM" if abs(ensemble_proba - 0.5) > 0.15 else "LOW"
        
        return {
            "prediction": ensemble_pred,
            "confidence": confidence,
            "probability_fake": float(ensemble_proba),
            "probability_real": float(1 - ensemble_proba),
            "model_type": model_type,
            "model_consensus": predictions,
            "model_probabilities": probabilities,
            "profile": profile
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "ULTIMATE Bot Detection API v4.0",
        "model": model_type,
        "features": len(features),
        "models": list(models.keys())
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_type != "DUMMY_FALLBACK",
        "model_type": model_type,
        "features_count": len(features)
    }

@app.get("/model-info")
async def model_info():
    """Get detailed model information"""
    return {
        "model_type": model_type,
        "features": features,
        "features_count": len(features),
        "has_scaler": scaler is not None,
        "models_available": list(models.keys())
    }

if __name__ == "__main__":
    print("🚀 Starting ULTIMATE Bot Detection API...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)