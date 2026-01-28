# api.py - UPDATED VERSION WITH BEHAVIORAL ANALYSIS
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
    title="Twitter Behavioral Analysis API",
    description="Analyze Twitter accounts for bot-like behavior patterns",
    version="5.0"
)

# Load the ENHANCED model
try:
    # Try different model paths in order
    MODEL_PATHS = [
        os.path.join("..", "models", "clean_bot_detector.pkl"),  # NEW
        os.path.join("..", "models", "ultimate_bot_detector_v3.pkl"),  # OLD
        os.path.join("models", "clean_bot_detector.pkl"),
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
    features = ["followers_count", "following_count", "tweet_count"]
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
    bio_length: Optional[int] = None
    account_age_days: Optional[int] = None
    tweets_per_day: Optional[float] = None

class UsernameList(BaseModel):
    """Input schema for batch prediction"""
    usernames: List[str]

def extract_features_from_profile(profile):
    """
    Extract features from a Twitter profile dictionary
    WITH IMPROVED BEHAVIORAL FEATURES
    """
    from datetime import datetime, timezone, timedelta
    
    # Create base feature dict
    feature_dict = {}
    
    # 1. Basic metrics (NO verified in logic)
    feature_dict['followers_count'] = profile.get('followers_count', 0)
    feature_dict['following_count'] = profile.get('following_count', 0)
    feature_dict['tweet_count'] = profile.get('tweet_count', 0)
    feature_dict['verified'] = 0  # Always 0, not used in logic
    
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
    
    # ==================== BEHAVIORAL ACTIVITY LEVELS ====================
    
    # Activity levels for behavioral analysis
    feature_dict['activity_safe'] = 1 if tweets_per_day < 20 else 0
    feature_dict['activity_risky'] = 1 if 20 <= tweets_per_day < 50 else 0
    feature_dict['activity_extreme'] = 1 if 50 <= tweets_per_day < 100 else 0
    feature_dict['activity_very_extreme'] = 1 if tweets_per_day >= 100 else 0
    
    # Account age based activity suspicion
    if feature_dict['account_age_days'] < 30:
        # Very new account
        feature_dict['new_account'] = 1
        feature_dict['young_account'] = 0
        feature_dict['mature_account'] = 0
        feature_dict['new_account_high_activity'] = 1 if tweets_per_day > 10 else 0
        feature_dict['new_account_very_high_activity'] = 1 if tweets_per_day > 20 else 0
    elif feature_dict['account_age_days'] < 180:
        # Young account (1-6 months)
        feature_dict['new_account'] = 0
        feature_dict['young_account'] = 1
        feature_dict['mature_account'] = 0
        feature_dict['young_account_high_activity'] = 1 if tweets_per_day > 15 else 0
    else:
        # Mature account
        feature_dict['new_account'] = 0
        feature_dict['young_account'] = 0
        feature_dict['mature_account'] = 1
        feature_dict['mature_account_high_activity'] = 1 if tweets_per_day > 25 else 0
    
    # 4. Bio analysis (for profile completeness)
    description = profile.get('description', '')
    feature_dict['bio_length'] = len(str(description))
    feature_dict['bio_word_count'] = len(str(description).split())
    feature_dict['has_bio'] = 1 if description else 0
    
    # 5. Ratios and rates for behavioral patterns
    feature_dict['followers_following_ratio'] = (
        feature_dict['followers_count'] / max(feature_dict['following_count'], 1)
    )
    
    feature_dict['followers_per_tweet'] = (
        feature_dict['followers_count'] / max(feature_dict['tweet_count'], 1)
    )
    
    feature_dict['tweets_per_follower'] = (
        feature_dict['tweet_count'] / max(feature_dict['followers_count'], 1)
    )
    
    # 6. BEHAVIORAL ANOMALIES
    # Low followers but high tweets (shadowbanned behavior)
    feature_dict['low_followers_high_tweets'] = (
        1 if (feature_dict['followers_count'] < 100 and 
              feature_dict['tweet_count'] > 1000) else 0
    )
    
    # Following many but few followers (amplifier behavior)
    feature_dict['high_following_ratio'] = (
        1 if (feature_dict['following_count'] > feature_dict['followers_count'] * 3) else 0
    )
    
    # New account with few followers
    feature_dict['low_followers_new_account'] = (
        1 if (feature_dict['followers_count'] < 50 and 
              feature_dict['new_account'] == 1) else 0
    )
    
    # Egg account
    feature_dict['egg_account'] = (
        1 if (feature_dict['followers_count'] < 10 and 
              feature_dict['tweet_count'] < 5) else 0
    )
    
    # High activity with low engagement
    feature_dict['high_activity_low_engagement'] = (
        1 if (feature_dict['tweets_per_day'] > 30 and 
              feature_dict['followers_per_tweet'] < 0.1) else 0
    )
    
    # 7. ACCOUNT MATURITY and COMPLETENESS (for human-like scoring)
    if feature_dict['account_age_days'] > 0:
        feature_dict['account_maturity_score'] = min(
            np.log1p(feature_dict['account_age_days']) / np.log1p(365*5), 1
        )
    else:
        feature_dict['account_maturity_score'] = 0
    
    # Profile completeness (without verification)
    feature_dict['profile_completeness_score'] = (
        feature_dict['has_bio'] * 0.5 +
        (feature_dict['followers_count'] > 0) * 0.3 +
        (feature_dict['tweet_count'] > 10) * 0.2
    )
    
    # 8. BEHAVIORAL RISK SCORES (NO verified in weights)
    bot_risk_factors = 0
    bot_risk_weights = {
        'activity_extreme': 0.25,
        'activity_very_extreme': 0.35,
        'new_account_high_activity': 0.25,
        'new_account_very_high_activity': 0.30,
        'low_followers_high_tweets': 0.20,
        'high_following_ratio': 0.15,
        'low_followers_new_account': 0.25,
        'high_activity_low_engagement': 0.20,
        'is_future_account': 0.50,
        'egg_account': 0.10,
    }
    
    for factor, weight in bot_risk_weights.items():
        if factor in feature_dict and feature_dict[factor] == 1:
            bot_risk_factors += weight
    
    feature_dict['bot_risk_score'] = min(bot_risk_factors, 1.0)
    
    # 9. Ensure all required features are present
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

def classify_behavior(features_dict, ml_score):
    """
    Classify account behavior based on features and ML score
    Returns behavioral pattern, not identity
    """
    age = features_dict['account_age_days']
    tpd = features_dict['tweets_per_day']
    fpt = features_dict['followers_per_tweet']
    ratio = features_dict['followers_following_ratio']
    fcount = features_dict['followers_count']
    
    # CASE 1: TRUE AUTOMATED BEHAVIOR (High risk)
    if (age < 30 and tpd > 50 and ratio < 0.1):
        return {
            "behavior": "AUTOMATED_BEHAVIOR",
            "risk_level": "HIGH",
            "description": "Very new account with extreme posting and poor engagement ratios",
            "confidence": "HIGH"
        }
    
    # CASE 2: AGGRESSIVE HUMAN POSTING
    elif (age > 180 and tpd > 15 and tpd < 50):
        return {
            "behavior": "AGGRESSIVE_POSTING",
            "risk_level": "MEDIUM",
            "description": "Established account with unusually high posting frequency",
            "confidence": "MEDIUM"
        }
    
    # CASE 3: LOW VISIBILITY / SHADOWED
    elif (tpd > 15 and fpt < 0.05 and age > 365):
        return {
            "behavior": "LOW_VISIBILITY_ACTIVITY",
            "risk_level": "MEDIUM",
            "description": "High activity with very low engagement, possibly shadowbanned",
            "confidence": "MEDIUM"
        }
    
    # CASE 4: AMPLIFIER / FOLLOW-BACK
    elif (ratio < 0.2 and fcount < 1000 and age > 60):
        return {
            "behavior": "AMPLIFIER_BEHAVIOR",
            "risk_level": "LOW",
            "description": "Follows many more than followed, typical of engagement farming",
            "confidence": "MEDIUM"
        }
    
    # CASE 5: PROMOTIONAL / SCHEDULED
    elif (15 <= tpd <= 30 and age > 90 and features_dict.get('has_bio', 0) == 1):
        return {
            "behavior": "SCHEDULED_CONTENT",
            "risk_level": "LOW",
            "description": "Consistent posting pattern, possibly using scheduling tools",
            "confidence": "LOW"
        }
    
    # CASE 6: BOT-LIKE PATTERNS (Medium risk)
    elif ml_score > 0.7:
        return {
            "behavior": "BOT_LIKE_PATTERNS",
            "risk_level": "MEDIUM",
            "description": "Multiple behavioral indicators suggest automated patterns",
            "confidence": "MEDIUM"
        }
    
    # CASE 7: NORMAL BEHAVIOR
    else:
        return {
            "behavior": "NORMAL_BEHAVIOR",
            "risk_level": "LOW",
            "description": "Standard posting patterns with healthy engagement",
            "confidence": "HIGH"
        }

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
        probabilities = {}
        for model_name, model_obj in models.items():
            try:
                proba = model_obj.predict_proba(X_scaled)[0, 1]
                probabilities[model_name] = float(proba)
            except:
                probabilities[model_name] = 0.5

        # Ensemble prediction (weighted average)
        if 'weights' in model_data:
            weights = model_data['weights']
            weighted_sum = sum(probabilities[model] * weights.get(model, 0.25) 
                             for model in probabilities if model in weights)
            ensemble_proba = weighted_sum
        else:
            ensemble_proba = np.mean(list(probabilities.values()))

        # BEHAVIOR CLASSIFICATION (NOT identity)
        behavior_result = classify_behavior(features_dict, ensemble_proba)

        # Confidence based on agreement
        probs_list = list(probabilities.values())
        std_dev = np.std(probs_list) if len(probs_list) > 1 else 0
        confidence = "HIGH" if std_dev < 0.15 else "MEDIUM" if std_dev < 0.25 else "LOW"

        # Detailed behavioral indicators
        detailed_analysis = {
            "account_age_category": (
                "NEW (<30 days)" if features_dict['account_age_days'] < 30 else
                "YOUNG (1-6 months)" if features_dict['account_age_days'] < 180 else
                "MATURE (>6 months)"
            ),
            "activity_level": (
                "SAFE (<20/day)" if features_dict['activity_safe'] == 1 else
                "RISKY (20-50/day)" if features_dict['activity_risky'] == 1 else
                "EXTREME (50-100/day)" if features_dict['activity_extreme'] == 1 else
                "VERY EXTREME (>100/day)"
            ),
            "engagement_quality": (
                "GOOD" if features_dict['followers_per_tweet'] > 0.5 else
                "MODERATE" if features_dict['followers_per_tweet'] > 0.1 else
                "POOR"
            ),
            "bot_like_indicators": [],
            "normal_indicators": []
        }

        # Bot-like indicators
        if features_dict['activity_extreme'] == 1:
            detailed_analysis["bot_like_indicators"].append(
                f"Extreme posting rate: {features_dict['tweets_per_day']:.1f}/day"
            )
        if features_dict['high_following_ratio'] == 1:
            detailed_analysis["bot_like_indicators"].append(
                f"High following ratio: {features_dict['followers_following_ratio']:.2f}"
            )
        if features_dict['low_followers_high_tweets'] == 1:
            detailed_analysis["bot_like_indicators"].append(
                "Low followers but high tweet count"
            )
        if features_dict['high_activity_low_engagement'] == 1:
            detailed_analysis["bot_like_indicators"].append(
                "High activity with low engagement"
            )

        # Normal indicators
        if features_dict['account_maturity_score'] > 0.7:
            detailed_analysis["normal_indicators"].append(
                f"Mature account ({features_dict['account_age_days']} days)"
            )
        if features_dict['activity_safe'] == 1:
            detailed_analysis["normal_indicators"].append(
                f"Normal posting rate: {features_dict['tweets_per_day']:.1f}/day"
            )
        if features_dict['has_bio'] == 1 and features_dict['bio_length'] > 50:
            detailed_analysis["normal_indicators"].append(
                "Complete profile with detailed bio"
            )

        # BASE RESPONSE
        response_data = {
            "analysis_scope": "PROFILE_BEHAVIORAL_ANALYSIS",
            "behavior": behavior_result["behavior"],
            "risk_level": behavior_result["risk_level"],
            "confidence": confidence,
            "description": behavior_result["description"],
            "bot_probability": float(ensemble_proba),
            "model_type": model_type,
            "account_metrics": {
                "account_age_days": int(features_dict.get('account_age_days', 0)),
                "tweets_per_day": round(features_dict.get('tweets_per_day', 0), 2),
                "followers_following_ratio": round(features_dict.get('followers_following_ratio', 0), 2),
                "followers_per_tweet": round(features_dict.get('followers_per_tweet', 0), 3),
                "bot_risk_score": round(features_dict.get('bot_risk_score', 0), 2),
                "account_maturity_score": round(features_dict.get('account_maturity_score', 0), 2)
            },
            "detailed_analysis": detailed_analysis,
            "disclaimer": "This analysis detects behavioral patterns, not account identity. " +
                        "Many human accounts exhibit bot-like behaviors."
        }

        return response_data

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict-direct")
async def predict_direct(profile: dict):
    """
    Direct prediction from profile dictionary with behavioral analysis
    """
    try:
        features_dict = extract_features_from_profile(profile)
        X = pd.DataFrame([features_dict])[features]
        
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
        
        # Get probability
        proba = model.predict_proba(X_scaled)[0, 1]
        
        # Behavioral classification
        behavior_result = classify_behavior(features_dict, proba)
        
        return {
            "behavior": behavior_result["behavior"],
            "risk_level": behavior_result["risk_level"],
            "bot_probability": float(proba),
            "description": behavior_result["description"],
            "key_metrics": {
                "account_age_days": features_dict.get('account_age_days'),
                "tweets_per_day": round(features_dict.get('tweets_per_day', 0), 2),
                "followers_following_ratio": round(features_dict.get('followers_following_ratio', 0), 2)
            }
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
            
            # Behavioral classification
            behavior_result = classify_behavior(features_dict, proba)
            
            results.append({
                "username": profile.get('username', 'unknown'),
                "behavior": behavior_result["behavior"],
                "risk_level": behavior_result["risk_level"],
                "bot_probability": float(proba),
                "followers": profile.get('followers_count', 0),
                "account_age_days": features_dict.get('account_age_days', 0),
                "tweets_per_day": round(features_dict.get('tweets_per_day', 0), 2)
            })
            
        except Exception as e:
            results.append({
                "username": profile.get('username', 'unknown'),
                "error": str(e),
                "behavior": "ANALYSIS_ERROR"
            })
    
    # Summary statistics
    behaviors = [r.get('behavior', 'UNKNOWN') for r in results if 'behavior' in r]
    behavior_counts = {behavior: behaviors.count(behavior) for behavior in set(behaviors)}
    
    return {
        "total": len(results),
        "successful": len([r for r in results if 'error' not in r]),
        "behavior_distribution": behavior_counts,
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
        
        # Get predictions from all models
        probabilities = {}
        for model_name, model_obj in models.items():
            try:
                proba = model_obj.predict_proba(X_scaled)[0, 1]
                probabilities[model_name] = float(proba)
            except Exception:
                probabilities[model_name] = 0.5
        
        # Ensemble prediction
        if 'weights' in model_data:
            weights = model_data['weights']
            weighted_sum = sum(probabilities[model] * weights.get(model, 0.25) 
                             for model in probabilities if model in weights)
            ensemble_proba = weighted_sum
        else:
            ensemble_proba = np.mean(list(probabilities.values()))
        
        # Behavioral classification
        behavior_result = classify_behavior(features_dict, ensemble_proba)
        
        # Confidence
        probs_list = list(probabilities.values())
        std_dev = np.std(probs_list) if len(probs_list) > 1 else 0
        confidence = "HIGH" if std_dev < 0.15 else "MEDIUM" if std_dev < 0.25 else "LOW"
        
        return {
            "username": username,
            "behavior": behavior_result["behavior"],
            "risk_level": behavior_result["risk_level"],
            "confidence": confidence,
            "description": behavior_result["description"],
            "bot_probability": float(ensemble_proba),
            "model_consensus": {k: round(v, 3) for k, v in probabilities.items()},
            "profile_summary": {
                "followers": profile.get('followers_count', 0),
                "following": profile.get('following_count', 0),
                "tweets": profile.get('tweet_count', 0),
                "account_age_days": features_dict.get('account_age_days', 0),
                "tweets_per_day": round(features_dict.get('tweets_per_day', 0), 2)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "Twitter Behavioral Analysis API v5.0",
        "description": "Analyzes Twitter accounts for bot-like behavioral patterns",
        "model": model_type,
        "features": len(features),
        "models": list(models.keys()),
        "note": "This API detects behavioral patterns, not account identity"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_type != "DUMMY_FALLBACK",
        "model_type": model_type,
        "features_count": len(features),
        "analysis_type": "BEHAVIORAL_PATTERNS"
    }

@app.get("/model-info")
async def model_info():
    """Get detailed model information"""
    return {
        "model_type": model_type,
        "features_count": len(features),
        "has_scaler": scaler is not None,
        "models_available": list(models.keys()),
        "analysis_approach": "BEHAVIORAL_CLASSIFICATION",
        "behavior_categories": [
            "AUTOMATED_BEHAVIOR",
            "AGGRESSIVE_POSTING",
            "LOW_VISIBILITY_ACTIVITY",
            "AMPLIFIER_BEHAVIOR",
            "SCHEDULED_CONTENT",
            "BOT_LIKE_PATTERNS",
            "NORMAL_BEHAVIOR"
        ]
    }

@app.get("/behavior-examples")
async def behavior_examples():
    """Examples of different behavioral patterns"""
    return {
        "examples": [
            {
                "behavior": "AUTOMATED_BEHAVIOR",
                "description": "New account (<30 days) with extreme posting (>50/day) and poor engagement",
                "typical_metrics": {
                    "account_age_days": "< 30",
                    "tweets_per_day": "> 50",
                    "followers_following_ratio": "< 0.1"
                }
            },
            {
                "behavior": "AGGRESSIVE_POSTING",
                "description": "Established account with unusually high but consistent posting",
                "typical_metrics": {
                    "account_age_days": "> 180",
                    "tweets_per_day": "15-50",
                    "followers_per_tweet": "> 0.1"
                }
            },
            {
                "behavior": "NORMAL_BEHAVIOR",
                "description": "Standard posting patterns with healthy engagement ratios",
                "typical_metrics": {
                    "account_age_days": "> 90",
                    "tweets_per_day": "< 20",
                    "followers_following_ratio": "0.5-2.0"
                }
            }
        ],
        "note": "These are behavioral patterns, not identity labels. Many human accounts exhibit bot-like behaviors."
    }

if __name__ == "__main__":
    print("🚀 Starting Twitter Behavioral Analysis API...")
    print("📊 Analysis type: BEHAVIORAL PATTERNS")
    print("🎯 Detecting: Automated behaviors, not identity")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)