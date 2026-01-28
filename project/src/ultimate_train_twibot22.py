import os
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from simple_data_utils import load_twibot22_dataset, verify_twibot22_labels, get_training_data
from features import extract_features

# PATHS
TWIBOT22_PATH = "A:/Supervised learning/Data/TwiBot-22-master/TwiBot-22-master"
MODEL_DIR = "..\\models"
MODEL_PATH = os.path.join(MODEL_DIR, "ultimate_twibot22_model.pkl")

def train_twibot22_model():
    print("🚀 TWIBOT-22 ULTIMATE MODEL TRAINING STARTED...")
    
    # 1. Load ONLY TwiBot-22 dataset
    df = load_twibot22_dataset(TWIBOT22_PATH)
    verify_twibot22_labels(df)
    
    # 2. Use only training split (as per original paper)
    train_df = get_training_data(df)
    
    # 3. Extract features
    print("\n🔧 Extracting features...")
    train_df, feature_columns = extract_features(train_df)
    
    print(f"📊 Features: {len(feature_columns)}")
    print(f"📊 Feature names: {feature_columns}")
    
    # 4. Prepare data
    X = train_df[feature_columns].fillna(0)
    y = train_df['label'].astype(int)
    
    print(f"📊 X shape: {X.shape}, y shape: {y.shape}")
    print(f"📊 Label distribution: {y.value_counts().to_dict()}")
    
    # 5. Split data (train/validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 6. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 7. Train optimized XGBoost
    print("\n🤖 Training XGBoost...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        random_state=42,
        eval_metric=['logloss', 'auc', 'error'],
        reg_alpha=0.1,
        reg_lambda=0.1
    )
    
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=50
    )
    
    # 8. Evaluate
    print("\n📈 Evaluating model...")
    y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    accuracy = accuracy_score(y_val, y_pred)
    auc_score = roc_auc_score(y_val, y_pred_proba)
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"🔥 Accuracy: {accuracy:.4f}")
    print(f"📊 AUC Score: {auc_score:.4f}")
    print(f"\n📋 Classification Report:")
    print(classification_report(y_val, y_pred, target_names=['Human', 'Bot']))
    
    # 9. Save everything
    print("💾 Saving model...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    model_data = {
        'model': model,
        'features': feature_columns,
        'scaler': scaler,
        'model_type': 'ULTIMATE_TWIBOT22_XGBOOST',
        'performance': {
            'accuracy': accuracy,
            'auc_score': auc_score
        },
        'dataset_info': {
            'name': 'TwiBot-22',
            'samples': len(train_df),
            'bot_percentage': train_df['label'].mean()*100
        }
    }
    
    joblib.dump(model_data, MODEL_PATH)
    print(f"✅ Model saved to: {MODEL_PATH}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n📊 Top 10 Feature Importance:")
    print(feature_importance.head(10))
    
    return model_data

if __name__ == "__main__":
    train_twibot22_model()