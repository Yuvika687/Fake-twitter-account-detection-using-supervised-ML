import os
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from fixed_data_utils import load_clean_data, verify_labels
from features import extract_features

# PATHS
CONFIG_PATH = "..\\datasets_config.json"
MODEL_DIR = "..\\models"
MODEL_PATH = os.path.join(MODEL_DIR, "ultimate_model.pkl")

def train_ultimate_model():
    print("🚀 ULTIMATE MODEL TRAINING STARTED...")
    
    # 1. Load data with FIXED labels
    print("📥 Step 1: Loading data...")
    df = load_clean_data(CONFIG_PATH)
    verify_labels(df)
    
    # 2. Extract features
    print("🔧 Step 2: Extracting features...")
    df, feature_columns = extract_features(df)
    
    print(f"📊 Features: {len(feature_columns)}")
    print(f"📊 Feature names: {feature_columns}")
    
    # 3. Prepare data
    X = df[feature_columns].fillna(0)
    y = df['label'].astype(int)
    
    print(f"📊 X shape: {X.shape}, y shape: {y.shape}")
    print(f"📊 Label distribution: {y.value_counts().to_dict()}")
    
    # 4. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Train model with better parameters
    print("🤖 Step 3: Training XGBoost...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        random_state=42,
        eval_metric=['logloss', 'auc', 'error']
    )
    
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=50
    )
    
    # 7. Evaluate
    print("📈 Step 4: Evaluating model...")
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"🔥 Accuracy: {accuracy:.4f}")
    print(f"📊 AUC Score: {auc_score:.4f}")
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Human', 'Bot']))
    
    # 8. Save everything
    print("💾 Step 5: Saving model...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    model_data = {
        'model': model,
        'features': feature_columns,
        'scaler': scaler,
        'model_type': 'ULTIMATE_XGBOOST',
        'performance': {
            'accuracy': accuracy,
            'auc_score': auc_score
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

if __name__ == "__main__":
    train_ultimate_model()