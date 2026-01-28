import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier
import lightgbm as lgb
from advanced_data_utils import load_all_advanced
from super_features import extract_super_features

# PATHS
CONFIG_PATH = "..\\datasets_config.json"
MODEL_DIR = "..\\models"
MODEL_PATH = os.path.join(MODEL_DIR, "ultimate_model.pkl")

def ultimate_train():
    print("🚀 LOADING ULTIMATE DATASET COLLECTION...")
    df = load_all_advanced(CONFIG_PATH)
    
    if df.empty:
        print("❌ No data loaded. Check your dataset paths.")
        return
    
    print(f"📊 Total dataset size: {df.shape}")
    print(f"🎯 Label distribution:\n{df['label'].value_counts()}")
    
    print("🔧 EXTRACTING SUPER FEATURES...")
    df, feat_cols = extract_super_features(df)
    
    print(f"✅ Features extracted: {len(feat_cols)} features")
    print(f"📋 Features: {feat_cols}")
    
    X = df[feat_cols]
    y = df["label"].astype(int)
    
    print("📊 Feature statistics:")
    print(X.describe())

    print("🎯 SPLITTING DATA...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # Feature selection - keep best features
    print("🔍 SELECTING BEST FEATURES...")
    selector = SelectKBest(f_classif, k=min(20, len(feat_cols)))
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    selected_features = [feat_cols[i] for i in selector.get_support(indices=True)]
    print(f"🎯 Selected {len(selected_features)} best features: {selected_features}")

    # Robust scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    print("🤖 TRAINING ENSEMBLE OF MODELS...")
    
    # Define multiple advanced models
    xgb = XGBClassifier(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    lgbm = lgb.LGBMClassifier(
        n_estimators=1000,
        max_depth=7,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    # Ensemble with weighted voting
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb),
            ('lgbm', lgbm),
            ('rf', rf)
        ],
        voting='soft',
        weights=[3, 2, 2]  # XGBoost gets highest weight
    )
    
    # Train individual models
    print("🔄 Training XGBoost...")
    xgb.fit(X_train_scaled, y_train)
    
    print("🔄 Training LightGBM...")
    lgbm.fit(X_train_scaled, y_train)
    
    print("🔄 Training Random Forest...")
    rf.fit(X_train_scaled, y_train)
    
    print("🔄 Training Ensemble...")
    ensemble.fit(X_train_scaled, y_train)

    # Evaluate all models
    models = {
        'XGBoost': xgb,
        'LightGBM': lgbm, 
        'Random Forest': rf,
        'ENSEMBLE': ensemble
    }
    
    print("\n📊 MODEL PERFORMANCE COMPARISON:")
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n{name}:")
        print(f"  ✅ Accuracy: {acc:.4f}")
        print(f"  ✅ AUC: {auc:.4f}")
        
        if auc > best_score:
            best_score = auc
            best_model = model

    print(f"\n🎯 BEST MODEL: {best_model.__class__.__name__} (AUC: {best_score:.4f})")

    # Final evaluation with best model
    final_preds = best_model.predict(X_test_scaled)
    final_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    print(f"\n📊 FINAL CLASSIFICATION REPORT:")
    print(classification_report(y_test, final_preds))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, final_preds)
    print(f"📊 Confusion Matrix:\n{cm}")

    print("\n💾 SAVING ULTIMATE MODEL...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({
        "model": best_model,
        "features": selected_features,
        "scaler": scaler,
        "selector": selector,
        "all_features": feat_cols,
        "performance": {
            "accuracy": accuracy_score(y_test, final_preds),
            "auc": best_score
        }
    }, MODEL_PATH)

    print(f"✅ ULTIMATE model saved at: {MODEL_PATH}")
    print(f"🎉 Training complete! Expected accuracy: {best_score*100:.1f}%")
    
    return best_score

if __name__ == "__main__":
    # Install lightgbm if not present: pip install lightgbm
    score = ultimate_train()
    print(f"\n🚀 ULTIMATE TRAINING COMPLETE! Best AUC: {score:.4f}")