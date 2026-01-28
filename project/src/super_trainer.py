# super_trainer.py - TRAIN 85% ACCURACY MODEL
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import our enhanced components
from data_loader_v2 import BotDatasetLoader
from feature_engineer_v2 import UltimateFeatureEngineer

# ML models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

class SuperBotTrainer:
    """Trains ULTIMATE bot detection model with 85%+ accuracy target"""
    
    def __init__(self, model_dir="../models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.loader = BotDatasetLoader()
        self.engineer = UltimateFeatureEngineer()
        
    def train(self, config_path="config.json"):
        """Main training pipeline"""
        print("="*70)
        print("🤖 TRAINING ULTIMATE BOT DETECTOR (85%+ ACCURACY TARGET)")
        print("="*70)
        
        # STEP 1: Load MORE data
        print("\n📊 STEP 1: Loading enhanced datasets...")
        df = self.loader.load_all_datasets(config_path)
        
        if df.empty:
            raise Exception("❌ No data loaded!")
        
        # STEP 2: Extract SUPERIOR features
        print("\n🔧 STEP 2: Extracting 50+ powerful features...")
        df_features, feature_columns = self.engineer.extract_features(df)
        
        # STEP 3: Prepare data
        print("\n📈 STEP 3: Preparing training data...")
        X = df_features[feature_columns].copy()
        y = df_features['label'].astype(int)
        
        # Handle missing values
        X = X.fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # STEP 4: Train ADVANCED ensemble
        print("\n🔥 STEP 4: Training advanced ensemble...")
        models = self._create_advanced_ensemble(y_train)
        trained_models = self._train_ensemble(models, X_train, y_train, X_test, y_test)
        
        # STEP 5: Create stacking ensemble (SUPER ACCURATE!)
        print("\n🎯 STEP 5: Creating stacking ensemble...")
        final_model = self._create_stacking_ensemble(trained_models, X_train, y_train, X_test, y_test)
        
        # STEP 6: Comprehensive evaluation
        print("\n📊 STEP 6: Evaluating model performance...")
        performance = self._evaluate_model(final_model, X_test, y_test, trained_models, feature_columns)
        
        # STEP 7: Save everything
        print("\n💾 STEP 7: Saving ultimate model...")
        model_data = self._save_model(final_model, trained_models, scaler, feature_columns, performance, df_features)
        
        return model_data
    
    def _create_advanced_ensemble(self, y_train):
        """Create powerful ensemble of models"""
        
        # Calculate class weights
        total = len(y_train)
        pos = y_train.sum()
        neg = total - pos
        scale_pos_weight = neg / pos if pos > 0 else 1
        
        print(f"   ⚖️ Class weights: Humans={neg:,}, Bots={pos:,}, Weight={scale_pos_weight:.2f}")
        
        models = {
            'xgb_advanced': XGBClassifier(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                scale_pos_weight=scale_pos_weight,
                n_jobs=-1,
                verbosity=0,
                eval_metric='auc'
            ),
            
            'lgbm_advanced': LGBMClassifier(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1,
                verbosity=-1,
                boosting_type='gbdt'
            ),
            
            'catboost_advanced': CatBoostClassifier(
                iterations=500,
                depth=8,
                learning_rate=0.01,
                random_state=42,
                verbose=0,
                auto_class_weights='Balanced',
                thread_count=-1,
                loss_function='Logloss'
            ),
            
            'rf_advanced': RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                bootstrap=True
            ),
            
            # ADD NEURAL NETWORK for diversity!
            'mlp': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size=256,
                learning_rate='adaptive',
                max_iter=300,
                random_state=42,
                early_stopping=True
            )
        }
        
        return models
    
    def _train_ensemble(self, models, X_train, y_train, X_test, y_test):
        """Train all models in ensemble"""
        trained = {}
        
        print("   Training 5-model ensemble:")
        for name, model in models.items():
            print(f"     • {name}...", end='')
            model.fit(X_train, y_train)
            
            # Quick validation
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
            print(f" AUC: {auc:.4f}")
            
            trained[name] = model
        
        return trained
    
    def _create_stacking_ensemble(self, base_models, X_train, y_train, X_test, y_test):
        """Create stacking ensemble for maximum accuracy"""
        print("   Creating stacking ensemble...")
        
        # Get base model predictions
        base_predictions = []
        for name, model in base_models.items():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_train)[:, 1]
                base_predictions.append(proba)
        
        # Stack predictions
        X_stacked = np.column_stack(base_predictions)
        
        # Train meta-learner
        meta_learner = LogisticRegression(
            C=0.1,
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
        
        meta_learner.fit(X_stacked, y_train)
        
        # Evaluate stacking
        test_predictions = []
        for name, model in base_models.items():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_test)[:, 1]
                test_predictions.append(proba)
        
        X_test_stacked = np.column_stack(test_predictions)
        stacking_proba = meta_learner.predict_proba(X_test_stacked)[:, 1]
        stacking_auc = roc_auc_score(y_test, stacking_proba)
        
        print(f"     Stacking AUC: {stacking_auc:.4f}")
        
        # Return a function that combines all models
        def stacking_predictor(X):
            """Predict using stacking ensemble"""
            base_preds = []
            for name, model in base_models.items():
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)[:, 1]
                    base_preds.append(proba)
            
            X_stacked = np.column_stack(base_preds)
            return meta_learner.predict_proba(X_stacked)[:, 1]
        
        return stacking_predictor
    
    def _evaluate_model(self, model, X_test, y_test, base_models, feature_columns):
        """Comprehensive model evaluation"""
        print("   Running comprehensive evaluation...")
        
        # Get predictions
        if callable(model):  # Stacking predictor
            y_pred_proba = model(X_test)
        else:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba),
            'f1': f1_score(y_test, y_pred),
            'precision': self._precision_score(y_test, y_pred),
            'recall': self._recall_score(y_test, y_pred)
        }
        
        # Cross-validation
        print("     Running 5-fold cross-validation...")
        cv_scores = self._cross_validate(base_models['xgb_advanced'], X_test, y_test)
        metrics['cv_mean_auc'] = cv_scores.mean()
        metrics['cv_std_auc'] = cv_scores.std()
        
        # Print results
        print("\n" + "="*50)
        print("📊 ULTIMATE MODEL PERFORMANCE")
        print("="*50)
        print(f"🎯 Accuracy: {metrics['accuracy']:.4f}")
        print(f"📊 AUC Score: {metrics['auc']:.4f}")
        print(f"⭐ F1-Score: {metrics['f1']:.4f}")
        print(f"🎯 Precision: {metrics['precision']:.4f}")
        print(f"📈 Recall: {metrics['recall']:.4f}")
        print(f"🔍 CV AUC: {metrics['cv_mean_auc']:.4f} (±{metrics['cv_std_auc']:.4f})")
        
        if metrics['accuracy'] > 0.85:
            print("\n✅ TARGET ACHIEVED! 85%+ ACCURACY! 🎉")
        elif metrics['accuracy'] > 0.80:
            print("\n⚠️ Close! Need minor improvements to reach 85%")
        else:
            print("\n❌ Need significant improvements")
        
        return metrics
    
    def _precision_score(self, y_true, y_pred):
        """Calculate precision"""
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        return tp / (tp + fp) if (tp + fp) > 0 else 0
    
    def _recall_score(self, y_true, y_pred):
        """Calculate recall"""
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        return tp / (tp + fn) if (tp + fn) > 0 else 0
    
    def _cross_validate(self, model, X, y):
        """Run cross-validation"""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        return scores
    
    def _save_model(self, final_model, base_models, scaler, features, performance, df_features):
        """Save complete model package"""
        model_path = os.path.join(self.model_dir, "ultimate_bot_detector_v4.pkl")
        
        model_data = {
            'final_model': final_model,  # Stacking predictor
            'base_models': base_models,  # Individual models
            'scaler': scaler,
            'features': features,
            'feature_count': len(features),
            'performance': performance,
            'model_type': 'ULTIMATE_V4_STACKING_ENSEMBLE',
            'training_date': pd.Timestamp.now().isoformat(),
            'training_info': {
                'total_samples': len(df_features),
                'bot_samples': df_features['label'].sum(),
                'human_samples': len(df_features) - df_features['label'].sum(),
                'bot_percentage': df_features['label'].mean(),
                'feature_engineer': 'UltimateFeatureEngineer_v2'
            },
            'prediction_function': self._create_prediction_function(final_model, base_models, scaler, features)
        }
        
        joblib.dump(model_data, model_path, compress=3)
        
        print(f"\n💾 Model saved to: {model_path}")
        print(f"📊 File size: {os.path.getsize(model_path) / 1024 / 1024:.1f} MB")
        print(f"🔧 Features: {len(features)}")
        print(f"🤖 Models: {list(base_models.keys())} + Stacking")
        
        return model_data
    
    def _create_prediction_function(self, final_model, base_models, scaler, features):
        """Create a clean prediction function"""
        def predict(profile_dict):
            """Predict bot probability for a single profile"""
            # Extract features
            engineered = self.engineer.extract_single_profile(profile_dict)
            
            # Create feature vector
            X = pd.DataFrame([engineered])
            X = X[features].fillna(0)
            
            # Scale
            X_scaled = scaler.transform(X)
            
            # Predict
            if callable(final_model):
                proba = final_model(X_scaled)[0]
            else:
                proba = final_model.predict_proba(X_scaled)[0, 1]
            
            # Get base model predictions for consensus
            base_predictions = {}
            for name, model in base_models.items():
                if hasattr(model, 'predict_proba'):
                    base_predictions[name] = float(model.predict_proba(X_scaled)[0, 1])
            
            return {
                'probability': float(proba),
                'prediction': 'BOT' if proba > 0.5 else 'HUMAN',
                'base_model_predictions': base_predictions,
                'consensus': np.mean(list(base_predictions.values())),
                'features_used': len(features)
            }
        
        return predict

# Main training function
def train_ultimate_model():
    """Train the ultimate bot detector"""
    print("\n" + "="*70)
    print("🚀 ULTIMATE BOT DETECTOR TRAINING SYSTEM")
    print("="*70)
    print("\n🎯 TARGET: 85%+ Accuracy")
    print("📊 Features: 50+ engineered features")
    print("🤖 Models: XGBoost + LightGBM + CatBoost + RandomForest + Neural Net")
    print("🎯 Strategy: Stacking Ensemble\n")
    
    trainer = SuperBotTrainer()
    
    try:
        model_data = trainer.train()
        
        print("\n" + "="*70)
        print("🎉 TRAINING COMPLETE!")
        print("="*70)
        
        perf = model_data['performance']
        print(f"\n📋 FINAL RESULTS:")
        print(f"   Accuracy: {perf['accuracy']:.3f}")
        print(f"   AUC: {perf['auc']:.3f}")
        print(f"   F1-Score: {perf['f1']:.3f}")
        print(f"   Cross-Validation AUC: {perf['cv_mean_auc']:.3f} (±{perf['cv_std_auc']:.3f})")
        
        if perf['accuracy'] > 0.85:
            print("\n✅ SUCCESS! 85%+ ACCURACY ACHIEVED! 🎉🎉🎉")
        else:
            print(f"\n⚠️ Accuracy: {perf['accuracy']:.1%} - Close to target!")
        
        return model_data
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    model_data = train_ultimate_model()