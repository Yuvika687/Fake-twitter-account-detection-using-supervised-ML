# data_loader_v2.py - SIMPLE WORKING VERSION
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

class BotDatasetLoader:
    """Loads bot datasets - SIMPLIFIED VERSION"""
    
    def __init__(self):
        self.datasets_loaded = []
        
    def load_all_datasets(self, config_path="../config.json"):
        """Load datasets - FOCUS ON WHAT WORKS"""
        print("="*60)
        print("🚀 LOADING BOT DETECTION DATASETS")
        print("="*60)
        
        try:
            with open(config_path, "r") as f:
                paths = json.load(f)
        except:
            print("⚠️ Config not found, using default paths")
            paths = {
                "twibot20_train": "../data/archive (1)/train.json",
                "twibot20_dev": "../data/archive (1)/dev.json", 
                "twibot20_test": "../data/archive (1)/test.json"
            }
        
        all_dataframes = []
        
        # 1. LOAD TWIBOT-20 (THIS WORKS!)
        print("\n📥 1. Loading TwiBot-20...")
        df_twibot = self._load_twibot20_simple(paths)
        if not df_twibot.empty:
            all_dataframes.append(df_twibot)
            print(f"   ✅ {len(df_twibot)} users | {df_twibot['label'].mean():.1%} bots")
        
        # 2. TRY KAGGLE (SKIP IF PROBLEMS)
        print("\n📥 2. Trying Kaggle dataset...")
        df_kaggle = self._load_kaggle_simple(paths)
        if not df_kaggle.empty:
            all_dataframes.append(df_kaggle)
            print(f"   ✅ {len(df_kaggle)} users | {df_kaggle['label'].mean():.1%} bots")
        else:
            print("   ⚠️ Skipping Kaggle dataset")
        
        # MERGE WHAT WE HAVE
        print("\n" + "="*60)
        print("🔄 MERGING DATASETS")
        print("="*60)
        
        if all_dataframes:
            merged = pd.concat(all_dataframes, ignore_index=True, sort=False)
            
            # Basic cleaning
            merged = self._clean_dataset(merged)
            
            print(f"\n🎉 FINAL DATASET STATS:")
            print(f"   Total users: {len(merged):,}")
            print(f"   Bot percentage: {merged['label'].mean():.1%}")
            print(f"   Human: {len(merged[merged['label']==0]):,}")
            print(f"   Bot: {len(merged[merged['label']==1]):,}")
            
            # Save
            os.makedirs("../data/processed", exist_ok=True)
            merged.to_csv("../data/processed/combined_dataset.csv", index=False)
            print(f"💾 Saved to: ../data/processed/combined_dataset.csv")
            
            return merged
        else:
            raise Exception("❌ No data loaded!")
    
    def _load_twibot20_simple(self, paths):
        """Simple TwiBot-20 loader"""
        import json
        
        data_paths = [
            paths.get("twibot20_train", "../data/archive (1)/train.json"),
            paths.get("twibot20_dev", "../data/archive (1)/dev.json"),
            paths.get("twibot20_test", "../data/archive (1)/test.json")
        ]
        
        all_data = []
        for path in data_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    for user in data:
                        profile = user.get('profile', {})
                        label = user.get('label', '0')
                        
                        all_data.append({
                            'user_id': str(user.get('id', '')),
                            'followers_count': int(profile.get('followers_count', 0)),
                            'following_count': int(profile.get('friends_count', 0)),
                            'tweet_count': int(profile.get('statuses_count', 0)),
                            'verified': bool(profile.get('verified', False)),
                            'description': profile.get('description', ''),
                            'created_at': profile.get('created_at', ''),
                            'label': 1 if str(label) == '1' else 0
                        })
                    
                except Exception as e:
                    print(f"   ⚠️ Error loading {path}: {e}")
        
        return pd.DataFrame(all_data)
    
    def _load_kaggle_simple(self, paths):
        """Simple Kaggle loader - skip if problems"""
        try:
            if "kaggle_bot_dataset" not in paths:
                return pd.DataFrame()
            
            path = paths["kaggle_bot_dataset"]
            if not os.path.exists(path):
                print(f"   ⚠️ Kaggle file not found: {path}")
                return pd.DataFrame()
            
            df = pd.read_csv(path)
            print(f"   Found Kaggle with columns: {list(df.columns)}")
            
            # Try to identify label column
            label_col = None
            for col in df.columns:
                if 'bot' in col.lower():
                    label_col = col
                    break
            
            if label_col:
                df['label'] = df[label_col].apply(
                    lambda x: 1 if str(x).lower() in ['1', 'true', 'bot', 'fake'] else 0
                )
                
                # Try to get basic columns
                if 'followers' in df.columns.str.lower().any():
                    df['followers_count'] = df[[c for c in df.columns if 'follower' in c.lower()][0]]
                if 'following' in df.columns.str.lower().any() or 'friends' in df.columns.str.lower().any():
                    col = [c for c in df.columns if 'following' in c.lower() or 'friends' in c.lower()][0]
                    df['following_count'] = df[col]
                
                return df[['followers_count', 'following_count', 'label']].dropna()
            else:
                print("   ⚠️ No bot label found in Kaggle dataset")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"   ❌ Kaggle error: {e}")
            return pd.DataFrame()
    
    def _clean_dataset(self, df):
        """Clean dataset"""
        # Ensure required columns
        required = ['followers_count', 'following_count', 'tweet_count', 'label']
        for col in required:
            if col not in df.columns:
                if col == 'tweet_count':
                    df[col] = 0
                elif col in ['followers_count', 'following_count']:
                    df[col] = df.get(col, 0)
        
        # Fill NaN
        df = df.fillna({
            'followers_count': 0,
            'following_count': 0,
            'tweet_count': 0,
            'verified': False,
            'description': '',
            'created_at': '',
            'label': 0
        })
        
        # Remove extreme outliers
        df = df[df['followers_count'] <= 10000000]
        df = df[df['following_count'] <= 50000]
        df = df[df['tweet_count'] <= 100000]
        
        return df

# Quick test
if __name__ == "__main__":
    print("🧪 TESTING DATA LOADER...")
    loader = BotDatasetLoader()
    
    try:
        df = loader.load_all_datasets()
        print(f"\n✅ SUCCESS! Loaded {len(df)} users")
        print(f"📊 Sample data:")
        print(df[['followers_count', 'following_count', 'tweet_count', 'label']].head())
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n⚠️ Falling back to TwiBot-20 only...")
        
        # Try just TwiBot-20
        import json
        paths = {
            "twibot20_train": "../../data/archive (1)/train.json",
            "twibot20_dev": "../../data/archive (1)/dev.json",
            "twibot20_test": "../../data/archive (1)/test.json"
        }
        df = loader._load_twibot20_simple(paths)
        print(f"\n✅ Loaded TwiBot-20: {len(df)} users")
        print(f"📊 Bot percentage: {df['label'].mean():.1%}")