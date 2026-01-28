import os
import json
import pandas as pd
import numpy as np

def load_json_fixed(path):
    """Load JSON with proper error handling"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return pd.json_normalize(data)
    except Exception as e:
        print(f"❌ Error loading {path}: {e}")
        return pd.DataFrame()

def unify_labels_fixed(df):
    """FIXED: Correct label mapping for ALL datasets"""
    mapping = {}
    
    # Standardize column names
    for c in df.columns:
        lc = c.lower()
        
        if "user_id" == lc or (("id" in lc) and ("user" in lc)):
            mapping[c] = "user_id"
        elif "username" in lc or "screen_name" in lc:
            mapping[c] = "username"
        elif "created" in lc:
            mapping[c] = "created_at"
        elif "followers" in lc:
            mapping[c] = "followers_count"
        elif "following" in lc or "friends" in lc:
            mapping[c] = "following_count"
        elif "tweet" in lc or "status" in lc:
            mapping[c] = "tweet_count"
        elif "verified" in lc:
            mapping[c] = "verified"
        elif "description" in lc or "bio" in lc:
            mapping[c] = "description"
        elif "label" in lc or "bot" in lc or "class" in lc:
            mapping[c] = "label"
    
    df = df.rename(columns=mapping)
    
    # FIXED: Correct label mapping
    if "label" in df.columns:
        def map_label(x):
            x_str = str(x).lower().strip()
            # Map to: 0 = Human, 1 = Bot
            if x_str in ['bot', '1', 'true', 'fake', 'spam']:
                return 1
            elif x_str in ['human', '0', 'false', 'real', 'genuine']:
                return 0
            else:
                return np.nan  # Invalid labels
        
        df["label"] = df["label"].apply(map_label)
    
    return df

def safe_concat_dataframes(df_list):
    """Safely concatenate dataframes with different columns"""
    if not df_list:
        return pd.DataFrame()
    
    # Get all unique columns from all dataframes
    all_columns = set()
    for df in df_list:
        all_columns.update(df.columns)
    
    # Ensure all dataframes have the same columns
    standardized_dfs = []
    for df in df_list:
        # Add missing columns with default values
        for col in all_columns:
            if col not in df.columns:
                df[col] = np.nan  # or 0 for numeric columns
        
        # Reorder columns to be consistent
        df = df[list(all_columns)]
        standardized_dfs.append(df)
    
    # Now concatenate
    return pd.concat(standardized_dfs, ignore_index=True)

def load_clean_data(config_path):
    """Load and clean ALL datasets with FIXED labels"""
    with open(config_path, "r") as f:
        paths = json.load(f)
    
    all_dfs = []
    
    # 1. Load TwiBot datasets (FIXED)
    print("📥 Loading TwiBot datasets...")
    for dataset_name in ['twibot20_train', 'twibot20_dev', 'twibot20_test']:
        path = paths[dataset_name]
        df = load_json_fixed(path)
        if not df.empty:
            df = unify_labels_fixed(df)
            all_dfs.append(df)
            print(f"   ✅ {dataset_name}: {len(df)} rows, {df['label'].sum()} bots")
    
    # 2. Load Kaggle dataset
    print("📥 Loading Kaggle dataset...")
    try:
        df_kaggle = pd.read_csv(paths["kaggle_bot_dataset"])
        df_kaggle = unify_labels_fixed(df_kaggle)
        all_dfs.append(df_kaggle)
        print(f"   ✅ Kaggle: {len(df_kaggle)} rows, {df_kaggle['label'].sum()} bots")
    except Exception as e:
        print(f"   ❌ Kaggle load failed: {e}")
    
    # 3. Merge everything SAFELY
    if all_dfs:
        merged = safe_concat_dataframes(all_dfs)
        
        # Remove duplicates
        if 'user_id' in merged.columns:
            # Remove rows where user_id is NaN before deduplication
            merged = merged.dropna(subset=['user_id'])
            merged = merged.drop_duplicates(subset=['user_id'])
        
        # Remove rows with invalid labels
        merged = merged.dropna(subset=['label'])
        
        print(f"\n📊 FINAL DATASET:")
        print(f"   Total rows: {len(merged)}")
        print(f"   Bots: {merged['label'].sum()}")
        print(f"   Humans: {len(merged) - merged['label'].sum()}")
        print(f"   Bot percentage: {merged['label'].mean()*100:.1f}%")
        
        return merged
    else:
        raise Exception("❌ No data loaded!")

def verify_labels(df):
    """Verify label distribution"""
    if 'label' not in df.columns:
        print("❌ No label column found!")
        return
    
    print("\n🔍 LABEL VERIFICATION:")
    print(f"   Unique values: {df['label'].unique()}")
    print(f"   Value counts:\n{df['label'].value_counts().sort_index()}")
    
    # Check sample labels
    sample_cols = [col for col in ['user_id', 'username', 'label'] if col in df.columns]
    if sample_cols:
        sample = df[sample_cols].head(10)
        print(f"   Sample data:\n{sample}")
        