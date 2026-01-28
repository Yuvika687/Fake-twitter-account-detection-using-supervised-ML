import os
import json
import pandas as pd
import numpy as np
from data_utils import unify

def load_twibot22_simple(base_path):
    """Load TwiBot-22 dataset without PyTorch"""
    print("📂 Loading TwiBot-22 dataset...")
    
    try:
        # Check if main files exist
        users_path = os.path.join(base_path, "users.json")
        labels_path = os.path.join(base_path, "labels.json")
        
        if not os.path.exists(users_path) or not os.path.exists(labels_path):
            print("❌ TwiBot-22 main files not found. Skipping...")
            return pd.DataFrame()
        
        # Load user data
        with open(users_path, 'r', encoding='utf-8') as f:
            users_data = json.load(f)
        
        # Load labels
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels_data = json.load(f)
        
        # Convert to DataFrame
        users_list = []
        for user_id, user_info in users_data.items():
            if isinstance(user_info, dict):
                user_info['user_id'] = user_id
                users_list.append(user_info)
        
        if not users_list:
            print("❌ No valid user data in TwiBot-22")
            return pd.DataFrame()
            
        df = pd.DataFrame(users_list)
        
        # Map labels to DataFrame
        label_map = {}
        for user_id, label in labels_data.items():
            if isinstance(label, str):
                label_map[user_id] = 1 if label.lower() == "bot" else 0
            else:
                label_map[user_id] = int(label)
                
        df['label'] = df['user_id'].map(label_map)
        
        # Remove rows with missing labels
        df = df.dropna(subset=['label'])
        
        # Standardize column names
        df = unify(df)
        
        print(f"✅ TwiBot-22 loaded: {len(df)} users")
        return df
        
    except Exception as e:
        print(f"❌ Error loading TwiBot-22: {e}")
        return pd.DataFrame()

def load_mgstbot_simple(base_path):
    """Skip MGStBot for now since it requires PyTorch"""
    print("⚠️ MGStBot requires PyTorch. Skipping for now...")
    return pd.DataFrame()

def load_cresci15_simple(base_path):
    """Load Cresci-15 dataset without PyTorch"""
    print("📂 Loading Cresci-15 dataset...")
    
    try:
        if not os.path.exists(base_path):
            print("❌ Cresci-15 path not found. Skipping...")
            return pd.DataFrame()
            
        # Look for CSV files
        csv_files = [f for f in os.listdir(base_path) if f.endswith('.csv')]
        
        if not csv_files:
            print("❌ No CSV files found in Cresci-15 directory")
            return pd.DataFrame()
        
        all_dfs = []
        for csv_file in csv_files:
            try:
                csv_path = os.path.join(base_path, csv_file)
                df_temp = pd.read_csv(csv_path)
                df_temp = unify(df_temp)
                
                # Ensure label column exists
                if 'label' not in df_temp.columns:
                    print(f"⚠️ No label column in {csv_file}, skipping...")
                    continue
                    
                all_dfs.append(df_temp)
                print(f"✅ Loaded {csv_file}: {len(df_temp)} rows")
                
            except Exception as e:
                print(f"❌ Error loading {csv_file}: {e}")
                continue
        
        if all_dfs:
            df = pd.concat(all_dfs, ignore_index=True)
            print(f"✅ Cresci-15 loaded: {len(df)} total users")
            return df
        else:
            print("❌ No valid CSV files could be loaded from Cresci-15")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Error loading Cresci-15: {e}")
        return pd.DataFrame()

def load_all_advanced_simple(config_path):
    """Load ALL datasets including advanced ones without PyTorch"""
    
    try:
        with open(config_path, "r") as f:
            paths = json.load(f)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return pd.DataFrame()
    
    all_dfs = []
    
    # 1. Original datasets (always try this first)
    try:
        from data_utils import load_all as load_original
        df_original = load_original(config_path)
        if not df_original.empty:
            all_dfs.append(df_original)
            print(f"✅ Original datasets: {len(df_original)} users")
        else:
            print("❌ Original datasets returned empty")
    except Exception as e:
        print(f"❌ Error loading original datasets: {e}")
    
    # 2. Try advanced datasets (without PyTorch dependencies)
    advanced_datasets = [
        ("TwiBot-22", "twibot22_path", load_twibot22_simple),
        ("Cresci-15", "cresci15_path", load_cresci15_simple)
    ]
    
    for name, path_key, loader_func in advanced_datasets:
        if path_key in paths:
            try:
                df_advanced = loader_func(paths[path_key])
                if not df_advanced.empty:
                    all_dfs.append(df_advanced)
                    print(f"✅ {name}: {len(df_advanced)} users")
                else:
                    print(f"⚠️ {name}: No data loaded")
            except Exception as e:
                print(f"❌ Error loading {name}: {e}")
        else:
            print(f"⚠️ {name} path not in config")
    
    # Merge all datasets
    if all_dfs:
        try:
            merged = pd.concat(all_dfs, ignore_index=True, sort=False)
            
            # Remove duplicates
            if 'user_id' in merged.columns:
                merged = merged.drop_duplicates(subset=['user_id'], keep='first')
            
            # Remove duplicate columns
            merged = merged.loc[:, ~merged.columns.duplicated()]
            
            # Keep only labeled rows
            if 'label' in merged.columns:
                merged = merged.dropna(subset=['label'])
                # Ensure labels are 0/1
                merged['label'] = merged['label'].apply(lambda x: 1 if str(x).lower() in ['1', 'true', 'bot'] else 0)
            
            print(f"🎉 FINAL MERGED DATASET: {len(merged)} users")
            if 'label' in merged.columns:
                print(f"📊 Label distribution:")
                print(merged['label'].value_counts())
            
            return merged
            
        except Exception as e:
            print(f"❌ Error merging datasets: {e}")
            return pd.DataFrame()
    else:
        print("❌ No datasets could be loaded at all!")
        return pd.DataFrame()