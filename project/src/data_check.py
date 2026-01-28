# data_check.py
from fixed_data_utils import load_clean_data, verify_labels
import pandas as pd

# Load your data
print("🔍 CHECKING DATA LOADING...")
df = load_clean_data("path/to/your/config.json")

# Check labels
verify_labels(df)

# Check feature distributions
print("\n📊 FEATURE DISTRIBUTIONS:")
if 'followers_count' in df.columns:
    print(f"Followers: min={df['followers_count'].min()}, max={df['followers_count'].max()}")
if 'tweet_count' in df.columns:
    print(f"Tweets: min={df['tweet_count'].min()}, max={df['tweet_count'].max()}")
if 'account_age_days' in df.columns:
    print(f"Account Age: min={df['account_age_days'].min()}, max={df['account_age_days'].max()}")

# Check if features make sense for bot detection
print("\n🎯 BOT/HUMAN COMPARISON:")
if 'label' in df.columns:
    bots = df[df['label'] == 1]
    humans = df[df['label'] == 0]
    
    print(f"\n🤖 BOTS (n={len(bots)}):")
    print(f"  Avg followers: {bots['followers_count'].mean():.0f}")
    print(f"  Avg tweets/day: {(bots['tweet_count'] / bots['account_age_days'].clip(1)).mean():.1f}")
    
    print(f"\n👤 HUMANS (n={len(humans)}):")
    print(f"  Avg followers: {humans['followers_count'].mean():.0f}")
    print(f"  Avg tweets/day: {(humans['tweet_count'] / humans['account_age_days'].clip(1)).mean():.1f}")