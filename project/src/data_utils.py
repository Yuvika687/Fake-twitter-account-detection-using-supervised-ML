import os
import json
import pandas as pd
import ast

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return pd.json_normalize(json.load(f))

def load_twibot(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.json_normalize(data)

# ---------------------------------------------
# Extract CSV from GitHub Notebook (.ipynb)
# ---------------------------------------------
def load_github_notebook_csv(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except Exception as e:
        raise Exception(f"Error reading notebook: {e}")

    extracted_df = None

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        source_code = "".join(cell.get("source", []))

        # 1) pd.DataFrame({...})
        if "DataFrame" in source_code and "{" in source_code and "}" in source_code:
            try:
                dict_str = source_code.split("DataFrame(", 1)[1]
                dict_str = dict_str.split(")", 1)[0]
                parsed = ast.literal_eval(dict_str)
                df = pd.DataFrame(parsed)
                extracted_df = df
                break
            except:
                pass

        # 2) Dictionary literal assigned e.g. x = { ... }
        if "{" in source_code and "}" in source_code:
            try:
                dict_str = source_code[source_code.index("{"): source_code.rindex("}") + 1]
                parsed = ast.literal_eval(dict_str)

                if isinstance(parsed, dict):
                    df = pd.DataFrame(parsed)
                    extracted_df = df
                    break
            except:
                pass

    if extracted_df is None:
        raise Exception("No DataFrame found inside the notebook.")

    # Save CSV for future runs
    csv_path = path.replace(".ipynb", ".csv")
    extracted_df.to_csv(csv_path, index=False)
    print("Extracted CSV saved to:", csv_path)

    return extracted_df

# ---------------------------------------------
# Standardize columns across datasets
# ---------------------------------------------
def unify(df):
    mapping = {}

    for c in df.columns:
        lc = c.lower()

        if "user_id" == lc or (("id" in lc) and ("user" in lc)):
            mapping[c] = "user_id"

        if "username" in lc or "screen_name" in lc:
            mapping[c] = "username"

        if "created" in lc:
            mapping[c] = "created_at"

        if "followers" in lc:
            mapping[c] = "followers_count"

        if "following" in lc or "friends" in lc:
            mapping[c] = "following_count"

        if "tweet" in lc or "status" in lc:
            mapping[c] = "tweet_count"

        if "verified" in lc:
            mapping[c] = "verified"

        if "description" in lc or "bio" in lc:
            mapping[c] = "description"

        if "label" in lc or "bot" in lc:
            mapping[c] = "label"

    df = df.rename(columns=mapping)

    if "label" in df.columns:
        df["label"] = df["label"].apply(
            lambda x: 1 if str(x).lower() in ["bot", "fake", "1", "true"] else 0
        )

    return df

# ---------------------------------------------
# Load ALL datasets
# ---------------------------------------------
def load_all(config_path):

    with open(config_path, "r") as f:
        paths = json.load(f)

    # 1) Kaggle dataset
    df_kaggle = pd.read_csv(paths["kaggle_bot_dataset"])
    df_kaggle = unify(df_kaggle)

    # Remove duplicate columns
    df_kaggle = df_kaggle.loc[:, ~df_kaggle.columns.duplicated()]

    # 2) TwiBot dataset
    df_twibot_train = load_twibot(paths["twibot20_train"])
    df_twibot_dev   = load_twibot(paths["twibot20_dev"])
    df_twibot_test  = load_twibot(paths["twibot20_test"])

    df_twibot = pd.concat(
        [df_twibot_train, df_twibot_dev, df_twibot_test],
        ignore_index=True
    )

    df_twibot = unify(df_twibot)

    # Remove duplicate columns
    df_twibot = df_twibot.loc[:, ~df_twibot.columns.duplicated()]

    # 3) Merge (GitHub REMOVED)
    merged = pd.concat([df_kaggle, df_twibot], ignore_index=True, sort=False)

    # Remove duplicates on rows
    merged = merged.drop_duplicates(subset=["user_id"], keep="first")

    # Remove duplicate columns AGAIN after merge
    merged = merged.loc[:, ~merged.columns.duplicated()]

    # Keep only labeled rows
    merged = merged.dropna(subset=["label"])

    return merged
