# src/feature_engineering.py
"""
Module: feature_engineering
Purpose: Generate features for fraud and credit card datasets,
and save final datasets ready for modeling.

Outputs:
- data/processed/fraud_data_final.csv
- data/processed/creditcard_final.csv
"""

import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Load processed data
# -----------------------------
def load_processed_data(filename: str, folder: str = "data/processed") -> pd.DataFrame:
    try:
        df = pd.read_csv(Path(folder) / filename)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Processed file not found: {filename}")
    except Exception as e:
        raise Exception(f"Error loading processed CSV {filename}: {e}")

# -----------------------------
# Feature Engineering - Fraud Data
# -----------------------------
def engineer_fraud_features(fraud_df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = fraud_df.copy()

        # Temporal features
        df['time_diff'] = (pd.to_datetime(df['purchase_time']) - pd.to_datetime(df['signup_time'])).dt.total_seconds() / 3600
        df['purchase_hour'] = pd.to_datetime(df['purchase_time']).dt.hour
        df['purchase_day'] = pd.to_datetime(df['purchase_time']).dt.day_name()

        # Behavioral features
        df['instant_purchase'] = (df['time_diff'] < 1).astype(int)
        df['is_night'] = df['purchase_hour'].apply(lambda x: 1 if x >= 23 or x <= 5 else 0)
        median_purchase = df['purchase_value'].median()
        df['high_value'] = (df['purchase_value'] > median_purchase).astype(int)

        # Country risk feature
        risky_countries = df[df['class']==1]['country'].value_counts().head(3).index
        df['risk_country'] = df['country'].isin(risky_countries).astype(int)

        # Drop columns not needed for modeling
        df = df.drop(columns=['device_id', 'ip_address', 'signup_time', 'purchase_time'])

        # One-hot encode categorical variables
        cat_cols = ['source', 'browser', 'sex', 'country', 'purchase_day']
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        # Scale numerical features
        num_cols = ['purchase_value', 'age', 'time_diff', 'purchase_hour']
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])

    except Exception as e:
        raise Exception(f"Error in fraud feature engineering: {e}")

    return df

# -----------------------------
# Feature Engineering - Credit Card
# -----------------------------
def engineer_creditcard_features(credit_df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = credit_df.copy()
        scaler = StandardScaler()
        df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])
        # PCA features (V1-V28) are already numeric, keep as-is
    except Exception as e:
        raise Exception(f"Error in credit card feature engineering: {e}")
    return df

# -----------------------------
# Save final dataset
# -----------------------------
def save_final_data(df: pd.DataFrame, filename: str, folder: str = "data/processed"):
    try:
        Path(folder).mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(folder) / filename, index=False)
        print(f"Saved final dataset: {folder}/{filename}")
    except Exception as e:
        raise Exception(f"Error saving final CSV {filename}: {e}")

# -----------------------------
# Main function
# -----------------------------
def process_all_features():
    try:
        # Fraud data
        fraud_df = load_processed_data("fraud_data_processed.csv")
        fraud_final = engineer_fraud_features(fraud_df)
        save_final_data(fraud_final, "fraud_data_final.csv")

        # Credit card data
        credit_df = load_processed_data("creditcard_processed.csv")
        credit_final = engineer_creditcard_features(credit_df)
        save_final_data(credit_final, "creditcard_final.csv")

    except Exception as e:
        print(f"[ERROR] Feature engineering failed: {e}")

# -----------------------------
# Run script
# -----------------------------
if __name__ == "__main__":
    process_all_features()