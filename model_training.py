# forecasting_pipeline.py

import pandas as pd
import numpy as np
import os
import pickle
from prophet import Prophet
from pycaret.regression import *
from sqlalchemy import create_engine
from dotenv import load_dotenv

# === Helper ===

def sanitize_filename(name):
    return name.replace('/', '_').replace('\\', '_').replace(' ', '_').replace(':','_')

# === Params ===

OUTPUT_DIR = 'models/forecasting'
TARGET_RESOLUTION_MIN = 5
SHAP_MODELS = ['lightgbm', 'xgboost']

# === Load Vertica connection ===

load_dotenv()

vertica_user = os.getenv('VERTICA_USER')
vertica_password = os.getenv('VERTICA_PASSWORD')
vertica_host = os.getenv('VERTICA_HOST')
vertica_port = os.getenv('VERTICA_PORT')
vertica_db = os.getenv('VERTICA_DB')

engine = create_engine(
    f"vertica+vertica_python://{vertica_user}:{vertica_password}@{vertica_host}:{vertica_port}/{vertica_db}")

# === Load data ===

transactions_df = pd.read_sql("SELECT * FROM preprocessed.transaction_metrics", engine)
network_df = pd.read_sql("SELECT * FROM preprocessed.network_metrics", engine)
server_df = pd.read_sql("SELECT * FROM preprocessed.server_metrics", engine)

# === Normalize timestamps ===

transactions_df['timestamp'] = pd.to_datetime(transactions_df['timestamp']).dt.floor(f'{TARGET_RESOLUTION_MIN}min')
network_df['timestamp'] = pd.to_datetime(network_df['timestamp']).dt.floor(f'{TARGET_RESOLUTION_MIN}min')
server_df['timestamp'] = pd.to_datetime(server_df['timestamp']).dt.floor(f'{TARGET_RESOLUTION_MIN}min')

# === Add node_id columns ===

transactions_df['trans_node_id'] = transactions_df['relatedci']
network_df['net_node_id'] = network_df['node_name']
server_df['serv_node_id'] = server_df['node_short_name']

# === Prefix columns ===

transactions_df = transactions_df.add_prefix('trans_')
network_df = network_df.add_prefix('net_')
server_df = server_df.add_prefix('serv_')

# Restore timestamp column
transactions_df = transactions_df.rename(columns={'trans_timestamp': 'timestamp'})
network_df = network_df.rename(columns={'net_timestamp': 'timestamp'})
server_df = server_df.rename(columns={'serv_timestamp': 'timestamp'})

# === Merge all groups ===

merged_df = transactions_df.merge(network_df, on='timestamp', how='outer')
merged_df = merged_df.merge(server_df, on='timestamp', how='outer')

print(f"\nMerged dataframe shape: {merged_df.shape}")

# === Force numeric types ===

for col in merged_df.columns:
    if col != 'timestamp':
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

# === Load feature influence CSV ===

influence_df = pd.read_csv('models/feature_influence_global/feature_influence_logical.csv')

# === Main loop for target features ===

os.makedirs(OUTPUT_DIR, exist_ok=True)

for idx, row in influence_df.iterrows():

    target_feature = row['target_feature']
    influencing_features = row['influencing_features'].split(', ')
    target_group = target_feature.split('.')[0]
    target_metric = target_feature.split('.')[1]

    print(f"\nTarget: {target_feature}")
    print(f"    Influencing features: {len(influencing_features)} features")

    node_df = merged_df.copy()

    # Prophet seasonality removal
    df_prophet = node_df[['timestamp', target_feature]].dropna()
    df_prophet = df_prophet.rename(columns={'timestamp': 'ds', target_feature: 'y'}).sort_values('ds')

    if df_prophet.shape[0] < 100:
        print(f"Skipping target {target_feature}, insufficient data ({df_prophet.shape[0]} rows)")
        continue

    prophet_model = Prophet()
    prophet_model.fit(df_prophet[['ds', 'y']])
    future = prophet_model.make_future_dataframe(periods=0, freq=f'{TARGET_RESOLUTION_MIN}min')
    forecast = prophet_model.predict(future)

    # Residual
    df_prophet['residual'] = df_prophet['y'] - forecast['yhat']

    # Build training dataframe
    train_df = node_df[['timestamp'] + influencing_features].copy()
    train_df = train_df.merge(df_prophet[['ds', 'residual']], left_on='timestamp', right_on='ds', how='inner')
    train_df = train_df.drop(columns=['ds'])
    train_df = train_df.dropna()

    print(f"    Final training shape: {train_df.shape}")

    # PyCaret regression
    exp = setup(
        data=train_df,
        target='residual',
        ignore_features=['timestamp'],
        fold_strategy='timeseries',
        fold=3,
        data_split_shuffle=False,
        fold_shuffle=False,
        session_id=42
    )

    best_model = compare_models(include=SHAP_MODELS, n_select=1)
    tuned_model = tune_model(best_model)

    safe_target = sanitize_filename(target_feature)
    model_path = os.path.join(OUTPUT_DIR, f"{safe_target}_model")
    save_model(tuned_model, model_path)

    print(f"Model saved: {model_path}.pkl")

    # Save feature list
    feature_list_path = os.path.join(OUTPUT_DIR, f"{safe_target}_features.txt")
    with open(feature_list_path, 'w') as f:
        f.write(f'Feature list for model: {target_feature}\n\n')
        for feat in get_config('X').columns:
            f.write(f'{feat}\n')

    print(f"Feature list saved: {feature_list_path}")

print("\n?? Forecasting pipeline completed!")
