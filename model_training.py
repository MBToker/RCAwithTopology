
import pandas as pd
import numpy as np
import os
import pickle
from prophet import Prophet
from pycaret.regression import *
import shap
from sqlalchemy import create_engine
from dotenv import load_dotenv


# === Helper functions ===

def sanitize_filename(name):
    return name.replace('/', '_').replace('\\', '_').replace(' ', '_').replace(':','_')


# === SEASONALITY REMOVAL ===

def seasonality_removal(df, group_col, node_type, time_col='timestamp'):
    print(f"\n🌀 Running seasonality removal for group_col: {group_col}")

    df_out = df.copy()
    df_out[time_col] = pd.to_datetime(df_out[time_col])

    metrics = [col for col in df_out.columns if col not in [time_col, group_col]]
    group_names = df_out[group_col].unique()

    for metric in metrics:
        print(f"\n  📈 Processing metric: {metric}")

        df_out[f"{metric}_residual"] = np.nan

        for group in group_names:
            df_group = df_out[df_out[group_col] == group].copy()
            df_group = df_group[[time_col, group_col, metric]].rename(columns={time_col: 'ds', metric: 'y'}).sort_values('ds')

            model = Prophet()
            model.fit(df_group[['ds', 'y']])

            future = model.make_future_dataframe(periods=0, freq='5min')
            forecast = model.predict(future)

            residual = df_group['y'].values - forecast['yhat'].values
            df_out.loc[df_out[group_col] == group, f"{metric}_residual"] = residual

            # Save Prophet model
            save_dir = os.path.join('models', 'anomaly_detection', node_type, sanitize_filename(group))
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, f"{sanitize_filename(metric)}_prophet.pkl"), 'wb') as f:
                pickle.dump(model, f)

    print(f"\n✅ Seasonality removal completed for {group_col}")
    return df_out


# === FORECASTING PIPELINE ===

def forecasting_pipeline(df, node_type, group_col):
    print(f"\n🚀 Starting forecasting pipeline for {node_type}")

    model_dir = os.path.join('models', 'forecasting', node_type)
    shap_dir = os.path.join('shap_outputs', node_type)

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(shap_dir, exist_ok=True)

    time_col = 'timestamp'
    df[time_col] = pd.to_datetime(df[time_col])

    nodes = df[group_col].unique()

    for node_name in nodes:
        print(f"\n⚙️ Node: {node_name}")

        safe_node_name = sanitize_filename(node_name)

        node_df = df[df[group_col] == node_name].copy()
        node_df = node_df.sort_values(time_col)

        residual_cols = [col for col in node_df.columns if col.endswith('_residual')]
        print(f"  Found {len(residual_cols)} residual metrics")

        for target_col in residual_cols:
            print(f"\n    🎯 Training model for target: {target_col}")

            exp = setup(
                data=node_df,
                target=target_col,
                ignore_features=[time_col, group_col],
                fold_strategy='timeseries',
                fold=5,
                data_split_shuffle=False,
                fold_shuffle=False,
                session_id=42,
                silent=True
            )

            best_model = compare_models(include=['lightgbm', 'xgboost'], n_select=1)
            tuned_model = tune_model(best_model)

            safe_target = sanitize_filename(target_col)

            model_path = os.path.join(model_dir, f"{node_type}_{safe_node_name}_{safe_target}")
            save_model(tuned_model, model_path)
            print(f'    💾 Model saved: {model_path}.pkl')

            feature_list_path = os.path.join(model_dir, f"{node_type}_{safe_node_name}_{safe_target}_features.txt")
            with open(feature_list_path, 'w') as f:
                f.write(f'Feature list for model: {node_type}_{node_name}\n\n')
                f.write(f'Target column: {target_col}\n\n')
                f.write('Features:\n')
                for feat in get_config('X').columns:
                    f.write(f' - {feat}\n')

            explainer = shap.Explainer(tuned_model)
            X = get_config('X')
            shap_values = explainer(X)

            shap_df = pd.DataFrame(shap_values.values, columns=X.columns)
            shap_df['timestamp'] = node_df[time_col].values[:len(shap_df)]
            shap_output_path = os.path.join(shap_dir, f"{node_type}_{safe_node_name}_{safe_target}_shap.csv")
            shap_df.to_csv(shap_output_path, index=False)
            print(f'SHAP saved: {shap_output_path}')
            

# === ANOMALY DETECTION PIPELINE ===

def prophet_residual_anomaly_detection(df, group_col, metric_col, timestamp_col='timestamp', target_group=None, user_sensitivity=3):
    df_prophet = df[[group_col, timestamp_col, metric_col]].copy()
    df_prophet[timestamp_col] = pd.to_datetime(df_prophet[timestamp_col])
    
    if target_group is None:
        target_group = df[group_col].value_counts().idxmax()
    
    df_group = df_prophet[df_prophet[group_col] == target_group].copy()
    df_group = df_group.rename(columns={timestamp_col: 'ds', metric_col: 'y'})
    df_group = df_group.sort_values('ds')
    
    model = Prophet()
    model.fit(df_group[['ds', 'y']])
    
    future = model.make_future_dataframe(periods=0, freq='5min')
    forecast = model.predict(future)
    
    df_group['yhat'] = forecast['yhat'].values
    df_group['residual'] = df_group['y'] - df_group['yhat']
    mean_res = df_group['residual'].mean()
    std_res = df_group['residual'].std()
    df_group['z_score'] = (df_group['residual'] - mean_res) / std_res
    
    z_thresh_map = {1: 1.5, 2: 2.0, 3: 2.5, 4: 3.0, 5: 3.5}
    z_thresh = z_thresh_map.get(user_sensitivity, 3.0)
    
    df_group['is_anomaly'] = (df_group['z_score'].abs() > z_thresh).astype(int)
    
    return df_group, model, z_thresh


def anomaly_detection_pipeline(df, group_col_name, node_type, filename, sensitivity=3):
    print(f"\n🚀 Starting anomaly detection for {node_type}")

    metrics = [col for col in df.columns if col.endswith('_residual')]
    group_names = df[group_col_name].unique().tolist()

    df_output = df.copy()
    results = []

    z_thresh_map = {1: 1.5, 2: 2.0, 3: 2.5, 4: 3.0, 5: 3.5}
    z_thresh = z_thresh_map.get(sensitivity, 3.0)

    for metric in metrics:
        print(f"✅ Processing metric: {metric}")

        anomaly_col = f"{metric}_is_anomaly"
        df_output[anomaly_col] = 0

        for group in group_names:
            df_group = df_output[df_output[group_col_name] == group].copy()
            residual_values = df_group[metric]

            mean_res = residual_values.mean()
            std_res = residual_values.std()
            df_group['z_score'] = (residual_values - mean_res) / std_res
            df_group['is_anomaly'] = (df_group['z_score'].abs() > z_thresh).astype(int)

            anomaly_timestamps = df_group[df_group['is_anomaly'] == 1]['timestamp']
            mask = (df_output[group_col_name] == group) & (df_output['timestamp'].isin(anomaly_timestamps))
            df_output.loc[mask, anomaly_col] = 1

            for _, row in df_group.iterrows():
                results.append({
                    'ds': row['timestamp'],
                    'group': group,
                    'metric': metric,
                    'residual': row[metric],
                    'z_score': row['z_score'],
                    'is_anomaly': row['is_anomaly']
                })

            # Save threshold
            save_dir = f'models/anomaly_detection/{node_type}/{sanitize_filename(group)}'
            os.makedirs(save_dir, exist_ok=True)
            with open(f'{save_dir}/{sanitize_filename(metric)}_zthresh.txt', 'w') as f:
                f.write(f'Z threshold used: {z_thresh}\n')

    out_dir_default = "anomaly_detected_datasets/default"
    out_dir_sens = "anomaly_detected_datasets/sensitivity"
    os.makedirs(out_dir_default, exist_ok=True)
    os.makedirs(out_dir_sens, exist_ok=True)

    df_output.to_csv(f"{out_dir_default}/{filename}_with_anomalies_v2.csv", index=False)
    print(f"✅ Saved: {out_dir_default}/{filename}_with_anomalies_v2.csv")

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{out_dir_sens}/{filename}_anomaly_results.csv", index=False)
    print(f"✅ Saved: {out_dir_sens}/{filename}_anomaly_results.csv")

    print(f"Anomalies detected for {filename}")


# === DUPLICATE TIMESTAMP CHECKER ===
def check_group_duplicate_timestamps(df, group_col_name, timestamp_col='timestamp'):
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    duplicate_mask = df.duplicated(subset=[group_col_name, timestamp_col], keep=False)
    total_duplicates = duplicate_mask.sum()
    print(f"✅ Total duplicate (group + timestamp) count: {total_duplicates}")
    group_duplicate_summary = df[duplicate_mask].groupby(group_col_name).size().sort_values(ascending=False)
    return group_duplicate_summary

# === MAIN FUNCTION ===

def main():
    # === Load .env ===
    load_dotenv()

    vertica_user = os.getenv('VERTICA_USER')
    vertica_password = os.getenv('VERTICA_PASSWORD')
    vertica_host = os.getenv('VERTICA_HOST')
    vertica_port = os.getenv('VERTICA_PORT')
    vertica_db = os.getenv('VERTICA_DB')

    # === Create connection ===
    engine = create_engine(
        f'vertica+vertica_python://{vertica_user}:{vertica_password}@{vertica_host}:{vertica_port}/{vertica_db}'
    )

    # === Load data from Vertica ===
    transactions_df = pd.read_sql("SELECT * FROM preprocessed.transaction_metrics", engine)
    network_df = pd.read_sql("SELECT * FROM preprocessed.network_metrics", engine)
    server_df = pd.read_sql("SELECT * FROM preprocessed.server_metrics", engine)
    connections_df = pd.read_sql("SELECT * FROM preprocessed.connections", engine)

    connections_df.columns = connections_df.columns.str.lower()

    # === TOPOLOGY-AWARE FEATURES ===

    transactions_df['source'] = transactions_df['relatedci'].map(
        connections_df.set_index('target')['source']
    )

    network_metrics_to_join = [
        'timestamp',
        'node_name',
        'snmp_response_time_ms',
        'cpu_util_pct'
    ]

    transactions_df = pd.merge(
        transactions_df,
        network_df[network_metrics_to_join],
        left_on=['timestamp', 'source'],
        right_on=['timestamp', 'node_name'],
        how='left',
        suffixes=('', '_network_source')
    )

    print("\n✅ Topology-aware features added to transactions_df")

    network_df['source'] = network_df['node_name'].map(
        connections_df.set_index('target')['source']
    )

    server_metrics_to_join = [
        'timestamp',
        'node_short_name',
        'cpu_util_pct',
        'mem_util_pct'
    ]

    network_df = pd.merge(
        network_df,
        server_df[server_metrics_to_join],
        left_on=['timestamp', 'source'],
        right_on=['timestamp', 'node_short_name'],
        how='left',
        suffixes=('', '_server_source')
    )

    print("\n✅ Topology-aware features added to network_df")

    # === Seasonality removal ===

    transactions_df = seasonality_removal(transactions_df, group_col='relatedci', node_type='transactions')
    network_df = seasonality_removal(network_df, group_col='node_name', node_type='network')
    server_df = seasonality_removal(server_df, group_col='node_short_name', node_type='server')

    # === Forecasting ===

    forecasting_pipeline(transactions_df, 'transactions', group_col='relatedci')
    forecasting_pipeline(network_df, 'network', group_col='node_name')
    forecasting_pipeline(server_df, 'server', group_col='node_short_name')

    # === Duplicate timestamp checks ===

    app_summary = check_group_duplicate_timestamps(transactions_df, group_col_name='relatedci')
    network_summary = check_group_duplicate_timestamps(network_df, group_col_name='node_name')
    server_summary = check_group_duplicate_timestamps(server_df, group_col_name='node_short_name')

    print(f"\nTransactions duplicate summary:\n{app_summary}")
    print(f"\nNetwork duplicate summary:\n{network_summary}")
    print(f"\nServer duplicate summary:\n{server_summary}")

    print("\n🎉 Pipeline run completed!")

if __name__ == "__main__":
    main()