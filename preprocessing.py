import pandas as pd
import numpy as np

def cleaning_data(network_df, app_df, server_df):
    network_df.replace('(null)', np.nan, inplace=True)
    app_df.replace('(null)', np.nan, inplace=True)
    server_df.replace('(null)', np.nan, inplace=True)
    
    # Removing null columns
    print(f"Column count before cleaning: {len(network_df.columns)}")
    network_df = network_df.dropna(axis=1, how='all')
    print(f"Column count after cleaning: {len(network_df.columns)}")

    print(f"Column count before cleaning: {len(app_df.columns)}")
    app_df = app_df.dropna(axis=1, how='all')
    print(f"Column count after cleaning: {len(app_df.columns)}")

    print(f"Column count before cleaning: {len(server_df.columns)}")
    server_df = server_df.dropna(axis=1, how='all')
    print(f"Column count after cleaning: {len(server_df.columns)}")
    
    # === Timestamp transformation ===
    
    # Server
    server_df["timestamp"] = pd.to_datetime(server_df["timestamp_utc_s"], unit="s")
    server_df = server_df.sort_values(by="timestamp", ascending=True)
    
    # Network
    network_df["timestamp"] = pd.to_datetime(network_df["timestamp_utc_s"], unit="s")
    network_df = network_df.sort_values(by="timestamp", ascending=True)
    
    # App
    app_df["timestamp"] = pd.to_datetime(app_df["timestamp_utc"])
    app_df = app_df.sort_values(by="timestamp", ascending=True)
    
    # === Dropping unnecessary features ===
    
    # App
    app_df = app_df.drop(["id", "sourceid", "source", "host_name", "timestamp_utc", "node"], axis=1)
    # !!! timestamp kolonunu burada DROP ETMİYORUZ !!!
    
    # Server
    server_df = server_df.drop([
        "agent_collector", "boot_time", "cmdb_global_id", "cmdb_id",
        "collection_data_flow", "collection_policy_name", "collection_type",
        "cpu_multithreading_enabled", "cpu_phys_processor_count", "logical_system_role",
        "logical_system_type", "machine_model", "node_fqdn", "node_ip_type",
        "node_ipv4_address", "node_timezone_offset_h", "os_distribution", "os_type",
        "processor_architecture", "producer_instance_id", "producer_instance_type",
        "system_id", "gmtoffset_min", "interval_time_s", "timestamp_utc_s"], axis=1)
    
    # Network
    network_df = network_df.drop([
        "component_unique_id", "node_family",
        "security_group_unique_id", "security_group_name",
        "policy_unique_id", "node_unique_id", "tenant_id",
        "producer_instance_type", "producer_instance_id",
        "collection_timestamp_ms", "collection_timestamp_s", "timestamp_utc_s"], axis=1)
    
    return network_df, app_df, server_df


def aggregate_duplicate_timestamps(df, timestamp_col, group_col):
    # Node bazında timestamp duplicate'leri çöz
    df_grouped = df.groupby([timestamp_col, group_col]).mean(numeric_only=True).reset_index()
    return df_grouped


def pivot_app_features(df, timestamp_col, node_col, relatedci_col, metric_cols):
    df = df.copy()

    # Create source_target identifier
    df["source_target"] = df[node_col] + "_to_" + df[relatedci_col]

    # Create pivoted metric DataFrames
    pivoted_metrics = []
    for metric in metric_cols:
        pivot = df.pivot(index=timestamp_col, columns="source_target", values=metric)
        pivot.columns = [f"{col}_{metric}" for col in pivot.columns]
        pivoted_metrics.append(pivot)

    # Combine all metric pivots into a single DataFrame
    merged = pd.concat(pivoted_metrics, axis=1).reset_index()

    return merged


def pivot_server_features(df, timestamp, group_feature):
    df_grouped = df.groupby([timestamp, group_feature]).mean(numeric_only=True).reset_index()
    df_pivot = df_grouped.pivot(index=timestamp, columns=group_feature)
    df_pivot.columns = [f"{node}_{feature}" for feature, node in df_pivot.columns]
    df_pivot = df_pivot.reset_index()

    return df_pivot


def string_to_numeric(df):
    temp_df = df.copy()
    for col in temp_df.columns:
        if temp_df[col].dtype == "object" or temp_df[col].dtype.name == "string":
           temp_df[col] = temp_df[col].str.replace(",", "", regex=False).str.strip()  
           temp_df[col] = pd.to_numeric(temp_df[col], errors="coerce")  
    return temp_df


def filling_cols(df, low_ratio=0.02, mid_ratio=0.22, high_ratio=0.66):
    total_rows = len(df)
    nan_counts = df.isna().sum()

    low_threshold = int(total_rows * low_ratio)
    mid_threshold = int(total_rows * mid_ratio)
    high_threshold = int(total_rows * high_ratio)

    low_nan_cols = [col for col in df.columns if 0 < nan_counts[col] <= low_threshold]
    mid_nan_cols = [col for col in df.columns if low_threshold < nan_counts[col] <= mid_threshold]
    high_nan_cols = [col for col in df.columns if nan_counts[col] > high_threshold]

    df[low_nan_cols] = df[low_nan_cols].fillna(method="ffill")

    df[mid_nan_cols] = df[mid_nan_cols].interpolate(method="linear", limit_direction="both")

    df = df.drop(columns=high_nan_cols)

    for col in df.columns:
        if df[col].isna().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    remaining = df.isna().sum()
    if remaining.sum() == 0:
        print("[INFO] All data has been filled.")
    else:
        print("[WARNING] Still some Nan values exists:")
        print(remaining[remaining > 0])

    return df

    
def preprocessing():
    # Loading data
    network_df = pd.read_csv("datasets/nom_component_health.csv", delimiter=';')
    app_df = pd.read_csv("datasets/oa_appdynamics_transaction.csv", delimiter=';')
    server_df = pd.read_csv("datasets/opsb_agent_node.csv", delimiter=';')
    
    # Cleaning data
    network_df, app_df, server_df = cleaning_data(network_df, app_df, server_df)
    
    # Setting timestamp as index
    app_df = app_df.set_index('timestamp')
    server_df = server_df.set_index('timestamp')
    network_df = network_df.set_index('timestamp')
    
    # String to numeric conversion
    app_num_df = string_to_numeric(app_df)
    server_num_df = string_to_numeric(server_df)
    network_num_df = string_to_numeric(network_df)
    
    # Filling columns
    app_filled_df = filling_cols(app_num_df)
    server_filled_df = filling_cols(server_num_df)
    network_filled_df = filling_cols(network_num_df)
    
    # Re-adding group columns
    app_filled_df["relatedci"] = app_df["relatedci"]
    server_filled_df["node_short_name"] = server_df["node_short_name"]
    network_filled_df["node_name"] = network_df["node_name"]
    
    # Aggregate duplicate timestamps → Prophet garantili çalışsın
    server_filled_df = aggregate_duplicate_timestamps(server_filled_df, timestamp_col='timestamp', group_col='node_short_name')
    network_filled_df = aggregate_duplicate_timestamps(network_filled_df, timestamp_col='timestamp', group_col='node_name')
    
    # Adding timestamp as index
    server_filled_df = server_filled_df.set_index('timestamp')
    network_filled_df = network_filled_df.set_index('timestamp')
    
    # Save cleared datasets
    network_filled_df.to_csv("cleared_datasets/network.csv", index=True)
    app_filled_df.to_csv("cleared_datasets/app_transactions.csv", index=True)
    server_filled_df.to_csv("cleared_datasets/server.csv", index=True)
    
    print("✅ Preprocessing completed, cleared datasets saved.")
    
    return app_filled_df, server_filled_df, network_filled_df


def main():
    app_df, server_df, network_df = preprocessing()
    """app_og = pd.read_csv("cleared_datasets/app_with_anomalies.csv")
    print(app_df.columns)
    print(app_og.columns)
    
    server_og = pd.read_csv("cleared_datasets/server_with_anomalies.csv")
    print(server_df.columns)
    print(server_og.columns)
    
    
    print(network_df)"""
    
    
main()
