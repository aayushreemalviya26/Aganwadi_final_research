import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import ml_config as config

def load_and_preprocess_data():
    """Loads CSV, maps 3-class target, encodes baseline features, and extracts groups."""
    # 1. Load raw data
    df = pd.read_csv(config.DATA_PATH)
    
    # 2. Filter valid outcomes and map 3-class target
    df = df[df[config.TARGET_COL].isin(config.TARGET_MAP.keys())].copy()
    y = df[config.TARGET_COL].map(config.TARGET_MAP).astype(int)
    
    # 3. Extract grouping series (replication_id)
    groups = df[config.GROUP_COL]
    
    # 4. Extract and encode features
    X = pd.get_dummies(
        df[config.FEATURE_COLS], 
        columns=config.CATEGORICAL_FEATURES, 
        drop_first=True
    )
    
    return X, y, groups

def split_data_by_group(X, y, groups, test_size=0.2, random_state=42):
    """Splits data strictly by replication seed to prevent leakage across runs."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    return X_train, X_test, y_train, y_test