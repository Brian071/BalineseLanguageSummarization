import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath='korpus_mentah_new.csv'):
    """
    Loads the dataset from CSV, handling encoding.
    """
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='ISO-8859-1')
    return df

def clean_and_select_features(df):
    """
    Selects relevant features and target, and drops missing values.
    """
    feature_cols = ['f1_lead', 'f2_judul_sim', 'f3_freq_word',
                    'f4_sim_sent', 'f5_len_norm', 'f6_overlap']
    target_col = 'Label_Final'

    # Ensure columns exist
    for col in feature_cols + [target_col]:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in dataset.")

    df_selected = df[feature_cols + [target_col]].copy()

    # Drop missing values
    df_cleaned = df_selected.dropna()

    return df_cleaned, feature_cols, target_col

def prepare_data(filepath='korpus_mentah_new.csv', test_size=0.2, random_state=42):
    """
    Full pipeline: load, clean, split.
    Returns X, y (Unscaled - Scaling handled in model pipeline).
    """
    df = load_data(filepath)
    df_clean, features, target = clean_and_select_features(df)

    X = df_clean[features]
    y = df_clean[target]

    return X, y
