# preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path: str) -> pd.DataFrame:
    """
    Loads dataset from a CSV file.

    Args:
        path (str): File path to the dataset.

    Returns:
        pd.DataFrame: Loaded data.
    """
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the input DataFrame.

    - Removes unnamed columns.
    - Handles missing values (if any).

    Args:
        df (pd.DataFrame): Raw DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.dropna(inplace=True)
    return df


def split_features_labels(df: pd.DataFrame, target_column: str):
    """
    Splits the DataFrame into features and labels.

    Args:
        df (pd.DataFrame): Preprocessed data.
        target_column (str): Name of the target column.

    Returns:
        X (pd.DataFrame): Features.
        y (pd.Series): Labels.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def scale_features(X_train, X_test):
    """
    Applies standard scaling to features.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.

    Returns:
        Scaled X_train and X_test
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def preprocess_pipeline(path: str, target_column: str, test_size=0.2, random_state=42):
    """
    Full preprocessing pipeline:
    - Load data
    - Clean data
    - Split into train/test
    - Scale features

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test
    """
    df = load_data(path)
    df = clean_data(df)
    X, y = split_features_labels(df, target_column)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test
