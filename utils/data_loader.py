# utils/data_loader.py

import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split(
    filepath: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_col: str = 'class'
):
    """
    Load the mushroom dataset, perform minimal cleaning, 
    and return stratified train/test splits.
    
    :param filepath: Path to CSV file.
    :param test_size: Fraction of data to reserve for testing.
    :param random_state: Seed for reproducibility.
    :param stratify_col: Column name to stratify on.
    :return: X_train, X_test, y_train, y_test (all as pandas.DataFrame/Series)
    """
    # 1. Read CSV
    data = pd.read_csv(filepath)
    
    # 2. Basic integrity check
    if data.isnull().any().any():
        raise ValueError("Dataset contains missing values.")
    
    # 3. Separate features & target
    X = data.drop(columns=[stratify_col])
    y = data[stratify_col].map({'e': 0, 'p': 1})  # encode target to 0/1
    
    # 4. Stratified split to preserve class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test
