# utils/preprocessing.py

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd

def build_preprocessor(categorical_cols: list, numeric_cols: list):
    """
    Create a ColumnTransformer that one-hot encodes categoricals
    and standardizes numerics.
    
    :param categorical_cols: List of categorical feature names.
    :param numeric_cols: List of numeric feature names.
    :return: A scikit-learn ColumnTransformer.
    """
    # 1. Pipeline for categorical features
    cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # 2. Pipeline for numeric features
    num_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    # 3. Combine into ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ('cat', cat_pipeline, categorical_cols),
        ('num', num_pipeline, numeric_cols)
    ])
    
    return preprocessor

def preprocess_data(
    preprocessor, 
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame
):
    """
    Fit the preprocessor on X_train and transform both train and test.
    
    :param preprocessor: The ColumnTransformer from build_preprocessor.
    :param X_train: Training features.
    :param X_test: Testing features.
    :return: Transformed NumPy arrays: X_train_prep, X_test_prep
    """
    # Fit on training data and apply to both
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep  = preprocessor.transform(X_test)
    return X_train_prep, X_test_prep
