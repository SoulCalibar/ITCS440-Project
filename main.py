# main.py

from utils.data_loader import load_and_split
from utils.preprocessing import build_preprocessor, preprocess_data
from models.init import (
    LogisticRegressionModel,
    DecisionTreeModel,
    SVMModel,
    RandomForestModel,
    NeuralNetworkModel
)
from sklearn.metrics import accuracy_score

def main():
    # 1. Load & split data
    X_train, X_test, y_train, y_test = load_and_split('data/mushrooms.csv')
    
    # 2. Identify feature types
    cat_cols = X_train.select_dtypes(include='object').columns.tolist()
    num_cols = X_train.select_dtypes(include='number').columns.tolist()
    
    # 3. Build & apply preprocessor
    preprocessor = build_preprocessor(cat_cols, num_cols)
    X_train_prep, X_test_prep = preprocess_data(preprocessor, X_train, X_test)
    
    # 4. Initialize models
    models = {
        'logistic_regression': LogisticRegressionModel(max_iter=200),
        'decision_tree':      DecisionTreeModel(),
        'svm':                SVMModel(kernel='rbf'),
        'random_forest':      RandomForestModel(n_estimators=100, random_state=42),
        'neural_network':     NeuralNetworkModel(input_dim=X_train_prep.shape[1])
    }
    
    # 5. Train, evaluate, save
    for name, m in models.items():
        if name == 'neural_network':
            m.train(X_train_prep, y_train, epochs=10, batch_size=32, validation_split=0.1)
            save_path = f'artifacts/{name}.h5'
        else:
            m.train(X_train_prep, y_train)
            save_path = f'artifacts/{name}.joblib'
        
        preds = m.predict(X_test_prep)
        print(f"{name} accuracy: {accuracy_score(y_test, preds):.4f}")
        m.save(save_path)
