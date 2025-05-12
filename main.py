# main.py

from utils.data_loader import load_and_preprocess
from models import (
    LogisticRegressionModel,
    DecisionTreeModel,
    SVMModel,
    RandomForestModel,
    NeuralNetworkModel
)
from sklearn.metrics import accuracy_score

def main():
    # 1. Load data
    X_train, X_test, y_train, y_test = load_and_preprocess('data/mushrooms.csv')

    # 2. Initialize models
    models = {
        'logistic_regression': LogisticRegressionModel(max_iter=200),
        'decision_tree': DecisionTreeModel(),
        'svm': SVMModel(kernel='rbf'),
        'random_forest': RandomForestModel(n_estimators=100, random_state=42),
        'neural_network': NeuralNetworkModel(input_dim=X_train.shape[1])
    }

    # 3. Train, evaluate, and save each model
    for name, model in models.items():
        if name == 'neural_network':
            model.train(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)
            save_path = f'artifacts/{name}.h5'
        else:
            model.train(X_train, y_train)
            save_path = f'artifacts/{name}.joblib'

        preds = model.predict(X_test)
        print(f"{name} accuracy: {accuracy_score(y_test, preds):.4f}")
        model.save(save_path)

if __name__ == '__main__':
    main()
