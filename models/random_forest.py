# models/random_forest.py

from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel

class RandomForestModel(BaseModel):
    """Random Forest classifier."""

    def __init__(self, **kwargs):
        # Hyperparameters: n_estimators, max_depth, random_state, etc.
        self._model = RandomForestClassifier(**kwargs)

    def train(self, X_train, y_train):
        """Fit the random forest to training data."""
        self._model.fit(X_train, y_train)

    def predict(self, X):
        """Predict class labels for samples in X."""
        return self._model.predict(X)
