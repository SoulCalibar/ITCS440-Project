# models/decision_tree.py

from sklearn.tree import DecisionTreeClassifier
from .base_model import BaseModel

class DecisionTreeModel(BaseModel):
    """Decision Tree classifier."""

    def __init__(self, **kwargs):
        # Hyperparameters: criterion, max_depth, min_samples_leaf, etc.
        self._model = DecisionTreeClassifier(**kwargs)

    def train(self, X_train, y_train):
        """Fit the decision tree to training data."""
        self._model.fit(X_train, y_train)

    def predict(self, X):
        """Predict class labels for samples in X."""
        return self._model.predict(X)
