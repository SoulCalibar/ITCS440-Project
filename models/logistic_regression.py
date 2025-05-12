# models/logistic_regression.py

from sklearn.linear_model import LogisticRegression
from .base_model import BaseModel

class LogisticRegressionModel(BaseModel):
    """Logistic Regression classifier."""

    def __init__(self, **kwargs):
        # Pass any hyperparameters via kwargs (e.g., max_iter, C, solver)
        self._model = LogisticRegression(**kwargs)

    def train(self, X_train, y_train):
        """Fit the logistic regression model."""
        self._model.fit(X_train, y_train)

    def predict(self, X):
        """Return predicted class labels."""
        return self._model.predict(X)
