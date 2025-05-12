# models/svm.py

from sklearn.svm import SVC
from .base_model import BaseModel

class SVMModel(BaseModel):
    """Support Vector Machine classifier."""

    def __init__(self, **kwargs):
        # Hyperparameters: kernel, C, gamma, etc.
        self._model = SVC(**kwargs)

    def train(self, X_train, y_train):
        """Train the SVM on the provided data."""
        self._model.fit(X_train, y_train)

    def predict(self, X):
        """Return predicted class labels."""
        return self._model.predict(X)
