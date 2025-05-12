# models/base_model.py

import abc
import joblib

class BaseModel(abc.ABC):
    """Abstract base class for all classifiers."""

    @abc.abstractmethod
    def train(self, X_train, y_train):
        """Train the model on the provided data."""
        pass

    @abc.abstractmethod
    def predict(self, X):
        """Predict labels for the given feature matrix."""
        pass

    def save(self, path: str):
        """Serialize the trained model to disk."""
        joblib.dump(self._model, path)

    @classmethod
    def load(cls, path: str):
        """Load a serialized model from disk."""
        instance = cls.__new__(cls)
        instance._model = joblib.load(path)
        return instance
