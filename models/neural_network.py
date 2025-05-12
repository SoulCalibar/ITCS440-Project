# models/neural_network.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from .base_model import BaseModel

class NeuralNetworkModel(BaseModel):
    """Simple feedforward Neural Network."""

    def __init__(self, input_dim: int, **kwargs):
        """
        Build a two-layer NN.
        :param input_dim: Number of input features.
        :param kwargs: Additional compile/training args (e.g., optimizer, loss).
        """
        self._model = Sequential([
            Dense(64, activation='relu', input_shape=(input_dim,)),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self._model.compile(
            optimizer=kwargs.get('optimizer', 'adam'),
            loss=kwargs.get('loss', 'binary_crossentropy'),
            metrics=kwargs.get('metrics', ['accuracy'])
        )

    def train(self, X_train, y_train, **fit_kwargs):
        """
        Train the neural network.
        :param fit_kwargs: epochs, batch_size, validation_split, etc.
        """
        self._model.fit(X_train, y_train, **fit_kwargs)

    def predict(self, X):
        """Return binary class predictions (0 or 1)."""
        probs = self._model.predict(X)
        return (probs.flatten() > 0.5).astype(int)

    def save(self, path: str):
        """Save Keras model to HDF5."""
        self._model.save(path)

    @classmethod
    def load(cls, path: str, input_dim: int, **compile_kwargs):
        """
        Load a saved Keras model.
        :param input_dim: Required to instantiate the wrapper (not used internally).
        """
        instance = cls(input_dim, **compile_kwargs)
        instance._model = Sequential().from_config(instance._model.get_config())
        instance._model.load_weights(path)
        return instance
