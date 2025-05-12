# models/__init__.py

"""
The models package contains implementations of all machine learning
classifiers, each adhering to a common interface (BaseModel).
"""

from .logistic_regression import LogisticRegressionModel
from .decision_tree import DecisionTreeModel
from .svm import SVMModel
from .random_forest import RandomForestModel
from .neural_network import NeuralNetworkModel

__all__ = [
    "BaseModel",
    "LogisticRegressionModel",
    "DecisionTreeModel",
    "SVMModel",
    "RandomForestModel",
    "NeuralNetworkModel",
]
