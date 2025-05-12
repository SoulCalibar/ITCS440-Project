import unittest
import numpy as np
from models.logistic_regression import LogisticRegressionModel

class TestModels(unittest.TestCase):
    def test_logistic_regression(self):
        X = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
        y = np.array([0, 1, 0, 1])
        model = LogisticRegressionModel()
        model.train(X, y)
        preds = model.predict(X)
        self.assertTrue((preds == y).all())
