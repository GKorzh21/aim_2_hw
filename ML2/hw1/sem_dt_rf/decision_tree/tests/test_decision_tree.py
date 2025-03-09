import numpy as np
import unittest

from sem_dt_rf.decision_tree.decision_tree import ClassificationDecisionTree, RegressionDecisionTree


class TestDecisionTree(unittest.TestCase):
    def test_small_decision_tree(self):
        np.random.seed(1)
        clas_tree = ClassificationDecisionTree(max_depth=4, min_leaf_size=1)
        x = np.vstack((
            np.random.normal(loc=(-5, -5), size=(10, 2)),
            np.random.normal(loc=(-5, 5), size=(10, 2)),
            np.random.normal(loc=(5, -5), size=(10, 2)),
            np.random.normal(loc=(5, 5), size=(10, 2)),
        ))
        y = np.array(
            [0] * 20 + [1] * 20
        )
        clas_tree.fit(x, y)
        predictions = clas_tree.predict(x)
        assert (predictions == y).mean() == 1

    def test_decision_tree(self):
        np.random.seed(1)
        clas_tree = ClassificationDecisionTree(max_depth=4, min_leaf_size=1)
        x = np.vstack((
            np.random.normal(loc=(-5, -5), size=(100, 2)),
            np.random.normal(loc=(-5, 5), size=(100, 2)),
            np.random.normal(loc=(5, -5), size=(100, 2)),
            np.random.normal(loc=(5, 5), size=(100, 2)),
        ))
        y = np.array(
            [0] * 100 + [1] * 100 + [2] * 100 + [3] * 100
        )
        clas_tree.fit(x, y)
        predictions = clas_tree.predict(x)
        assert (predictions == y).mean() == 1


class TestRegressionTree(unittest.TestCase):
    def test_small_regression_tree(self):
        np.random.seed(1)
        reg_tree = RegressionDecisionTree(max_depth=10, min_leaf_size=1)
        x = np.linspace(-2, 2, 20).reshape(-1, 1)
        y = 2 * x.ravel() + 3
        reg_tree.fit(x, y)
        predictions = reg_tree.predict(x)
        np.testing.assert_allclose(predictions, y, atol=1e-3)

    def test_regression_tree(self):
        np.random.seed(1)
        reg_tree = RegressionDecisionTree(max_depth=5, min_leaf_size=2)
        x = np.random.uniform(-5, 5, size=(200, 2))
        y = np.sin(x[:, 0]) + np.cos(x[:, 1])
        reg_tree.fit(x, y)
        predictions = reg_tree.predict(x)
        assert np.corrcoef(predictions, y)[0, 1] > 0.9

    def test_feature_importance(self):
        np.random.seed(1)
        reg_tree = RegressionDecisionTree(max_depth=5, min_leaf_size=2)
        x = np.random.uniform(-5, 5, size=(200, 2))
        y = np.sin(x[:, 0]) + np.cos(x[:, 1])
        reg_tree.fit(x, y)
        importance = reg_tree.feature_importance_()
        assert np.sum(importance) > 0
        assert np.isclose(np.sum(importance), 1)
