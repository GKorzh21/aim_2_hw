from typing import Optional

import numpy as np

from sem_dt_rf.decision_tree.criterio import Criterion, GiniCriterion, EntropyCriterion, MSECriterion
from sem_dt_rf.decision_tree.tree_node import TreeNode


class DecisionTree:
    def __init__(self, max_depth: int = 10, min_leaf_size: int = 5, min_improvement: Optional[float] = None):
        self.criterion = None
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.min_improvement = min_improvement


    def _build_nodes(self, x: np.ndarray, y: np.ndarray, criterion: Criterion, indices: np.ndarray, node: TreeNode):
        """
        Builds tree recursively

        Parameters
        ----------
        x : samples in node, np.ndarray.shape = (n_samples, n_features)
        y : target values, np.ndarray.shape = (n_samples, )
        criterion : criterion to split by, Criterion
        indices : samples' indices in node,
            np.ndarray.shape = (n_samples, )
            nd.ndarray.dtype = int
        node : current node to split, TreeNode
        """
        #Если достигли макс глубины то делаем узел листом и записываем предсказание 
        if self.max_depth is not None and self.max_depth <= node.depth:
            node.set_predictions(criterion.get_predict_val(y[indices]))
            return
        
        #Если соатлось только одно уникальное значение, то делаем узел листом, потому что дальнейшее разбиение бессмусленно 
        if len(np.unique(y[indices])) <= 1:
            node.set_predictions(criterion.get_predict_val(y[indices]))
            return
        
        #Если узел содержит меньеш обьектов, чем min_leaf_size, то делаем его листом
        if self.min_leaf_size is not None and self.min_leaf_size >= len(indices):
            node.set_predictions(criterion.get_predict_val(y[indices]))
            return

        #Ищем лучшее разбиение
        node.find_best_split(x[indices], y[indices], criterion)
        #Под капотом нашись node.feature_id - номер признака по которому будем делать, node.threshold - порог, node.q_value_max - уменьшение шума

        #Проверяем есть ли смысл делать дальше, если нет то делаем узел листом
        if self.min_improvement is not None and self.min_improvement >= node.q_value_max:
            node.set_predictions(criterion.get_predict_val(y[indices]))
            return
        
        #СОздаем детей, маску и строим дераво больше
        node.create_children()
        mask = node.get_best_split_mask(x[indices])
        self._build_nodes(x, y, criterion, indices[mask], node.left_child)
        self._build_nodes(x, y, criterion, indices[~mask], node.right_child)

    def _get_nodes_predictions(self, x: np.ndarray, predictions: np.ndarray, indices: np.ndarray, node: TreeNode):
        if node.is_terminal():
            predictions[indices] = node.predictions
            return
        mask = node.get_best_split_mask(x[indices])
        self._get_nodes_predictions(x, predictions, indices[mask], node.left_child)
        self._get_nodes_predictions(x, predictions, indices[~mask], node.right_child)


class ClassificationDecisionTree(DecisionTree):
    def __init__(self, criterion: str = "gini", **kwargs):
        super().__init__(**kwargs)

        if criterion not in ["gini", "entropy"]:
            raise ValueError('Unsupported criterion', criterion)
        self.criterion = criterion
        self.n_classes = 0
        self.n_features = 0
        self.root = None

    def fit(self, x, y):
        self.n_classes = np.unique(y).shape[0]
        self.n_features = x.shape[1]
        criterion = None
        if self.criterion == "gini":
            criterion = GiniCriterion(self.n_classes)
        elif self.criterion == "entropy":
            criterion = EntropyCriterion(self.n_classes)
        self.root = TreeNode(depth=0)
        self._build_nodes(x, y, criterion, np.arange(x.shape[0]), self.root)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba(x).argmax(axis=1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        predictions = np.zeros((x.shape[0], self.n_classes))
        self._get_nodes_predictions(x, predictions, np.arange(x.shape[0]), self.root)
        return predictions

    #Вернуть массив сумм уменьшения шума в каждой ноду при делении вдоль соответствуюей фичи
    def feature_importance_(self) -> np.ndarray:
        """
        Returns
        -------
        importance : cumulative improvement per feature, np.ndarray.shape = (n_features, )
        
        """

        self.importance = np.zeros(self.n_features)

        def collect_imp(node: TreeNode):
            if node.is_terminal():
                return 
            self.importance[node.feature_id] += node.q_value_max
            collect_imp(node.left_child)
            collect_imp(node.right_child)

        collect_imp(self.root)

        #Просто важность 
        #return self.importance

        #Нормированая важность
        return self.importance / np.sum(self.importance)


class RegressionDecisionTree(DecisionTree):
    def __init__(self, criterion: str = "mse", **kwargs):
        super().__init__(**kwargs)

        if criterion not in ["mse"]:
            raise ValueError('Unsupported criterion', criterion)
        self.criterion = criterion
        self.root = None
        self.n_features = 0

    def fit(self, x, y):
        self.n_features = x.shape[1]
        criterion = MSECriterion()

        self.root = TreeNode(depth=0)
        self._build_nodes(x, y, criterion, np.arange(x.shape[0]), self.root)

    def predict(self, x: np.ndarray) -> np.ndarray:
        predictions = np.zeros(x.shape[0])
        self._get_nodes_predictions(x, predictions, np.arange(x.shape[0]), self.root)
        return predictions

    #Вернуть массив сумм уменьшения шума в каждой ноду при делении вдоль соответствуюей фичи
    def feature_importance_(self) -> np.ndarray:
        """
        Returns
        -------
        importance : cumulative improvement per feature, np.ndarray.shape = (n_features, )

        """

        self.importance = np.zeros(self.n_features)

        def collect_imp(node: TreeNode):
            if node.is_terminal():
                return 
            self.importance[node.feature_id] += node.q_value_max
            collect_imp(node.left_child)
            collect_imp(node.right_child)

        collect_imp(self.root)

        #Нормированая важность
        return self.importance / np.sum(self.importance)


