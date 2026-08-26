import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        answer = - np.mean(y_true * np.log(y_pred + 1e-7) + (1 - y_true) * np.log(1 - y_pred + 1e-7))
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        return np.round(answer, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        answer = - np.mean(np.sum(y_true*np.log(y_pred + 1e-7), axis = 1))
        return np.round(answer, 4)
