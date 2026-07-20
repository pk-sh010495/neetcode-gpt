import numpy as np
from numpy.typing import NDArray

class Solution:
    def __init__(self, y_pred=None, y_true=None):
        pass

    def _stablize_probs(self, y_pred: NDArray[np.float64], epsilon: float = 1e-7) -> NDArray[np.float64]:
        return np.clip(y_pred, epsilon, 1.0 - epsilon, out=y_pred)

    def _final_loss(self, loss: NDArray[np.float64], total_samples: int, round_fig: int = 4) -> float:
        loss_scalar = - 1/total_samples * np.nansum(loss)
        return round(float(loss_scalar), round_fig)

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        self._stablize_probs(y_pred)
        loss = 1 * \
            ( \
                ( (y_true) * np.log(y_pred)) + \
                ( (1-y_true) * np.log(1-y_pred)) \
            )
        return self._final_loss(loss, len(y_true), 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        self._stablize_probs(y_pred)
        loss = y_true * np.log(y_pred)
        return self._final_loss(loss, len(y_true), 4)


