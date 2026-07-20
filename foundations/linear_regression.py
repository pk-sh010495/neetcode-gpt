import numpy as np
from numpy.typing import NDArray

class Solution:
    def __init__(self):
        pass

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        try:
            predictions = X @ weights
            return np.round(predictions,5)
        except ValueError: #triggered if X.shape[1] != len(weights):
            return 0.0

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        mse = np.sum(np.square(ground_truth - model_prediction))/len(ground_truth)
        return np.round(mse,5)
        
