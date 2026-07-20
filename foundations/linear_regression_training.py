import numpy as np
from numpy.typing import NDArray


class Solution:
    learning_rate = 0.01
    
    def __init__(self, learning_rate=None):
        self.learning_rate = learning_rate or self.learning_rate

    def get_derivative(
        self, 
        model_prediction: NDArray[np.float64], 
        ground_truth: NDArray[np.float64], 
        N: int, 
        X: NDArray[np.float64], 
        desired_weight: int
    ) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(
        self, 
        X: NDArray[np.float64], 
        weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # For each iteration:
        #   1. Compute predictions with get_model_prediction(X, weights)
        #   2. For each weight index j, compute gradient with get_derivative()
        #   3. Update: weights[j] -= learning_rate * gradient
        # Return np.round(final_weights, 5)
        
        updated_weights = np.copy(initial_weights)

        for _ in range(num_iterations):
            y_pred = self.get_model_prediction(X, updated_weights)
            for i in range(len(updated_weights)):
                derivative = self.get_derivative(y_pred, Y, len(X), X, i)
                updated_weights[i] -= self.learning_rate*derivative

        return np.round(updated_weights, 5)

