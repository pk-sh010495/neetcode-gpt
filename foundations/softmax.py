import numpy as np
from numpy.typing import NDArray


class Solution:
    def __init__(self):
        pass
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array 
        # Formula: 1 / (1 + e^(-z))
        sigmoid_z = 1 / (1 + np.exp(-z))
        # return np.round(your_answer, 5)
        return np.round(sigmoid_z, 5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        relu_z = np.maximum(0, z)
        return relu_z
