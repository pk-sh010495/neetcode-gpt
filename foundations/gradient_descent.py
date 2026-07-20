class Solution:
    def __init__(self, x=None, derivative=None): 
        self.x = x
        self.derivative = derivative

    def update_rule(self, learning_rate: int) -> float:
        self.x = self.x - learning_rate * self.derivative
        
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        self.x = init

        for iteration in range(iterations):
            self.derivative = 2 * self.x
            self.update_rule(learning_rate)
        
        self.x = round(self.x, 5)
        return(self.x)
