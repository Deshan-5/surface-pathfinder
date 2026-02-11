import numpy as np

class Pathfinder:
    """
    A gradient descent engine designed for 3D surfaces and this 
    class handles the logic of walking down a mathematical function.
    """
    def __init__(self, learning_rate=0.01, precision=1e-6, max_iters=1000):
        self.lr = learning_rate
        self.precision = precision
        self.max_iters = max_iters
      
# 1. learning rate-->  parameter reasoning. 
         # (e.g: 0.9): we  might overshoot the valley and fly off the map.--> too big 
         # (e.g:  0.0001): The pathfinder will move at a turtle's pace. --> too small
         # 0.01 is the standard  and safe default for modern optimizers. 
# 2. precision -> the bottom line 
        # In calculus, the bottom of a hill has a slope/gradient/ of 0.
        # Since code can't always reach exact 0, we stop when the slope is 
        # tiny enough (for ex: 0.000001) that it doesn't matter anymore.
# 3. max_iters -> emergency stop.
        # If the surface is a flat plane or an infinite slide, the pathfinder
        # could run forever. its a hard-limit  to prevent CPU from hanging.

    def minimize(self, start_pos, grad_func):
        """
        Executes the descent from a starting (x, y) point.
        grad_func: A function that returns partial derivatives [df/dx, df/dy].
        """
      
        current_pos = np.array(start_pos, dtype=float) # Ensures start_pos is a float array for math precision
        history = [current_pos.copy()]   # Tracks the path for the 3D visualization later.

        for i in range(self.max_iters):  # Calculates the steepness at the current location.
           grad = np.array(grad_func(*current_pos))
            
            # convergence  check :
            # If the magnitude of the gradient is less than our precision,
            # we are standing on flat ground the bottom.  so we have to Stop early..
            if np.linalg.norm(grad) < self.precision:
                print(f"Converged at iteration {i}!")
                break
                
            # THE UPDATE RULE:
            # New_Position = Old_Position - (Step_Size * Direction_of_Steepest_Slope)
            # We subtract because the gradient points uphill but  we want to go down hill.
            current_pos -= self.lr * grad
            
            # Records the step
            history.append(current_pos.copy())

        return np.array(history)
