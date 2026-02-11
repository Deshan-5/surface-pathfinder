import numpy as np

#1. Paraboloid 
# eqn: f(x, y) = x^2 + y^2
def bowl_f(x, y):
    return x**2 + y**2

def bowl_grad(x, y):
    # Partial derivatives: df/dx = 2x, df/dy = 2y
    return np.array([2*x, 2*y])


# 2. The Saddle Hyperbolic Paraboloid
# eqn f(x, y) = x^2 - y^2
# This is tricky because one goes down, but the other goes up.
def saddle_f(x, y):
    return x**2 - y**2

def saddle_grad(x, y):
    # df/dx = 2x, df/dy = -2y
    return np.array([2*x, -2*y])


#3. The "Egg Carton" Multi-modal 
# eqn : f(x, y) = sin(x) + cos(y)
# This is great for testing if the pathfinder gets stuck in local valleys.
def ripples_f(x, y):
    return np.sin(x) + np.cos(y)

def ripples_grad(x, y):
    # df/dx = cos(x), df/dy = -sin(y)
    return np.array([np.cos(x), -np.sin(y)])
