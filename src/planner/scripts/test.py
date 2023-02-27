import gurobipy as gp
import numpy as np

# Define the objective function
def rosenbrock(x):
    N = len(x)
    f = sum(100*(x[i+1]-x[i]**2)**2 + (1-x[i])**2 for i in range(N-1))
    return f

# Define the gradient function
def rosenbrock_grad(x):
    N = len(x)
    grad = np.zeros(N)
    grad[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    grad[N-1] = 200*(x[N-1]-x[N-2]**2)
    for i in range(1, N-1):
        grad[i] = -400*x[i]*(x[i+1]-x[i]**2) - 2*(1-x[i]) + 200*(x[i]-x[i-1]**2)
    return grad

# Create a model
model = gp.Model()

# Add variables to the model
N = 3 # number of variables
x = model.addMVar(shape=N, lb=-10, ub=10)

# Set the objective function and gradient
model.setObjectiveN(rosenbrock, 0, x)
model.setObjectiveN(rosenbrock_grad, 1, x)

# Optimize the model
model.optimize()

# Get the solution
solution = model.getAttr('x', x)
