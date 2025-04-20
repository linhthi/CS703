import cvxpy as cp
import numpy as np

# Variables
y1 = cp.Variable(name="y1")
y2 = cp.Variable(name="y2")
y3 = cp.Variable(name="y3")
u  = cp.Variable(name="u")

# Constraints
constraints = [
    y1 - y2 - u <= 0,  # y1 - y2 - u <= 0
    0.5*y3 - y2 - u <= 0,  # (1/2) y3 - y2 - u <= 0
    # log(exp(2y1-0.5y2) + 2 exp(0.5y2 - y3)) <= 0
    cp.log_sum_exp(cp.hstack([
        2*y1 - 0.5*y2,
        cp.log(2) + 0.5*y2 - y3
    ])) <= 0,
    2*y3 + y2 - y1 <= 0  # 2y3 + y2 - y1 <= 0
]

obj = cp.Minimize(u) # objective

# Solve
prob = cp.Problem(obj, constraints)
prob.solve()

# Convert y back to x via x_i = exp(y_i)
x1_val, x2_val, x3_val = np.exp(y1.value), np.exp(y2.value), np.exp(y3.value)

print(f"Optimal y1 = {y1.value:.4f}, y2 = {y2.value:.4f}, y3 = {y3.value:.4f}, u = {u.value:.4f}")
print(f"Corresponding x1 = exp(y1) = {x1_val:.4f}")
print(f"Corresponding x2 = exp(y2) = {x2_val:.4f}")
print(f"Corresponding x3 = exp(y3) = {x3_val:.4f}")

