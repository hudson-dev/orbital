import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# 1. Problem Parameters
N = 40          # Time steps
dt = 0.5        # Time increment
max_iter = 10   # SCP iterations
trust_region = 1.0

start = np.array([0.0, 0.0])
goal = np.array([10.0, 10.0])

# Obstacle: Circle centered at (5,5) with radius 2
obs_center = np.array([7.0, 8.0])
obs_radius = 2.0
slack = cp.Variable(N - 1, nonneg=True)
slack_weight = 1e3

# 2. Initialize Trajectory (Straight line guess from start to goal)
# This initial guess intentionally slices straight through the obstacle.
p_nom = np.zeros((N, 2))
for i in range(N):
    p_nom[i] = start + (goal - start) * (i / (N - 1))

# 3. SCP Main Loop
trajectory_history = [p_nom.copy()]

for iteration in range(max_iter):
    # Setup CVXPY Variables
    p = cp.Variable((N, 2))  # Positions
    u = cp.Variable((N-1, 2))  # Control inputs (velocities)
    
    constraints = []
    
    # Boundary Conditions
    constraints.append(p[0] == start)
    constraints.append(p[N-1] == goal)
    
    # Linear Kinematics: p[t+1] = p[t] + u[t]*dt
    for t in range(N-1):
        constraints.append(p[t+1] == p[t] + u[t] * dt)

    ## Obstacle constraint linearization
    # Optional clearance beyond the obstacle boundary
    clearance = 0.01
    safe_radius = obs_radius + clearance

    ## QUESTION: ## 
    # why do we do p-p_nom here? Is this just making sure we don't go too far from previous guess?
    # but my understanding was that the point of a trust region was to make sure we don't stray too far from the linearization

    # Trust regions apply to every node
    # note p_nom is the previous guess, p is the current guess
    constraints.append(cp.abs(p - p_nom) <= trust_region)

    # Protect entire segments using one shared half-space per segment
    for t in range(N - 1):
        a = p_nom[t]
        b = p_nom[t + 1]
        segment = b - a

        # Closest point on the nominal segment to the obstacle center
        # two ways to compute the closest point:
        # use dot product or just take derivative and set to 0
        length_sq = segment @ segment
        if length_sq > 1e-12:
            fraction = np.clip(
                ((obs_center - a) @ segment) / length_sq,
                0.0,
                1.0,
            )
            closest = a + fraction * segment
        else:
            closest = a

        outward = closest - obs_center
        distance = np.linalg.norm(outward)

        normal = outward / distance

        constraints.extend([
            normal @ (p[t] - obs_center) >= safe_radius - slack[t],
            normal @ (p[t + 1] - obs_center) >= safe_radius - slack[t],
        ])

    # Objective: Minimize control effort and obstacle slack
    objective = cp.Minimize(
        cp.sum_squares(u) + slack_weight * cp.sum(slack)
    )
    
    # Solve the convex sub-problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.QOCO)
    
    if prob.status != cp.OPTIMAL:
        print(f"Iteration {iteration}: Solver failed with status {prob.status}")
        break
        
    # Check convergence (max change in position)
    max_diff = np.max(np.abs(p.value - p_nom))
    print(f"Iteration {iteration}: Max trajectory update shift = {max_diff:.4f}")
    
    # Update nominal trajectory for the next iteration
    p_nom = p.value.copy()
    trajectory_history.append(p_nom.copy())
    
    # We didn't change much from previous guess, so we're done
    if max_diff < 1e-2:
        print("Converged!")
        break

# 4. Plotting Results
fig, ax = plt.subplots(figsize=(6, 6))

circle = plt.Circle(
    obs_center, obs_radius,
    color="r", alpha=0.4, label="Obstacle"
)
ax.add_patch(circle)

# Initial guess and all intermediate iterations
for i, trajectory in enumerate(trajectory_history[:-1]):
    ax.plot(
        trajectory[:, 0], trajectory[:, 1],
        color="blue", alpha=0.2, linewidth=1,
        label="Previous iterations" if i == 0 else None,
    )

# Final trajectory
ax.plot(
    p_nom[:, 0], p_nom[:, 1],
    "-o", color="blue", linewidth=2, markersize=3,
    label="Final path",
)

ax.scatter(
    [start[0], goal[0]], [start[1], goal[1]],
    color="black", zorder=5,
)

ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_aspect("equal", adjustable="box")
ax.legend()
ax.grid(True)
plt.show()