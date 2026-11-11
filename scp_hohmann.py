from math import sqrt
import numpy as np
import cvxpy as cvxpy
import matplotlib.pyplot as plt
from visualizer import TransferVisualizer

# forward euler of initial guess
euler_dt = 0.01;
def forward_euler(x_0, time):
    x = np.zeros((4, int(time/euler_dt)), dtype=float);
    x[:, 0] = x_0.flatten();
    for k in range(int(time/euler_dt) - 1):
        x[:, k+1] = nonlinear_dynamics(x[:, k])*euler_dt + x[:, k];
    return x;

# definition of our nonlinear dynamics
def nonlinear_dynamics(state):
    x = state[0];
    y = state[1];
    v_x = state[2];
    v_y = state[3];

    return np.array([
        v_x, 
        v_y, 
        -mu/(sqrt(x**2+y**2))**3*x, 
        -mu/sqrt(x**2+y**2)**3*y
    ]);

# definition of our nonlinear dynamics
def linearized_dynamics(x, x_bar):
    f_bar = nonlinear_dynamics(x_bar);
    return f_bar + linearized_jacobian(x_bar) @ (x - x_bar);

# our linearized jacobian of our non-linear dynamics
def linearized_jacobian(state):
    x = state[0];
    y = state[1];
    r = sqrt(x**2 + y**2);

    jacobian = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [(3*mu*x**2)/r**5 - mu/r**3, 3*mu*x*y/r**5, 0, 0],
        [3*mu*x*y/r**5, 3*mu*y**2/r**5 - mu/r**3, 0, 0]
    ]);

    return jacobian;

# problem parameters
N = 5000;            # time steps
max_iter = 500;      # SCP iterations
trust_region_position = 100;
trust_region_velocity = 0.5;
slack = cvxpy.Variable((4, N - 1));
slack_weight = 1e3;

# hohmann transfer parameters
mu = 398600.44;  # Earth's gravitational parameter
r_p = 6378.137;  # Earth's radius in km

# orbits
r_1 = r_p + 191.35; # inital orbit
r_2 = r_p + 35781; # final orbit
v_initial = np.array([sqrt(mu/r_1), 0.0]).reshape(2, 1);
v_final = np.array([-sqrt(mu/r_2), 0.0]).reshape(2, 1);

x_inital = np.array([
    0, 
    -r_1, 
    sqrt(mu/r_1), 
    0
]);

x_final = np.array([
    0, 
    r_2, 
    -sqrt(mu/r_2),
    0
]);

# initial guess
est_time = np.pi*sqrt(((((r_1 + r_2)/2)**3)/(mu))); # comes from Hohmann transfer time
dt = est_time/(N - 1); # we take the estimated time of the trajectory and divide by number of scp nodes
initial_trajectory_guess = forward_euler(np.array([0, -r_1, 1.3*sqrt(mu/r_1), 0]).reshape(4, 1), est_time); # 130% of circular velocity is our initial guess

# downsample to N columns so SCP nodes match reference layout (4, N)
idx = np.linspace(0, initial_trajectory_guess.shape[1] - 1, num = N).astype(int);
reference_trajectory = initial_trajectory_guess[:, idx];

# scp
x_prev = reference_trajectory.copy();
trajectory_history = [x_prev.copy()];

for iteration in range(max_iter):
    # scp variables
    x = cvxpy.Variable((4, N));

    delta_v1 = cvxpy.Variable((2,));
    delta_v2 = cvxpy.Variable((2,));

    terminal_pos_slack = cvxpy.Variable(2);
    terminal_vel_slack = cvxpy.Variable(2);
    
    constraints = [];
    
    # boundary conditions
    constraints.append(x[:2, 0] == x_inital[:2].flatten());
    constraints.append(x[2:, 0] == delta_v1 + x_inital[2:].flatten()); # immediately after
    constraints.append(x[:2, -1] == x_final[:2].flatten() + terminal_pos_slack);
    constraints.append(x[2:, -1] + delta_v2 == x_final[2:].flatten() + terminal_vel_slack); # immediately before + terminal slack on vel
    
    # discretized dynamics
    for k in range(N-1):
        x_bar_k = x_prev[:, k];
        x_bar_k_1 = x_prev[:, k+1];
        # constraints.append(x[:, k+1] == (x[:, k] + linearized_dynamics(x[:, k], x_bar_k) * dt) + slack[:, k]); # forward euler leaves large discretization errors
        constraints.append(x[:, k+1] == (x[:, k] + (linearized_dynamics(x[:, k], x_bar_k) + linearized_dynamics(x[:, k+1], x_bar_k_1)) * dt/2) + slack[:, k]); # trapezoidal propogation

    # trust region for both position and velocity
    constraints.append(cvxpy.abs(x[:2] - x_prev[:2]) <= trust_region_position)
    constraints.append(cvxpy.abs(x[2:] - x_prev[2:]) <= trust_region_velocity)

    # objective
    objective = cvxpy.Minimize(
        cvxpy.norm(delta_v1, 2) + cvxpy.norm(delta_v2, 2) + slack_weight * cvxpy.sum(cvxpy.abs(slack)) + slack_weight * cvxpy.norm(terminal_pos_slack, 1) + slack_weight * cvxpy.norm(terminal_vel_slack, 1)
    )
    
    # solve the convex sub-problem
    prob = cvxpy.Problem(objective, constraints)
    prob.solve(solver=cvxpy.QOCO, verbose=True)
    
    if prob.status != cvxpy.OPTIMAL:
        print(f"Iteration {iteration}: Solver failed with status {prob.status}")
        break
        
    # check convergence
    max_diff = np.max(np.abs(x.value - x_prev))
    print(f"Iteration {iteration}: Max trajectory update shift = {max_diff:.4f}")
    
    # update nominal trajectory for the next iteration
    x_prev = x.value.copy()
    trajectory_history.append(x_prev.copy())
    
    # we didn't change much from previous guess, so we're done
    if max_diff < 1e-2:
        print("Converged!")
        break

dv1 = delta_v1.value
dv2 = delta_v2.value
print(f"delta_v1 = {dv1} km/s, |delta_v1| = {np.linalg.norm(dv1):.4f} km/s")
print(f"delta_v2 = {dv2} km/s, |delta_v2| = {np.linalg.norm(dv2):.4f} km/s")
print(f"total delta_v = {np.linalg.norm(dv1) + np.linalg.norm(dv2):.4f} km/s")
print("Position slack:", terminal_pos_slack.value)
print("Velocity slack:", terminal_vel_slack.value)
print("Dynamics slack:", np.sum(np.abs(slack.value)))

# visualize the SCP solution
visualizer = TransferVisualizer(mu, r_p, x_inital, x_final)
fig, animation, result = visualizer.plot_transfer(dv1, dv2, (N - 1)*dt, scp_trajectory=x_prev, animate=True)
plt.show()

