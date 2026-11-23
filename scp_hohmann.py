from math import sqrt
import time
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

def interpolated(r_1, r_2):
    circle_center_y = (r_2 - r_1) / 2;
    circle_radius = (r_1 + r_2) / 2;

    theta = np.linspace(-np.pi/2, np.pi/2, N); # bottom of circle (start) to top (target)
    speed = np.linspace(sqrt(mu/r_1), sqrt(mu/r_2), N); # blend from initial to final circular speed

    x = np.zeros((4, N));
    x[0] = circle_radius*np.cos(theta);
    x[1] = circle_center_y + circle_radius*np.sin(theta);
    x[2] = -speed*np.sin(theta);
    x[3] = speed*np.cos(theta);
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

def rk4(f, x_k, s_k, dt):
    f_1 = f(x_k, s_k);
    f_2 = f(x_k + dt/2 * f_1, s_k);
    f_3 = f(x_k + dt/2 * f_2, s_k);
    f_4 = f(x_k + dt * f_3, s_k);

    return x_k + dt/6*(f_1 + 2*f_2 + 2*f_3 + f_4);

# time-dilated dynamics: s stretches or shrinks how much real time one interval covers
def dilated_dynamics(x, s):
    return s*nonlinear_dynamics(x);

# propagate the real nonlinear dynamics across one interval, using a few small rk4 steps for accuracy
def propagate(x_k, s_k, dt, substeps = 10):
    for _ in range(substeps):
        x_k = rk4(dilated_dynamics, x_k, s_k, dt/substeps);
    return x_k;

# linearize propagate() around (x_bar, s_bar) by nudging each input and seeing how the output moves:
# x_{k+1} ≈ x_next_bar + A @ (x_k - x_bar) + B * (s_k - s_bar)
def linearize_propagation(x_bar, s_bar, dt):
    x_next_bar = propagate(x_bar, s_bar, dt);

    A = np.zeros((4, 4));
    for i in range(4):
        nudge = np.zeros(4);
        nudge[i] = 1e-6*max(abs(x_bar[i]), 1.0);
        A[:, i] = (propagate(x_bar + nudge, s_bar, dt) - propagate(x_bar - nudge, s_bar, dt)) / (2*nudge[i]);

    s_nudge = 1e-6;
    B = (propagate(x_bar, s_bar + s_nudge, dt) - propagate(x_bar, s_bar - s_nudge, dt)) / (2*s_nudge);

    return x_next_bar, A, B;

# problem parameters
N = 50;            # nodees
max_iter = 50;    # SCP iterations
trust_region_position = 2000;
trust_region_velocity = 1.0;
trust_region_time_dilation = 0.2;
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

# scaling: the solver works in units where the initial orbit radius = 1 and mu = 1,
# so every number it sees is roughly 0.1 to 10 instead of spanning 1e-23 to 1e4
distance_unit = r_1;                      # km
time_unit = sqrt(r_1**3/mu);              # s
velocity_unit = distance_unit/time_unit;  # km/s (circular speed of the initial orbit)
S = np.diag([distance_unit, distance_unit, velocity_unit, velocity_unit]); # state in km = S @ scaled state
S_inv = np.linalg.inv(S);                                                  # scaled state = S_inv @ state in km

# initial guess
est_time = np.pi*sqrt(((((r_1 + r_2)/2)**3)/(mu))); # comes from Hohmann transfer time
dt = est_time/(N - 1); # we take the estimated time of the trajectory and divide by number of scp nodes
initial_guess_start = time.perf_counter();
initial_trajectory_guess = interpolated(r_1, r_2);
# initial_trajectory_guess = forward_euler(np.array([0, -r_1, 1.2*sqrt(mu/r_1), 0]).reshape(4, 1), est_time); # 130% of circular velocity is our initial guess
initial_guess_time = time.perf_counter() - initial_guess_start;

# downsample to N columns so SCP nodes match reference layout (4, N)
idx = np.linspace(0, initial_trajectory_guess.shape[1] - 1, num = N).astype(int);
reference_trajectory = initial_trajectory_guess[:, idx];

# scp
x_prev = reference_trajectory.copy();
trajectory_history = [x_prev.copy()];

s_prev = np.ones(N - 1);

scp_start = time.perf_counter();
for iteration in range(max_iter):
    iteration_start = time.perf_counter();

    # scp variables (x, delta_v1, delta_v2 and the slacks are all in scaled units)
    x = cvxpy.Variable((4, N));
    s = cvxpy.Variable(N - 1);

    delta_v1 = cvxpy.Variable((2,));
    delta_v2 = cvxpy.Variable((2,));

    terminal_pos_slack = cvxpy.Variable(2);
    terminal_vel_slack = cvxpy.Variable(2);

    x_prev_scaled = S_inv @ x_prev;
    
    constraints = [];
    
    # boundary condition constraints
    constraints.append(x[:2, 0] == x_inital[:2]/distance_unit);
    constraints.append(x[2:, 0] == delta_v1 + x_inital[2:]/velocity_unit); # immediately after
    constraints.append(x[:2, -1] == x_final[:2]/distance_unit + terminal_pos_slack);
    constraints.append(x[2:, -1] + delta_v2 == x_final[2:]/velocity_unit + terminal_vel_slack); # immediately before + terminal slack on vel

    # free final time & time dilation constraints (s has no units, so it doesn't need scaling)
    total_time = cvxpy.sum(s)*dt/time_unit;
    constraints.append(0.7*est_time/time_unit <= total_time);
    constraints.append(total_time <= 1.3*est_time/time_unit)
    constraints.append(0.2 <= s);
    constraints.append(s <= 2.0);
    constraints.append(cvxpy.abs(s - s_prev) <= trust_region_time_dilation);

    # trust region for both position and velocity
    constraints.append(cvxpy.abs(x[:2] - x_prev_scaled[:2]) <= trust_region_position/distance_unit)
    constraints.append(cvxpy.abs(x[2:] - x_prev_scaled[2:]) <= trust_region_velocity/velocity_unit)
    
    # discretized dynamics
    for k in range(N-1):
        x_bar_k = x_prev[:, k];
        s_bar_k = s_prev[k];
        x_next_bar, A_k, B_k = linearize_propagation(x_bar_k, s_bar_k, dt); # in km

        # convert the linearization to scaled units: S turns a scaled input into km, A_k propagates it, S_inv scales the output back
        x_next_bar_scaled = S_inv @ x_next_bar;
        A_k_scaled = S_inv @ A_k @ S;
        B_k_scaled = S_inv @ B_k;

        constraints.append(x[:, k+1] == x_next_bar_scaled + A_k_scaled @ (x[:, k] - x_prev_scaled[:, k]) + B_k_scaled*(s[k] - s_bar_k) + slack[:, k]);

    # objective
    objective = cvxpy.Minimize(
        cvxpy.norm(delta_v1, 2) + cvxpy.norm(delta_v2, 2) + slack_weight * cvxpy.sum(cvxpy.abs(slack)) + slack_weight * cvxpy.norm(terminal_pos_slack, 1) + slack_weight * cvxpy.norm(terminal_vel_slack, 1) + 0.5 * cvxpy.sum_squares(cvxpy.diff(s))
    )
    
    # solve the convex sub-problem
    prob = cvxpy.Problem(objective, constraints)
    prob.solve(solver=cvxpy.QOCO, verbose=False)
    
    if prob.status != cvxpy.OPTIMAL:
        print(f"Iteration {iteration}: Solver failed with status {prob.status}")
        break
        
    # convert the solution back to km before comparing
    x_new = S @ x.value;

    # check convergence
    max_x_diff = np.max(np.abs(x_new - x_prev))
    max_s_diff = np.max(np.abs(s.value - s_prev))
    print(f"Iteration {iteration}: Max trajectory update shift = {max_x_diff:.4f}, took {time.perf_counter() - iteration_start:.3f} s")
    
    # update nominal trajectory for the next iteration
    x_prev = x_new.copy()
    s_prev = s.value.copy()

    trajectory_history.append(x_prev.copy())
    
    # we didn't change much from previous guess, so we're done
    if max_x_diff < 1e-2 and max_s_diff < 1e-3:
        print("Converged!")
        break

scp_time = time.perf_counter() - scp_start;
print(f"Initial guess took {initial_guess_time:.2f} s");
print(f"SCP took {scp_time:.2f} s for {iteration + 1} iterations ({scp_time/(iteration + 1):.3f} s per iteration)");

# convert the solver's scaled answers back to km and km/s
dv1 = delta_v1.value*velocity_unit
dv2 = delta_v2.value*velocity_unit
print(f"delta_v1 = {dv1} km/s, |delta_v1| = {np.linalg.norm(dv1):.4f} km/s")
print(f"delta_v2 = {dv2} km/s, |delta_v2| = {np.linalg.norm(dv2):.4f} km/s")
print(f"total delta_v = {np.linalg.norm(dv1) + np.linalg.norm(dv2):.4f} km/s")
print("Position slack (km):", terminal_pos_slack.value*distance_unit)
print("Velocity slack (km/s):", terminal_vel_slack.value*velocity_unit)
print("Dynamics slack (km, km/s):", np.sum(np.abs(S @ slack.value), axis=1))

# visualize the SCP solution
visualizer = TransferVisualizer(mu, r_p, x_inital, x_final)
final_time = dt * np.sum(s_prev)
time_nodes = np.r_[0, np.cumsum(dt * s_prev)]

print(f"Optimized transfer time: {final_time:.2f} s")
print("Time intervals:", np.diff(time_nodes))

fig, animation, result = visualizer.plot_transfer(
    dv1, dv2, final_time,
    scp_trajectory=x_prev,
    animate=True
)
plt.show()

