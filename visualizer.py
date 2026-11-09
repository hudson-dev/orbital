from math import sqrt, pi, ceil
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# same problem setup as scp_hohmann.py
mu = 398600.44;  # Earth's gravitational parameter
r_p = 6378.137;  # Earth's radius in km

r_1 = r_p + 191.35; # inital orbit
r_2 = r_p + 29402.863; # final orbit

x_inital = np.array([0, -r_1, sqrt(mu/r_1), 0]);
x_final = np.array([0, r_2, -sqrt(mu/r_2), 0]);

# same two-body dynamics as scp_hohmann.py
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

def rk4_step(state, h):
    k1 = nonlinear_dynamics(state);
    k2 = nonlinear_dynamics(state + h/2*k1);
    k3 = nonlinear_dynamics(state + h/2*k2);
    k4 = nonlinear_dynamics(state + h*k3);
    return state + h/6*(k1 + 2*k2 + 2*k3 + k4);

# propagate the state for `duration` seconds with a step no larger than `max_step`
def propagate(x_0, duration, max_step=10.0):
    steps = max(int(ceil(duration/max_step)), 1);
    h = duration/steps;
    t = np.linspace(0.0, duration, steps + 1);
    x = np.zeros((4, steps + 1));
    x[:, 0] = x_0;
    for k in range(steps):
        x[:, k+1] = rk4_step(x[:, k], h);
    return t, x;

def circular_period(r):
    return 2*pi*sqrt(r**3/mu);

# analytical Hohmann impulses, using the same sign convention as scp_hohmann.py:
#   v_after_burn_1 = v_initial + delta_v1
#   v_arrival + delta_v2 = v_final
def hohmann_impulses():
    a_t = (r_1 + r_2)/2;
    v_perigee = sqrt(mu*(2/r_1 - 1/a_t));
    v_apogee = sqrt(mu*(2/r_2 - 1/a_t));
    delta_v1 = np.array([v_perigee - sqrt(mu/r_1), 0.0]);
    delta_v2 = np.array([v_apogee - sqrt(mu/r_2), 0.0]);
    time_of_flight = pi*sqrt(a_t**3/mu);
    return delta_v1, delta_v2, time_of_flight;

def simulate_transfer(delta_v1, delta_v2, time_of_flight, max_step=10.0):
    delta_v1 = np.asarray(delta_v1, dtype=float).flatten();
    delta_v2 = np.asarray(delta_v2, dtype=float).flatten();

    _, initial_orbit = propagate(x_inital, circular_period(r_1), max_step);
    _, target_orbit = propagate(x_final, circular_period(r_2), max_step);

    x_after_burn_1 = x_inital.astype(float).copy();
    x_after_burn_1[2:] += delta_v1;
    t_transfer, transfer = propagate(x_after_burn_1, time_of_flight, max_step);

    x_after_burn_2 = transfer[:, -1].copy();
    x_after_burn_2[2:] += delta_v2;
    t_coast, coast = propagate(x_after_burn_2, circular_period(r_2), max_step);

    return {
        "delta_v1": delta_v1,
        "delta_v2": delta_v2,
        "time_of_flight": time_of_flight,
        "initial_orbit": initial_orbit,
        "target_orbit": target_orbit,
        "t_transfer": t_transfer,
        "transfer": transfer,
        "t_coast": t_coast,
        "coast": coast,
        "position_error": np.linalg.norm(transfer[:2, -1] - x_final[:2]),
        "velocity_error": np.linalg.norm(x_after_burn_2[2:] - x_final[2:]),
    };

def print_summary(result):
    dv1 = result["delta_v1"];
    dv2 = result["delta_v2"];
    print(f"delta_v1 = {dv1} km/s, |delta_v1| = {np.linalg.norm(dv1):.4f} km/s");
    print(f"delta_v2 = {dv2} km/s, |delta_v2| = {np.linalg.norm(dv2):.4f} km/s");
    print(f"total delta_v = {np.linalg.norm(dv1) + np.linalg.norm(dv2):.4f} km/s");
    print(f"time of flight = {result['time_of_flight']:.2f} s ({result['time_of_flight']/3600:.3f} h)");
    print(f"arrival position error vs target = {result['position_error']:.3f} km");
    print(f"post-burn velocity error vs target = {result['velocity_error']:.5f} km/s");

def _draw_burn_arrow(ax, position, delta_v, length, color, label):
    norm = np.linalg.norm(delta_v);
    if norm < 1e-12:
        return;
    direction = delta_v/norm*length;
    ax.annotate(
        "", xy=position + direction, xytext=position,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5, mutation_scale=18),
        zorder=6,
    );
    ax.text(*(position + 1.15*direction), f"{label}\n|Δv| = {norm:.3f} km/s",
            color=color, fontsize=9, ha="center", va="center", zorder=6);

def visualize_transfer(delta_v1, delta_v2, time_of_flight, scp_trajectory=None,
                       animate=False, save_path=None, max_step=10.0):
    result = simulate_transfer(delta_v1, delta_v2, time_of_flight, max_step);
    print_summary(result);

    transfer = result["transfer"];
    coast = result["coast"];

    fig, (ax, ax_r) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1.4, 1]});

    # orbit plane view
    ax.add_patch(plt.Circle((0, 0), r_p, color="tab:blue", alpha=0.35, zorder=1));
    ax.plot(*result["initial_orbit"][:2], "--", color="gray", lw=1, label=f"Initial orbit (r = {r_1:.0f} km)");
    ax.plot(*result["target_orbit"][:2], "--", color="black", lw=1, label=f"Target orbit (r = {r_2:.0f} km)");
    ax.plot(*coast[:2], color="tab:green", lw=1, alpha=0.6, label="Coast after Δv2");
    ax.plot(*transfer[:2], color="tab:red", lw=2.5, label="Transfer arc", zorder=4);

    if scp_trajectory is not None:
        ax.plot(scp_trajectory[0], scp_trajectory[1], "o", ms=4, color="tab:purple",
                label="SCP solution nodes", zorder=5);

    ax.plot(*x_final[:2], "*", ms=14, color="gold", mec="black", label="Target point", zorder=7);

    arrow_length = 0.15*r_2;
    _draw_burn_arrow(ax, x_inital[:2], result["delta_v1"], arrow_length, "tab:orange", "Δv1");
    _draw_burn_arrow(ax, transfer[:2, -1], result["delta_v2"], arrow_length, "tab:purple", "Δv2");

    ax.set_aspect("equal");
    ax.set_xlabel("x [km]");
    ax.set_ylabel("y [km]");
    ax.set_title("Impulsive transfer (two-body dynamics)");
    ax.grid(alpha=0.3);
    ax.legend(loc="upper right", fontsize=8);

    # radius over time
    t_all = np.concatenate([result["t_transfer"], result["t_transfer"][-1] + result["t_coast"][1:]]);
    r_all = np.linalg.norm(np.concatenate([transfer[:2], coast[:2, 1:]], axis=1), axis=0);
    ax_r.plot(t_all/3600, r_all, color="tab:red");
    ax_r.axhline(r_1, ls="--", color="gray", lw=1, label="Initial orbit radius");
    ax_r.axhline(r_2, ls="--", color="black", lw=1, label="Target orbit radius");
    ax_r.axhline(r_p, ls=":", color="tab:blue", lw=1, label="Earth surface");
    ax_r.axvline(time_of_flight/3600, color="tab:purple", lw=1, label="Δv2 applied");
    ax_r.set_xlabel("time [h]");
    ax_r.set_ylabel("radius [km]");
    ax_r.set_title("Radius vs time");
    ax_r.grid(alpha=0.3);
    ax_r.legend(fontsize=8);

    fig.suptitle(
        f"total Δv = {np.linalg.norm(result['delta_v1']) + np.linalg.norm(result['delta_v2']):.3f} km/s, "
        f"time of flight = {time_of_flight/3600:.2f} h, "
        f"arrival miss = {result['position_error']:.1f} km"
    );
    fig.tight_layout();

    animation = None;
    if animate:
        path = np.concatenate([transfer[:2], coast[:2, 1:]], axis=1);
        frames = np.linspace(0, path.shape[1] - 1, num=min(400, path.shape[1])).astype(int);
        craft, = ax.plot([], [], "o", ms=8, color="black", zorder=8);
        trail, = ax.plot([], [], color="black", lw=1.5, zorder=7);
        marker_r, = ax_r.plot([], [], "o", ms=6, color="black");

        def update(i):
            k = frames[i];
            craft.set_data([path[0, k]], [path[1, k]]);
            trail.set_data(path[0, :k+1], path[1, :k+1]);
            marker_r.set_data([t_all[k]/3600], [r_all[k]]);
            return craft, trail, marker_r;

        animation = FuncAnimation(fig, update, frames=len(frames), interval=30, blit=True);

    if save_path:
        if animation is not None:
            animation.save(save_path, fps=30);
        else:
            fig.savefig(save_path, dpi=150);
        print(f"saved to {save_path}");
    else:
        plt.show();

    return result;

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a transfer orbit from two delta-v impulses.");
    parser.add_argument("--dv1", type=float, nargs=2, metavar=("DVX", "DVY"),
                        help="first impulse [km/s], applied at the initial state (default: Hohmann)");
    parser.add_argument("--dv2", type=float, nargs=2, metavar=("DVX", "DVY"),
                        help="second impulse [km/s], applied at arrival (default: Hohmann)");
    parser.add_argument("--tof", type=float, help="time of flight between impulses [s] (default: Hohmann)");
    parser.add_argument("--animate", action="store_true", help="animate the spacecraft along the trajectory");
    parser.add_argument("--save", help="save the figure (.png) or animation (.gif/.mp4) instead of showing it");
    args = parser.parse_args();

    hohmann_dv1, hohmann_dv2, hohmann_tof = hohmann_impulses();
    visualize_transfer(
        args.dv1 if args.dv1 is not None else hohmann_dv1,
        args.dv2 if args.dv2 is not None else hohmann_dv2,
        args.tof if args.tof is not None else hohmann_tof,
        animate=args.animate,
        save_path=args.save,
    );
