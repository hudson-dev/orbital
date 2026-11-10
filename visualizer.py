from math import sqrt, pi, ceil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class TransferVisualizer:
    # delta-v sign convention matches scp_hohmann.py:
    #   v_after_burn_1 = v_initial + delta_v1
    #   v_arrival + delta_v2 = v_final
    def __init__(self, mu, r_p, x_initial, x_final, max_step=10.0):
        self.mu = mu;
        self.r_p = r_p;
        self.x_initial = np.asarray(x_initial, dtype=float).flatten();
        self.x_final = np.asarray(x_final, dtype=float).flatten();
        self.r_1 = np.linalg.norm(self.x_initial[:2]);
        self.r_2 = np.linalg.norm(self.x_final[:2]);
        self.max_step = max_step;

    def nonlinear_dynamics(self, state):
        x, y, v_x, v_y = state;
        r3 = sqrt(x**2 + y**2)**3;
        return np.array([v_x, v_y, -self.mu/r3*x, -self.mu/r3*y]);

    def rk4_step(self, state, h):
        k1 = self.nonlinear_dynamics(state);
        k2 = self.nonlinear_dynamics(state + h/2*k1);
        k3 = self.nonlinear_dynamics(state + h/2*k2);
        k4 = self.nonlinear_dynamics(state + h*k3);
        return state + h/6*(k1 + 2*k2 + 2*k3 + k4);

    # propagate the state for `duration` seconds with a step no larger than `max_step`
    def propagate(self, x_0, duration):
        steps = max(int(ceil(duration/self.max_step)), 1);
        h = duration/steps;
        t = np.linspace(0.0, duration, steps + 1);
        x = np.zeros((4, steps + 1));
        x[:, 0] = x_0;
        for k in range(steps):
            x[:, k+1] = self.rk4_step(x[:, k], h);
        return t, x;

    def circular_period(self, r):
        return 2*pi*sqrt(r**3/self.mu);

    # analytical Hohmann impulses between the initial and final radii, useful as a reference
    def hohmann_impulses(self):
        a_t = (self.r_1 + self.r_2)/2;
        v_perigee = sqrt(self.mu*(2/self.r_1 - 1/a_t));
        v_apogee = sqrt(self.mu*(2/self.r_2 - 1/a_t));
        delta_v1 = np.array([v_perigee - sqrt(self.mu/self.r_1), 0.0]);
        delta_v2 = np.array([v_apogee - sqrt(self.mu/self.r_2), 0.0]);
        time_of_flight = pi*sqrt(a_t**3/self.mu);
        return delta_v1, delta_v2, time_of_flight;

    def simulate_transfer(self, delta_v1, delta_v2, time_of_flight):
        delta_v1 = np.asarray(delta_v1, dtype=float).flatten();
        delta_v2 = np.asarray(delta_v2, dtype=float).flatten();

        x_after_burn_1 = self.x_initial.copy();
        x_after_burn_1[2:] += delta_v1;
        t_transfer, transfer = self.propagate(x_after_burn_1, time_of_flight);

        x_after_burn_2 = transfer[:, -1].copy();
        x_after_burn_2[2:] += delta_v2;
        t_coast, coast = self.propagate(x_after_burn_2, self.circular_period(self.r_2));

        return {
            "delta_v1": delta_v1,
            "delta_v2": delta_v2,
            "time_of_flight": time_of_flight,
            "t_transfer": t_transfer,
            "transfer": transfer,
            "t_coast": t_coast,
            "coast": coast,
            "position_error": np.linalg.norm(transfer[:2, -1] - self.x_final[:2]),
            "velocity_error": np.linalg.norm(x_after_burn_2[2:] - self.x_final[2:]),
        };

    def _draw_burn_arrow(self, ax, position, delta_v, length, color, label):
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

    # returns (fig, animation, result); animation is None unless animate=True.
    # keep a reference to the animation until plt.show() or it gets garbage collected
    def plot_transfer(self, delta_v1, delta_v2, time_of_flight, scp_trajectory=None, animate=False):
        result = self.simulate_transfer(delta_v1, delta_v2, time_of_flight);
        transfer = result["transfer"];
        coast = result["coast"];

        fig, (ax, ax_r) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1.4, 1]});

        # orbit plane view
        theta = np.linspace(0, 2*pi, 400);
        ax.add_patch(plt.Circle((0, 0), self.r_p, color="tab:blue", alpha=0.35, zorder=1));
        ax.plot(self.r_1*np.cos(theta), self.r_1*np.sin(theta), "--", color="gray", lw=1,
                label=f"Initial orbit (r = {self.r_1:.0f} km)");
        ax.plot(self.r_2*np.cos(theta), self.r_2*np.sin(theta), "--", color="black", lw=1,
                label=f"Target orbit (r = {self.r_2:.0f} km)");
        ax.plot(*coast[:2], color="tab:green", lw=1, alpha=0.6, label="Coast after Δv2");
        ax.plot(*transfer[:2], color="tab:red", lw=2.5, label="Transfer arc", zorder=4);

        if scp_trajectory is not None:
            ax.plot(scp_trajectory[0], scp_trajectory[1], "o", ms=4, color="tab:purple",
                    label="SCP solution nodes", zorder=5);

        ax.plot(*self.x_final[:2], "*", ms=14, color="gold", mec="black", label="Target point", zorder=7);

        arrow_length = 0.15*self.r_2;
        self._draw_burn_arrow(ax, self.x_initial[:2], result["delta_v1"], arrow_length, "tab:orange", "Δv1");
        self._draw_burn_arrow(ax, transfer[:2, -1], result["delta_v2"], arrow_length, "tab:purple", "Δv2");

        # the coast can escape to huge radii, so frame the view on the target orbit and transfer arc only
        extent = 1.15*max(self.r_2, np.max(np.abs(transfer[:2])));
        ax.set_xlim(-extent, extent);
        ax.set_ylim(-extent, extent);
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
        ax_r.axhline(self.r_1, ls="--", color="gray", lw=1, label="Initial orbit radius");
        ax_r.axhline(self.r_2, ls="--", color="black", lw=1, label="Target orbit radius");
        ax_r.axhline(self.r_p, ls=":", color="tab:blue", lw=1, label="Earth surface");
        ax_r.axvline(time_of_flight/3600, color="tab:purple", lw=1, label="Δv2 applied");
        ax_r.set_ylim(0, extent);
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

        return fig, animation, result;
