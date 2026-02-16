# plots.py
from __future__ import annotations
import matplotlib.pyplot as plt


def plot_convergence(mui_hist, obj_hist, pow_hist, target_pow):
    plt.figure(figsize=(7, 4))
    plt.plot(mui_hist, label="PMUI")
    plt.plot(obj_hist, label="Objective")
    plt.plot(pow_hist, label="||X||_F^2")
    plt.axhline(target_pow, linestyle="--", label="M*P0")
    plt.xlabel("Stage / Iteration")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def plot_beampattern(angles_deg, Pd, Ptar, title="Beampattern"):
    plt.figure(figsize=(7, 4))
    plt.plot(angles_deg, Pd, label="Achieved")
    plt.plot(angles_deg, Ptar, label="Target")
    plt.xlabel("Angle (deg)")
    plt.ylabel("Beampattern (linear)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


class LivePlot:
    """
    Live plot 3 curves: PMUI, ||X||^2, Objective
    """
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 4))
        (self.l_mui,) = self.ax.plot([], [], label="PMUI")
        (self.l_pow,) = self.ax.plot([], [], label="||X||_F^2")
        (self.l_obj,) = self.ax.plot([], [], label="Objective")
        self.ax.set_xlabel("Iteration")
        self.ax.grid(True)
        self.ax.legend()

        self.mui_hist, self.pow_hist, self.obj_hist = [], [], []

    def update(self, t, mui, powv, obj, title=""):
        self.mui_hist.append(mui)
        self.pow_hist.append(powv)
        self.obj_hist.append(obj)

        xs = list(range(1, len(self.mui_hist) + 1))
        self.l_mui.set_data(xs, self.mui_hist)
        self.l_pow.set_data(xs, self.pow_hist)
        self.l_obj.set_data(xs, self.obj_hist)

        self.ax.relim()
        self.ax.autoscale_view()

        if title:
            self.ax.set_title(title)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def close(self):
        plt.ioff()
        plt.show()
