from pathlib import Path

import matplotlib as mpl
import numpy as np
from gol_patterns import apply_pattern, glider, lwss
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm

from pvyzmgs.animation import animate, animate_with_plot, no_animation
from pvyzmgs.data import read_data, save_data
from pvyzmgs.simulation import Simulation, progress_bar, run_until


class GOL(Simulation):
    """
    Game of Life cellular automaton in 2 dimensions.
    """

    def __init__(self, shape):
        super().__init__(shape)
        self.history = [0] * 5  # store measurements from last few iterations

    def randomize(self):
        """
        Randomize the contents of the simulation grid.
        """
        self.grid = np.random.choice([0, 1], self.grid.shape)
        return self

    def update(self):
        """
        Update grid according to game of life rules using Moore neighborhood.
        """
        # store number of living Moore neighbors in its own grid
        neighbors = (
            np.roll(self.grid, shift=(0, 1), axis=(0, 1))
            + np.roll(self.grid, shift=(1, 1), axis=(0, 1))
            + np.roll(self.grid, shift=(1, 0), axis=(0, 1))
            + np.roll(self.grid, shift=(1, -1), axis=(0, 1))
            + np.roll(self.grid, shift=(0, -1), axis=(0, 1))
            + np.roll(self.grid, shift=(-1, -1), axis=(0, 1))
            + np.roll(self.grid, shift=(-1, 0), axis=(0, 1))
            + np.roll(self.grid, shift=(-1, 1), axis=(0, 1))
        )

        # game of life update rule
        alive = ((self.grid == 1) & ((neighbors == 2) | (neighbors == 3))) | (
            (self.grid == 0) & (neighbors == 3)
        )
        self.grid = np.zeros(shape=self.grid.shape)
        self.grid[alive] = 1

        # update simulation history
        self.history.append(self.alive)
        del self.history[0]

    def run(self, iterations):
        """
        Run an game of life simulation.
        """
        return super().run(iterations)

    @property
    def alive(self):
        """
        Get total number of living cells.
        """
        return np.sum(self.grid)

    @property
    def com(self):
        """
        Get the center of mass of living cells.
        """
        return np.sum(np.where(self.grid == 1), axis=1)[::-1] / self.alive

    @property
    def evolving(self):
        """
        Boolean value for whether the simulation is yet to reach equilibrium.
        Oscillators with a changing number of living cells breaks this.
        """
        end = self.history[0]
        return not all(value == end for value in self.history)


def measure_equilibration_time(sim, N=100, *, path=None, max_iter=10_000):
    """
    Measure the time needed to reach equilibrium from a random initial state over N trials.
    """
    lifetimes = []
    headers = ("Trial, Equilibration Time",)
    for trial in tqdm(range(N)):
        gen = sim.clear().randomize().run(max_iter)
        gen = run_until(gen, lambda s: not s.evolving)
        # gen = progress_bar(gen, desc=f'{trial=}', total=max_iter)
        no_animation(gen)
        lifetimes.append((trial, sim.elapsed))

    return lifetimes, headers
    if path:
        save_data(lifetimes, path=path, headers=headers)


def display_equilibration_time(path):
    """
    Display a histogram of equilibration times.
    """
    _, lifetimes = read_data(path)

    rcParams = {
        "figure.autolayout": True,
        "axes.linewidth": 0.8,
        "font.size": 8,
        "legend.fontsize": 10,
        "legend.handlelength": 2.0,
        "xtick.direction": "in",
        "xtick.major.size": 5.0,
        "xtick.minor.size": 3.0,
        "ytick.direction": "in",
        "ytick.major.size": 5.0,
        "ytick.minor.size": 3.0,
    }
    with mpl.rc_context(rcParams):
        fig, ax = plt.subplots()
        ax.hist(lifetimes, bins=200, density=True)
        ax.set_xlabel("Equilibration time")
        ax.set_ylabel("Frequency density")
        plt.show()


def measure_speed_of(pattern, start=(0, 0), steps=50):
    """
    Measure the speed of a spaceship using its center of mass.
    """
    xs = []
    ys = []
    sim = GOL((50, 50))
    apply_pattern(sim.grid, pattern, start)
    for state in sim.run(steps):
        x, y = state.com
        xs.append(x)
        ys.append(y)

    def f(x, a, b):
        return a * x + b

    inds = np.arange(len(xs))
    (a1, b1), pcov = curve_fit(f, inds, xs)
    (a2, b2), pcov = curve_fit(f, inds, ys)
    print(f"Speed: {np.sqrt(a1**2 + a2**2)} (euclidean metric)")
    print(f"Speed: {a1+a2} (taxicab metric)")

    rcParams = {
        "figure.autolayout": True,
        "axes.grid": True,
        "axes.linewidth": 0.8,
        "font.size": 8,
        "grid.color": "silver",
        "legend.fontsize": 10,
        "legend.handlelength": 2.0,
        "xtick.direction": "in",
        "xtick.major.size": 5.0,
        "xtick.minor.size": 3.0,
        "ytick.direction": "in",
        "ytick.major.size": 5.0,
        "ytick.minor.size": 3.0,
    }
    with mpl.rc_context(rcParams):
        plt.plot(inds, xs, "ro", label="data")
        plt.plot(inds, f(np.array(inds), a1, b1), "k-", label="fit")
        plt.xlabel("Iteration")
        plt.ylabel("x-position")
        plt.legend()
        plt.show()

        plt.plot(inds, ys, "ro", label="data")
        plt.plot(inds, f(np.array(inds), a2, b2), "k-", label="fit")
        plt.xlabel("Iteration")
        plt.ylabel("y-position")
        plt.legend()
        plt.show()


def prepare():
    sim = GOL((50, 50))

    path = Path("gol/lifetimes.txt")
    measurements, headers = measure_equilibration_time(sim, 2000, path=path)
    save_data(measurements, path=path, headers=headers)


def display():
    # Histogram of equilibration time
    path = Path("gol/lifetimes.txt")
    display_equilibration_time(path=path)

    # Speed of a glider state (and LWSS state)
    measure_speed_of(glider, steps=80)
    measure_speed_of(lwss, steps=80)


def main():
    if not Path("gol/").exists():
        print("running preparation script")
        prepare()
    display()


if __name__ == "__main__":
    main()
