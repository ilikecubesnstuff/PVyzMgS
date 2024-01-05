from pathlib import Path

import matplotlib as mpl
import numpy as np
from ising import Glauber2D
from matplotlib import pyplot as plt
from tqdm import tqdm

from pvyzmgs.animation import animate, animate_with_cb, animate_with_plot, no_animation
from pvyzmgs.data import read_array, read_data, save_array, save_data
from pvyzmgs.resampling import bootstrap, jackknife
from pvyzmgs.simulation import (
    Simulation,
    progress_bar,
    record_measurements,
    run_until,
    skip_frames,
)


class UseRho(Simulation):
    def __init__(self, rho):
        """
        Create a simulation grid off an external grid variable rho.
        """
        self.rho = rho
        super().__init__(rho.shape)


class Antiferromagnet(Glauber2D):
    """
    Antiferromagnet Ising model simulation on an general N-dimensional lattice. (J = -1)
    This must be subclassed with an implementation of the `update` method for the dynamics of the Ising model.
    """

    def __init__(self, shape, *, random=True):
        """
        Initialize a lattice of spins of -1 or 1, either randomly or to the lowest energy state.
        """
        super().__init__(shape)
        self.evens = (
            np.sum(np.meshgrid(*[np.arange(i) for i in shape]), axis=0) % 2 == 0
        )
        self.odds = np.sum(np.meshgrid(*[np.arange(i) for i in shape]), axis=0) % 2 == 1
        if random:
            self.grid = np.random.choice((-1, 1), size=shape)
        else:
            self.grid = self.low_energy(shape)

    @staticmethod
    def low_energy(shape):
        """
        Return lattice with spins aligned and misaligned at even and odd indices.
        """
        odds = np.sum(np.meshgrid(*[np.arange(i) for i in shape]), axis=0) % 2 == 1
        lattice = np.ones(shape)
        lattice[odds] *= -1
        return lattice

    def update(self):
        """
        Flip lattice spins according to Boltzmann probability. Each update runs 1 full sweep.
        """
        sweep_coords = np.random.randint(0, self.grid.shape, size=(self.grid.size, 2))
        flips = np.random.random(size=self.grid.size)
        for (i, j), flip in zip(sweep_coords, flips):
            i -= 1
            j -= 1
            spin = self.grid[i, j]

            # calculate change in energy from von Neumann neighborhood
            dE = 2 * spin * (self.h - self.neighborhood((i, j)))

            # flip spin according to Boltzmann probability
            # note: predicate is True when dE <= 0
            if flip <= np.e ** (-dE / self.T):
                self.grid[i, j] = -spin

    def run(self, h, T=1, steps=10_000, eq=100):
        """
        Run an Ising model simulation at a specific temperature after equilibration.
        """
        self.h = h
        self.T = T
        for _ in range(eq):
            self.update()
            self.elapsed += 1
        return super().run(T, steps, eq)

    @property
    def energy(self):
        """
        Measure the energy of the lattice.
        """
        return np.sum(
            self.grid
            * (
                np.roll(self.grid, shift=1, axis=0)
                + np.roll(self.grid, shift=1, axis=1)
            )
        ) + self.h * np.sum(self.grid)

    @property
    def magnetization(self):
        """
        Measure the magnetization of the lattice.
        """
        return np.sum(self.grid)

    @property
    def staggered_magnetization(self):
        """
        Measure the staggered magnetization of the lattice.
        """
        return np.sum(self.grid[self.evens]) - np.sum(self.grid[self.odds])


def run_experiment_over(hs, sim: Antiferromagnet, max_iter=1_000):
    """
    Measure the average energy, average and variance of magnetization, and
    average and variance of staggered magnetization over a range of h.
    """
    measurements = []
    headers = (
        ", ".join(
            (
                "h",
                "Average Energy",
                "Average Magnetization",
                "Variance of Magnetization",
                "Average Staggered Magnetization",
                "Variance of Staggered Magnetization",
            )
        ),
    )

    for h in hs:
        # track variables
        history = []
        energy = lambda s: s.energy
        magnetization = lambda s: s.magnetization
        staggered_magnetization = lambda s: s.staggered_magnetization

        # run simulation
        gen = sim.run(h, steps=max_iter)
        gen = record_measurements(
            gen, energy, magnetization, staggered_magnetization, record=history
        )
        gen = progress_bar(gen, desc=f"{h=}", total=max_iter)
        no_animation(gen)

        # measure values
        E, M, S = np.array(history).T
        ebar = np.mean(E)
        mbar = np.mean(M)
        mvar = np.var(M)
        sbar = np.mean(S)
        svar = np.var(S)
        measurements.append((h, ebar, mbar, mvar, sbar, svar))

    return measurements, headers


def display_experiment_data(path):
    h, ebar, mbar, mvar, sbar, svar = read_data(path)

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
        fmt = "k-"

        plots = {
            "Average Energy": ebar,
            "Average Magnetization": mbar,
            "Variance of Magnetization": mvar,
            "Average Staggered Magnetization": sbar,
            "Variance of Staggered Magnetization": svar,
        }

        for name, var in plots.items():
            fig, ax = plt.subplots()
            ax.plot(h, var, fmt)
            ax.set_xlabel("h")
            ax.set_ylabel(name)
            plt.show()


class AntiferromagnetWithH(Antiferromagnet):
    def update(self):
        """
        Flip lattice spins according to Boltzmann probability. Each update runs 1 full sweep.
        """
        self.h = self.h_func(self.elapsed)
        sweep_coords = np.random.randint(0, self.grid.shape, size=(self.grid.size, 2))
        flips = np.random.random(size=self.grid.size)
        for (i, j), flip in zip(sweep_coords, flips):
            i -= 1
            j -= 1
            spin = self.grid[i, j]

            # calculate change in energy from von Neumann neighborhood
            dE = 2 * spin * (self.h[i, j] - self.neighborhood((i, j)))

            # flip spin according to Boltzmann probability
            # note: predicate is True when dE <= 0
            if flip <= np.e ** (-dE / self.T):
                self.grid[i, j] = -spin

    def run(self, h_func, T=1, steps=10_000, eq=100):
        """
        Run an Ising model simulation at a specific temperature after equilibration.
        """
        self.h_func = h_func
        self.T = T
        for _ in range(eq):
            self.update()
            self.elapsed += 1
        return super().run(None, T, steps, eq)


def functional_h(shape):
    axes = [np.arange(n) for n in shape]
    x, y = np.meshgrid(*axes)

    def h_factory(h0, P, tau):
        def h_func(t):
            return (
                h0
                * np.cos(2 * np.pi * x / P)
                * np.cos(2 * np.pi * y / P)
                * np.sin(2 * np.pi * t / tau)
            )

        return h_func

    return h_factory


def run_functional_h(shape, h_func):
    sim = AntiferromagnetWithH(shape, random=False)
    gen = sim.run(h_func, steps=2500, eq=0)
    gen = progress_bar(gen, total=2500)  # sin = +1
    gen = skip_frames(gen, 100)
    animate(gen)
    gen = sim.run(h_func, steps=2500, eq=0)
    gen = progress_bar(gen, total=2500)  # sin = 0
    gen = skip_frames(gen, 100)
    animate(gen)
    gen = sim.run(h_func, steps=2500, eq=0)
    gen = progress_bar(gen, total=2500)  # sin = -1
    gen = skip_frames(gen, 100)
    animate(gen)


def measure_field_vs_sm(shape, h_func, max_iter=20_000):
    measurements = []
    headers = (
        ", ".join(("Time", "Maximum Field Strength", "Staggered Magnetization")),
    )
    time = lambda s: s.elapsed
    field = lambda s: np.max(s.h)
    sm = lambda s: s.staggered_magnetization

    # run simulation
    sim = AntiferromagnetWithH(shape, random=False)
    gen = sim.run(h_func, steps=max_iter, eq=0)
    gen = record_measurements(gen, time, field, sm, record=measurements)
    gen = progress_bar(gen, total=max_iter)
    no_animation(gen)

    return measurements, headers


def display_field_vs_sm(path):
    """
    Display a slice of the phase diagram across a fixed p2 & p3 line.
    """
    t, field, sm = read_data(path)

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
        fmt = "k-"

        fig, ((ax1, ax2)) = plt.subplots(2, 1)

        ax1.plot(t, field)
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Maximum Field Strength")

        ax2.plot(t, sm)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Staggered Magnetization")
        plt.show()


def prepare():
    L = 50
    sim = Antiferromagnet((L, L), random=False)
    gen = sim.run(5, eq=0)
    animate(gen)

    hs = np.linspace(0, 10, 21)
    ms, hs = run_experiment_over(hs, sim)
    save_data(ms, Path("antiferromagnet/hs10.txt"), headers=hs)

    h_func = functional_h((L, L))(h0=10, P=25, tau=10_000)
    ms, hs = measure_field_vs_sm((L, L), h_func)
    save_data(ms, path=Path("antiferromagnet/p25.txt"), headers=hs)

    h_func = functional_h((L, L))(h0=10, P=10, tau=10_000)
    ms, hs = measure_field_vs_sm((L, L), h_func)
    save_data(ms, path=Path("antiferromagnet/p10.txt"), headers=hs)


def display():
    display_experiment_data(Path("antiferromagnet/hs10.txt"))
    display_field_vs_sm(Path("antiferromagnet/p25.txt"))


def test():
    L = 50
    sim = Antiferromagnet((L, L), random=False)
    gen = sim.run(5, steps=200, eq=0)
    no_animation(progress_bar(gen, total=200))

    hs = np.linspace(0, 10, 11)
    ms, hs = run_experiment_over(hs, sim, max_iter=100)
    save_data(ms, Path("test/hs10.txt"), headers=hs, overwrite=True)

    h_func = functional_h((L, L))(h0=10, P=10, tau=1000)
    ms, hs = measure_field_vs_sm((L, L), h_func, max_iter=1000)
    save_data(ms, path=Path("test/p10.txt"), headers=hs, overwrite=True)

    display_experiment_data(Path("test/hs10.txt"))
    display_field_vs_sm(Path("test/p10.txt"))


def main():
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        plt.ion()
        print("running test script")
        test()
        return

    if not Path("antiferromagnet/").exists():
        print("running preparation script")
        prepare()
    display()


if __name__ == "__main__":
    main()
