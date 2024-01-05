import operator
from abc import abstractmethod
from itertools import accumulate
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from pvyzmgs.animation import animate, animate_with_plot, no_animation
from pvyzmgs.data import read_data, save_data
from pvyzmgs.resampling import bootstrap, jackknife
from pvyzmgs.simulation import (
    Simulation,
    progress_bar,
    record_measurements,
    skip_frames,
)


class Ising(Simulation):
    """
    Ising model simulation on an general N-dimensional lattice.
    This must be subclassed with an implementation of the `update` method for the dynamics of the Ising model.
    """

    def __init__(self, shape, *, random=True):
        """
        Initialize a lattice of spins of -1 or 1, either randomly or to the lowest energy state.
        """
        super().__init__(shape)
        if random:
            self.grid = np.random.choice((-1, 1), size=shape)
        else:
            self.grid = self.low_energy(shape)

    @staticmethod
    @abstractmethod
    def low_energy(shape):
        pass

    def run(self, temperature, steps, eq=100):
        """
        Run an Ising model simulation at a specific temperature after equilibration.
        """
        self.T = temperature
        for _ in range(eq):
            self.update()
            self.elapsed += 1
        return super().run(steps)

    def neighborhood(self, coords):
        """
        Calculate the sum of spins in the von Neumann neighborhood of a lattice point.
        """
        neighbors = 0
        for i, coord in enumerate(coords):
            neighbors += self.grid[*coords[:i], coord + 1, *coords[i + 1 :]]
            neighbors += self.grid[*coords[:i], coord - 1, *coords[i + 1 :]]
        return neighbors

    @property
    def energy(self):
        """
        Measure the energy of the lattice.
        """
        return -np.sum(
            self.grid
            * sum(np.roll(self.grid, shift=1, axis=i) for i in range(self.dim))
        )

    @property
    def magnetization(self):
        """
        Measure the magnetization of the lattice.
        """
        return np.sum(self.grid)


class Glauber(Ising):
    """
    Ising model simulation using Glauber dynamics on an general N-dimensional lattice.
    """

    @staticmethod
    def low_energy(shape):
        """
        Return lattice with all spins aligned.
        """
        return np.ones(
            shape=shape
        )  # in Glauber dynamics, all-aligned spins is the lowest energy state

    def update(self):
        """
        Flip lattice spins according to Boltzmann probability. Each update runs 1 full sweep.
        """
        sweep_coords = np.random.randint(
            0, self.grid.shape, size=(self.grid.size, len(self.grid.shape))
        )
        flips = np.random.random(size=self.grid.size)
        for coords, flip in zip(sweep_coords, flips):
            coords = [coord - 1 for coord in coords]
            spin = self.grid[*coords]

            # calculate change in energy from von Neumann neighborhood
            dE = 2 * spin * self.neighborhood(coords)

            # flip spin according to Boltzmann probability
            # note: predicate is True when dE <= 0
            if flip <= np.e ** (-dE / self.T):
                self.grid[*coords] = -spin


class Kawasaki(Ising):
    """
    Ising model simulation using Kawasaki dynamics on an general N-dimensional lattice.
    """

    @staticmethod
    def low_energy(shape):
        """
        Return lattice with half-up and half-down spins.
        """
        # in Kawasaki dynamics, half-up and half-down spins is the lowest energy state
        # for generality, this split is done across the longest axis to minimize total energy
        shortest = np.argmax(shape)
        size = shape[shortest]
        left = shape[:shortest] + (size // 2,) + shape[shortest + 1 :]
        right = shape[:shortest] + (size - size // 2,) + shape[shortest + 1 :]
        return np.concatenate(
            (np.ones(shape=left), -np.ones(shape=right)), axis=shortest
        )

    def update(self):
        """
        Swap lattice spins according to Boltzmann probability. Each update runs 1 full sweep.
        """
        size = self.grid.size
        decomposition_factors = list(
            accumulate((1,) + self.grid.shape[::-1], operator.mul)
        )[-2::-1]

        sweep_pairs = np.random.randint(0, size * (size - 1), size)
        flips = np.random.random(size=self.grid.size)
        for pair, flip in zip(sweep_pairs, flips):
            # convert "pair" into two lattice points
            pair += pair // size + 1
            p1, p2 = divmod(pair, size)
            coords1 = []
            coords2 = []
            for factor in decomposition_factors:
                coord, p1 = divmod(p1, factor)
                coords1.append(coord - 1)
                coord, p2 = divmod(p2, factor)
                coords2.append(coord - 1)
            spin1 = self.grid[*coords1]
            spin2 = self.grid[*coords2]

            if spin1 == spin2:  # same spin, swapping has no effect
                continue

            # calculate change in energy from von Neumann neighborhoods
            # spins are opposite, so the difference is taken
            dE = 2 * spin1 * (self.neighborhood(coords1) - self.neighborhood(coords2))
            dist = sum(
                min(abs(c2 - c1), axis - abs(c2 - c1))
                for axis, c1, c2 in zip(self.grid.shape, coords1, coords2)
            )
            if (
                dist == 1
            ):  # correct change in energy if points neighbor, checked by periodic taxicab distance
                dE += 4

            # flip spin according to Boltzmann probability
            # note: predicate is True when dE <= 0
            if flip <= np.e ** (-dE / self.T):
                self.grid[*coords1] = spin2
                self.grid[*coords2] = spin1


class Ising2D(Ising):
    """
    Ising model simulation on an 2-dimensional lattice (for optimization purposes).
    This must be subclassed with an implementation of the `update` method for the dynamics of the Ising model.
    """

    def neighborhood(self, coords):
        """
        Calculate the sum of spins in the von Neumann neighborhood of a lattice point.
        """
        i, j = coords
        return (
            self.grid[i - 1, j]
            + self.grid[i + 1, j]
            + self.grid[i, j - 1]
            + self.grid[i, j + 1]
        )

    @property
    def energy(self):
        """
        Measure the energy of the lattice.
        """
        return -np.sum(
            self.grid
            * (
                np.roll(self.grid, shift=1, axis=0)
                + np.roll(self.grid, shift=1, axis=1)
            )
        )


class Glauber2D(Ising2D, Glauber):
    """
    Ising model simulation using Glauber dynamics on a 2-dimensional lattice (for optimization purposes).
    """

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
            dE = 2 * spin * self.neighborhood((i, j))

            # flip spin according to Boltzmann probability
            # note: predicate is True when dE <= 0
            if flip <= np.e ** (-dE / self.T):
                self.grid[i, j] = -spin


class Kawasaki2D(Ising2D, Kawasaki):
    """
    Ising model simulation using Kawasaki dynamics on a 2-dimensional lattice (for optimization purposes).
    """

    def update(self):
        """
        Swap lattice spins according to Boltzmann probability. Each update runs 1 full sweep.
        """
        size = self.grid.size
        ax1, ax2 = self.grid.shape

        sweep_pairs = np.random.randint(0, size * (size - 1), size)
        flips = np.random.random(size=self.grid.size)
        for pair, flip in zip(sweep_pairs, flips):
            # convert "pair" into two lattice points
            pair += pair // size + 1
            p1, p2 = divmod(pair, size)
            i1, j1 = divmod(p1, ax2)
            i2, j2 = divmod(p2, ax2)
            i1 -= 1
            j1 -= 1
            i2 -= 1
            j2 -= 1
            spin1 = self.grid[i1, j1]
            spin2 = self.grid[i2, j2]

            if spin1 == spin2:  # same spin, swapping has no effect
                continue

            # calculate change in energy from von Neumann neighborhoods
            # spins are opposite, so the difference is taken
            dE = 2 * spin1 * (self.neighborhood((i1, j1)) - self.neighborhood((i2, j2)))
            dist = min(abs(i2 - i1), ax1 - abs(i2 - i1)) + min(
                abs(j2 - j1), ax2 - abs(j2 - j1)
            )
            if (
                dist == 1
            ):  # correct change in energy if points neighbor, checked by periodic taxicab distance
                dE += 4

            # flip spin according to boltzmann probability
            # note: predicate is True when dE <= 0
            if flip <= np.e ** (-dE / self.T):
                self.grid[i1, j1] = spin2
                self.grid[i2, j2] = spin1


def run_experiment_at(T, sim, sweeps=10_000, skip=10, eq=100):
    """
    Records the energy and magnetization in an Ising model simulation at a specific temperature over time.
    Return a list of (iteration, energy, magnetization) tuples.
    """
    gen = sim.run(T, steps=sweeps, eq=eq)
    gen = progress_bar(gen, desc=f"{T=}", total=sweeps)
    gen = skip_frames(gen, n=skip)

    measurements = []
    elapsed = lambda s: s.elapsed
    energy = lambda s: s.energy
    magnetization = lambda s: s.magnetization
    gen = record_measurements(gen, elapsed, energy, magnetization, record=measurements)
    no_animation(gen)

    return measurements
    measurements = np.array(measurements)
    elapsed, energy, magnetization = measurements.T
    plt.plot(elapsed, energy)
    plt.show()
    plt.plot(elapsed, magnetization)
    plt.show()


def run_experiment_over(Ts, sim, sweeps=10_000, skip=10):
    """
    Runs a simulation of the Ising model over a range of temperatures and measures the following quantities and their uncertainties:
    - average energy
    - average (absolute) magnetization
    - heat capacity
    - susceptibility
    The heat capacity & susceptibility uncertainties are recorded using both bootstrap and jackknife resampling.
    """
    eq = 100

    measurements = []
    headers = (
        ", ".join(
            (
                "Temperature",
                "Average Energy",
                "Average Energy Error",
                "Average (Absolute) Magnetization",
                "Average (Absolute) Magnetization Error",
                "Heat Capacity",
                "Heat Capacity Error (Bootstrap Resampling)",
                "Heat Capacity Error (Jackknife Resampling)",
                "Susceptibility",
                "Susceptibility Error (Boostrap Resampling)",
                "Susceptibility Error (Jackknife Resampling)",
            )
        ),
    )

    for T in Ts:
        trial = run_experiment_at(T, sim, sweeps=sweeps, skip=skip, eq=eq)
        t, E, M = np.array(trial).T
        eq = 0  # no equilibration needed after first temperature

        ebar = np.mean(E)
        d_ebar = np.std(E)
        mbar = np.mean(
            np.abs(M)
        )  # abs taken to account for lattices with flipped spins
        d_mbar = np.std(
            np.abs(M)
        )  # abs taken to account for lattices with flipped spins

        heat_capacity = lambda x: np.var(x) / (sim.grid.size * T**2)
        hc = heat_capacity(E)
        d_hc_b = bootstrap(heat_capacity, E)
        d_hc_j = jackknife(heat_capacity, E)

        susceptibility = lambda x: np.var(x) / (sim.grid.size * T)
        sus = susceptibility(M)
        d_sus_b = bootstrap(susceptibility, M)
        d_sus_j = jackknife(susceptibility, M)

        measurements.append(
            (T, ebar, d_ebar, mbar, d_mbar, hc, d_hc_b, d_hc_j, sus, d_sus_b, d_sus_j)
        )
    return measurements, headers


def display_plots(path, errors="bootstrap"):
    (
        T,
        ebar,
        d_ebar,
        mbar,
        d_mbar,
        hc,
        d_hc_b,
        d_hc_j,
        sus,
        d_sus_b,
        d_sus_j,
    ) = read_data(path)

    rcParams = {
        "figure.autolayout": True,
        "axes.grid": True,
        "axes.linewidth": 0.8,
        "errorbar.capsize": 3,
        "font.size": 8,
        "grid.color": "silver",
        "legend.fontsize": 10,
        "legend.handlelength": 2.0,
        "lines.linewidth": 0.5,
        "xtick.direction": "in",
        "xtick.major.size": 5.0,
        "xtick.minor.size": 3.0,
        "ytick.direction": "in",
        "ytick.major.size": 5.0,
        "ytick.minor.size": 3.0,
    }
    with mpl.rc_context(rcParams):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        fmt = "k.-"

        ax1.errorbar(T, ebar, yerr=d_ebar, fmt=fmt)
        ax1.set_xlabel("Temperature")
        ax1.set_ylabel("Energy")
        ax1.set_title("Energy vs Temperature")

        ax2.errorbar(T, mbar, yerr=d_mbar, fmt=fmt)
        ax2.set_xlabel("Temperature")
        ax2.set_ylabel("|Magnetization|")
        ax2.set_title("|Magnetization| vs Temperature")

        if errors == "bootstrap":
            ax3.errorbar(T, hc, yerr=d_hc_b, fmt=fmt)
            ax4.errorbar(T, sus, yerr=d_sus_b, fmt=fmt)
        elif errors == "jackknife":
            ax3.errorbar(T, hc, yerr=d_hc_j, fmt=fmt)
            ax4.errorbar(T, sus, yerr=d_sus_j, fmt=fmt)
        else:
            raise ValueError('Errors must be "bootstrap" or "jackknife".')
        ax3.set_xlabel("Temperature")
        ax3.set_ylabel("Heat Capacity")
        ax3.set_title("Heat Capacity vs Temperature")
        ax4.set_xlabel("Temperature")
        ax4.set_ylabel("Susceptibility")
        ax4.set_title("Susceptibility vs Temperature")
    plt.show()


def prepare():
    # Glauber
    path = Path("ising/glauber_1k.txt")
    N = 50
    sweeps = 1000
    sim = Glauber2D((N, N), random=False)

    measurements, headers = run_experiment_over(
        np.arange(1, 3.05, 0.1), sim, sweeps=sweeps
    )
    save_data(measurements, path=path, headers=headers)

    # Kawasaki
    path = Path("ising/kawasaki_1k.txt")
    N = 50
    sweeps = 1000
    sim = Kawasaki2D((N, N), random=False)

    measurements, headers = run_experiment_over(
        np.arange(1, 3.05, 0.1), sim, sweeps=sweeps
    )
    save_data(measurements, path=path, headers=headers)


def display():
    # Basic functionality of the code
    N = 50
    gsim = Glauber2D((N, N), random=False)
    ksim = Kawasaki2D((N, N), random=False)

    # Glauber at low temp
    gen = gsim.run(1, steps=1000, eq=0)
    animate(gen)
    # Glauber at high temp
    gen = gsim.run(3, steps=1000, eq=0)
    animate(gen)
    # Kawasaki at low temp
    gen = ksim.run(1, steps=1000, eq=0)
    animate(gen)
    # Kawasaki at high temp
    gen = ksim.run(3, steps=1000, eq=0)
    animate(gen)

    # Quantitative analysis and plots
    display_plots(Path("ising/glauber_1k.txt"), errors="bootstrap")
    display_plots(Path("ising/glauber_1k.txt"), errors="jackknife")
    display_plots(Path("ising/kawasaki_1k.txt"), errors="bootstrap")
    display_plots(Path("ising/kawasaki_1k.txt"), errors="jackknife")


def test():
    # Glauber
    path = Path("test/glauber.txt")
    N = 20
    sweeps = 100
    sim = Glauber2D((N, N), random=False)

    measurements, headers = run_experiment_over(
        np.arange(1, 3.05, 0.2), sim, sweeps=sweeps
    )
    save_data(measurements, path=path, headers=headers, overwrite=True)

    # Kawasaki
    path = Path("test/kawasaki.txt")
    N = 20
    sweeps = 100
    sim = Kawasaki2D((N, N), random=False)

    measurements, headers = run_experiment_over(
        np.arange(1, 3.05, 0.2), sim, sweeps=sweeps
    )
    save_data(measurements, path=path, headers=headers, overwrite=True)

    # Basic functionality of the code
    N = 20
    gsim = Glauber2D((N, N), random=False)
    ksim = Kawasaki2D((N, N), random=False)

    # Glauber at low temp
    gen = gsim.run(1, steps=1000, eq=0)
    animate(progress_bar(gen, total=1000, desc="glauber low"))
    # Glauber at high temp
    gen = gsim.run(3, steps=1000, eq=0)
    animate(progress_bar(gen, total=1000, desc="glauber high"))
    # Kawasaki at low temp
    gen = ksim.run(1, steps=1000, eq=0)
    animate(progress_bar(gen, total=1000, desc="kawasaki low"))
    # Kawasaki at high temp
    gen = ksim.run(3, steps=1000, eq=0)
    animate(progress_bar(gen, total=1000, desc="kawasaki high"))

    # Quantitative analysis and plots
    display_plots(Path("test/glauber.txt"), errors="bootstrap")
    display_plots(Path("test/glauber.txt"), errors="jackknife")
    display_plots(Path("test/kawasaki.txt"), errors="bootstrap")
    display_plots(Path("test/kawasaki.txt"), errors="jackknife")


def main():
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        plt.ion()
        print("running test script")
        test()
        return

    if not Path("ising/").exists():
        print("running preparation script")
        prepare()
    display()


if __name__ == "__main__":
    main()
