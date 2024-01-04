import sys
from functools import partial
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from pvyzmgs.animation import animate, animate_with_cb, animate_with_plot, no_animation
from pvyzmgs.data import read_array, read_data, save_array, save_data
from pvyzmgs.plotting import display_grid
from pvyzmgs.simulation import (
    Simulation,
    progress_bar,
    record_measurements,
    run_until,
    skip_frames,
)


class RockPaperScissors(Simulation):
    """
    Deterministic rock-paper-scissors cellular automaton in 2 dimensions.
    """

    ROCK = 0
    PAPER = 1
    SCISSORS = 2

    @staticmethod
    def columns(shape):
        shortest = np.argmax(shape)
        size = shape[shortest]
        left = shape[:shortest] + (size // 3,) + shape[shortest + 1 :]
        mid = shape[:shortest] + (size // 3,) + shape[shortest + 1 :]
        right = shape[:shortest] + (size - 2 * (size // 3),) + shape[shortest + 1 :]
        return np.concatenate(
            (0 * np.ones(shape=left), 1 * np.ones(shape=mid), 2 * np.ones(shape=right)),
            axis=shortest,
        )

    @staticmethod
    def wedges(shape):
        i, j = shape

        I, J = np.meshgrid(np.arange(i), np.arange(j))
        x = I - i // 2
        y = J - j // 2
        angle = np.arctan2(y, x)

        grid = np.ones(shape)
        grid[angle < -np.pi / 3] = 0
        grid[angle > np.pi / 3] = 2
        return grid

    def randomize(self):
        """
        Randomize the contents of the simulation grid.
        """
        states = [self.ROCK, self.PAPER, self.SCISSORS]
        self.grid = np.random.choice(states, self.grid.shape)
        return self

    def neighborhood(self, state):
        """
        Return the number of cells in a specific state in the Moore neighborhood of points.
        Returns this value for every cell in the lattice.
        """
        neighbors = (
            np.roll((self.grid == state).astype(int), shift=(0, 1), axis=(0, 1))
            + np.roll((self.grid == state).astype(int), shift=(1, 1), axis=(0, 1))
            + np.roll((self.grid == state).astype(int), shift=(1, 0), axis=(0, 1))
            + np.roll((self.grid == state).astype(int), shift=(1, -1), axis=(0, 1))
            + np.roll((self.grid == state).astype(int), shift=(0, -1), axis=(0, 1))
            + np.roll((self.grid == state).astype(int), shift=(-1, -1), axis=(0, 1))
            + np.roll((self.grid == state).astype(int), shift=(-1, 0), axis=(0, 1))
            + np.roll((self.grid == state).astype(int), shift=(-1, 1), axis=(0, 1))
        )
        return neighbors

    def update(self):
        """
        Update grid according to rock-paper-scissors rules using Moore neighborhood.
        """
        # store number of living Moore neighbors in its own grid
        R = self.neighborhood(self.ROCK)
        P = self.neighborhood(self.PAPER)
        S = self.neighborhood(self.SCISSORS)

        # update if it neighbors 2 or more of the next state
        step = (
            ((self.grid == 0) & (P > 2))
            | ((self.grid == 1) & (S > 2))
            | ((self.grid == 2) & (R > 2))
        )
        self.grid += step
        self.grid %= 3

    def run(self, iterations, eq=100):
        """
        Run an rock-paper-scissors simulation.
        """
        for _ in range(eq):
            self.update()
            # note: equilibration does not increment elapsed
        return super().run(iterations)

    @property
    def rock(self):
        """
        Calculate number of total rock states.
        """
        return np.sum(self.grid == self.ROCK)

    @property
    def paper(self):
        """
        Calculate number of total paper states.
        """
        return np.sum(self.grid == self.PAPER)

    @property
    def scissors(self):
        """
        Calculate number of total scissors states.
        """
        return np.sum(self.grid == self.SCISSORS)


class RandomRockPaperScissors(RockPaperScissors):
    """
    Random sequential rock-paper-scissors cellular automaton in 2 dimensions.
    """

    def neighborhood(self, state, coords):
        """
        Return the number of cells in a specific state in the Moore neighborhood of points.
        Returns this value for every cell in the lattice.
        """
        i, j = coords
        neighbors = sum(
            (
                self.grid[i + 1, j - 1] == state,
                self.grid[i + 1, j] == state,
                self.grid[i + 1, j + 1] == state,
                self.grid[i, j + 1] == state,
                self.grid[i - 1, j + 1] == state,
                self.grid[i - 1, j] == state,
                self.grid[i - 1, j - 1] == state,
                self.grid[i, j - 1] == state,
            )
        )
        return neighbors

    def update(self):
        """
        Update grid according to rock-paper-scissors rules using Moore neighborhood.
        """
        sweep_coords = np.random.randint(0, self.grid.shape, size=(self.grid.size, 2))
        change_state = np.random.random(size=self.grid.size)
        for (i, j), r in zip(sweep_coords, change_state):
            i -= 1
            j -= 1
            state = self.grid[i, j]

            if state == self.ROCK:
                if r < self.p1 and self.neighborhood(self.PAPER, (i, j)):
                    self.grid[i, j] = self.PAPER
            if state == self.PAPER:
                if r < self.p2 and self.neighborhood(self.SCISSORS, (i, j)):
                    self.grid[i, j] = self.SCISSORS
            if state == self.SCISSORS:
                if r < self.p3 and self.neighborhood(self.ROCK, (i, j)):
                    self.grid[i, j] = self.ROCK

    def run(self, p1, p2, p3, iterations, eq=100):
        """
        Run an rock-paper-scissors simulation.
        """
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        return super().run(iterations, eq=eq)


def minority_fraction(
    p3s, sim: RandomRockPaperScissors, p1=0.5, p2=0.5, max_iter=1_000
):
    """
    Measure the average minority fraction for various values of p3.
    """
    measurements = []
    headers = (
        ", ".join(("p3", "Average Minority Fraction", "Minority Fraction Variance")),
    )

    for p3 in p3s:
        history = []
        minority = (
            lambda s: s.paper
        )  # by inspection, paper is the minority for p1 = p2 = 0.5, 0 <= p3 <= 0.1

        gen = sim.clear().randomize().run(p1, p2, p3, iterations=max_iter)
        gen = run_until(gen, lambda s: s.paper == 0)
        gen = record_measurements(gen, minority, record=history)
        gen = progress_bar(gen, desc=f"{p3=}", total=max_iter)
        no_animation(gen)

        (history,) = np.array(history).T
        pbar = np.mean(history) / sim.grid.size * sim.elapsed / max_iter
        pvar = np.var(history) / sim.grid.size * sim.elapsed / max_iter
        measurements.append((p3, pbar, pvar))
    return measurements, headers


def display_minority_fraction(path):
    p3, pbar, pvar = read_data(path, print_headers=False)

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
        fmt = "k.-"

        fig, ((ax1, ax2)) = plt.subplots(2, 1)

        ax1.plot(p3, pbar, fmt)
        ax1.set_xlabel("p3")
        ax1.set_ylabel("Average Minority Fraction")

        ax2.plot(p3, pvar, fmt)
        ax2.set_xlabel("p3")
        ax2.set_ylabel("Minority Fraction Variance")
        plt.show()


def heatmap(p2s, p3s, sim, p1=0.5, max_iter=1_000):
    """
    Measure a heatmap of the average fraction of the minority phase for varying p2 and p3.
    """
    minority_arr = []
    minority_fraction = []
    minority_fraction_var = []
    for p2 in p2s:
        minority_row = []
        minority_fraction_row = []
        minority_fraction_var_row = []
        for p3 in p3s:
            history = []
            rock = lambda s: s.rock
            paper = lambda s: s.paper
            scissors = lambda s: s.scissors

            gen = sim.clear().randomize().run(p1, p2, p3, iterations=max_iter)
            gen = run_until(gen, lambda s: s.rock == 0)
            gen = run_until(gen, lambda s: s.paper == 0)
            gen = run_until(gen, lambda s: s.scissors == 0)
            gen = record_measurements(gen, rock, paper, scissors, record=history)
            gen = progress_bar(gen, desc=f"{p2=}, {p3=}", total=max_iter)
            no_animation(gen)

            history = np.array(history).T
            minority = np.argmin(np.mean(history, axis=1))
            minority_row.append(minority)
            minority_fraction_row.append(
                np.mean(history[minority]) / sim.grid.size * sim.elapsed / max_iter
            )
            minority_fraction_var_row.append(
                np.var(history[minority]) / sim.grid.size * sim.elapsed / max_iter
            )
        minority_arr.append(minority_row)
        minority_fraction.append(minority_fraction_row)
        minority_fraction_var.append(minority_fraction_var_row)
    return (
        np.array(minority_arr),
        np.array(minority_fraction),
        np.array(minority_fraction_var),
    )


def display_heatmap(p2s, p3s, path, **kwargs):
    arr = read_array(path)
    plt.imshow(arr, origin="lower", extent=(p2s[0], p2s[-1], p3s[0], p3s[-1]), **kwargs)
    plt.xlabel("p2")
    plt.ylabel("p3")
    plt.colorbar()
    plt.show()


def prepare():
    # p3s data set
    N = 50
    sim = RandomRockPaperScissors((N, N))
    path = Path("data/p3s.txt")
    p3s = np.linspace(0, 0.1, 21)
    ms, hs = minority_fraction(p3s, sim, max_iter=5_000)
    save_data(ms, path, headers=hs)

    display_minority_fraction(path)

    # minority population data sets
    N = 50
    sim = RandomRockPaperScissors((N, N))
    p2s = np.linspace(0, 0.3, 16)
    p3s = np.linspace(0, 0.3, 16)
    minority, minfrac, minvar = heatmap(p2s, p3s, sim, max_iter=100)
    save_array(minority, Path("data/minority.txt"))
    save_array(minfrac, Path("data/minfrac.txt"))
    save_array(minvar, Path("data/minvar.txt"))

    display_heatmap(p2s, p3s, Path("data/minority.txt"), cmap="viridis")
    display_heatmap(p2s, p3s, Path("data/minfrac.txt"), cmap="gnuplot")
    display_heatmap(p2s, p3s, Path("data/minvar.txt"), cmap="inferno")


def main():
    """
    Example usage:

    python rps.py prepare
    python rps.py rps run 100
    python rps.py rps plot 100
    python rps.py rrps run 50 0.5 0.5 0.1
    python rps.py rrps plot 50 0.5 0.5 0.1 rock
    python rps.py rrps plot 50 0.5 0.5 0.1 paper
    python rps.py rrps plot 50 0.5 0.5 0.1 scissors
    python rps.py rrps minority data/p3s.txt
    python rps.py rrps heatmap data/minority.txt
    python rps.py rrps heatmap data/minfrac.txt
    python rps.py rrps heatmap data/minvar.txt
    """
    cmd = sys.argv[1]

    if cmd == "prepare":
        prepare()
    if cmd == "rps":
        if sys.argv[2] == "run":
            N = int(sys.argv[3])
            start = RockPaperScissors.wedges((N, N))
            sim = RockPaperScissors.from_array(start)
            gen = sim.run(1000, eq=0)
            animate(gen)
        if sys.argv[2] == "plot":
            N = int(sys.argv[3])
            start = RockPaperScissors.wedges((N, N))
            sim = RockPaperScissors.from_array(start)
            gen = sim.run(1000, eq=100)
            animate_with_plot(gen, lambda s: s.rock)
    if cmd == "rrps":
        if sys.argv[2] == "run":
            N = int(sys.argv[3])
            ps = map(float, sys.argv[4:])
            sim = RandomRockPaperScissors((N, N)).randomize()
            gen = sim.run(*ps, 1000, eq=0)
            animate(gen)
        if sys.argv[2] == "plot":
            N = int(sys.argv[3])
            ps = map(float, sys.argv[4:7])
            sim = RandomRockPaperScissors((N, N)).randomize()
            gen = sim.run(*ps, 1000, eq=100)
            animate_with_plot(gen, lambda s: getattr(s, sys.argv[7]))
        if sys.argv[2] == "minority":
            display_minority_fraction(Path(sys.argv[3]))
        if sys.argv[2] == "heatmap":
            display_heatmap((0, 0.3), (0, 0.3), Path(sys.argv[3]), cmap="gnuplot")


if __name__ == "__main__":
    main()
