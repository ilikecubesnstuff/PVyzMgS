from pathlib import Path

import matplotlib as mpl
import numpy as np
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


class ContactProcess(Simulation):
    """
    Contact process model cellular automaton in 2 dimensions.
    """

    INACTIVE = 0
    ACTIVE = 1

    def randomize(self):
        """
        Randomize the contents of the simulation grid.
        """
        states = [self.ACTIVE, self.INACTIVE]
        self.grid = np.random.choice(states, self.grid.shape)
        return self

    def update(self):
        """
        Update cells according to contact process model.
        """
        sweep_coords = np.random.randint(0, self.grid.shape, size=(self.grid.size, 2))
        change_state = np.random.random(size=(self.grid.size, 2))
        for (i, j), (r1, r2) in zip(sweep_coords, change_state):
            i -= 1
            j -= 1
            state = self.grid[i, j]

            if not state:  # if the cell is inactive, nothing happens
                continue

            if r1 < 1 - self.p:  # an active cell becomes inactive with probability 1-p
                self.grid[i, j] = self.INACTIVE

            if (
                r2 < self.p
            ):  # an active cell infects a random neighbor with probability p
                # pick a random neighbor to activate
                indices = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
                self.grid[*indices[np.random.randint(4)]] = self.ACTIVE

    def run(self, p, *, steps=10_000, eq=0):
        """
        Run an contact process simulation.
        """
        self.p = p

        for _ in range(eq):
            self.update()
            # note: equilibration does not increment elapsed
        return super().run(steps)

    @property
    def active(self):
        """
        Get number of active cells.
        """
        return np.sum(self.grid)


def record_infected_fraction_against_p(ps, sim, max_iter=10_000):
    """
    Record the average infected fraction against p.
    """
    measurements = []
    headers = (
        ", ".join(
            (
                "p",
                "Infected Fraction",
                "Infected Fraction Error",
                "Infected Fraction Variance",
                "Infected Fraction Variance (Bootstrap Resampling)",
                "Infected Fraction Variance (Jackknife Resampling)",
            )
        ),
    )

    for p in ps:
        # track history of infected population
        history = []
        infected = lambda s: s.active

        # run simulation
        gen = sim.clear().randomize().run(p, steps=max_iter, eq=100)
        gen = run_until(gen, lambda s: s.active == 0)
        gen = record_measurements(gen, infected, record=history)
        gen = progress_bar(gen, desc=f"{p}", total=max_iter)
        no_animation(gen)

        # record values
        history = np.array(history).T[0]

        inffrac = np.mean(history) / sim.grid.size * sim.elapsed / max_iter
        d_inffrac = np.std(history) / sim.grid.size * sim.elapsed / max_iter

        infected_variance = lambda x: np.var(x) / sim.grid.size * sim.elapsed / max_iter
        infvar = infected_variance(history)
        d_infvar_b = bootstrap(infected_variance, history)
        d_infvar_j = jackknife(infected_variance, history)

        measurements.append((p, inffrac, d_inffrac, infvar, d_infvar_b, d_infvar_j))

    return measurements, headers


def display_infected_fraction_against_p(path, errors="bootstrap"):
    """
    Display infected fraction against p.
    """
    p1, inffrac, d_inffrac, infvar, d_infvar_b, d_infvar_j = read_data(path)

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
        fmt = "k.-"

        fig, ax = plt.subplots()
        ax.errorbar(p1, inffrac, yerr=d_inffrac, fmt=fmt)
        ax.set_xlabel("p")
        ax.set_ylabel("Infected fraction")
        plt.show()

        fig, ax = plt.subplots()
        if errors == "bootstrap":
            ax.errorbar(p1, infvar, yerr=d_infvar_b, fmt=fmt)
        elif errors == "jackknife":
            ax.errorbar(p1, infvar, yerr=d_infvar_j, fmt=fmt)
        else:
            raise ValueError('Errors must be "bootstrap" or "jackknife".')
        ax.set_xlabel("p")
        ax.set_ylabel("Infected fraction variance")
        plt.show()


def survival_probability(p, sim, trials=100, its=300):
    """
    Compute the probability of there existing active cells over time.
    """
    active_time_series = np.zeros(its + 1)
    for trial in tqdm(range(trials)):
        history = [(1.0,)]
        active = lambda s: s.active

        sim.clear()
        sim.grid[0, 0] = 1
        gen = sim.run(p, steps=its)
        # gen = progress_bar(gen, desc=f'{trial=}', total=its)
        gen = record_measurements(gen, active, record=history)
        no_animation(gen)

        history = np.array(history).T[0]
        active_time_series += history > 0

    measurements = list(enumerate(active_time_series / trials))
    headers = (", ".join(("Time (Sweeps)", "Survival Probability")),)
    return measurements, headers


def display_survival_probability(path, ax=None):
    """
    Display a log-log plot of survival probability.
    """
    t, sp = read_data(path)

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

        if not ax:
            fig, ax = plt.subplots()
        ax.plot(np.log(t), np.log(sp), label=path.stem)
        ax.set_xlabel("log(Time)")
        ax.set_ylabel("log(Survival Probability)")
        if not ax:
            plt.show()


def prepare():
    sim = ContactProcess((50, 50)).randomize()
    # gen = sim.run(0.5)
    # animate(gen)

    # active = lambda s: s.active
    # title = lambda p: (lambda s: f'{p=}, {s.elapsed=}')
    # # p = 0.6
    # gen = sim.clear().randomize().run(0.6, steps=1000, eq=100)
    # gen = skip_frames(gen, 10)
    # animate_with_plot(gen, active, title(0.6))
    # # p = 0.7
    # gen = sim.clear().randomize().run(0.7, steps=1000, eq=100)
    # gen = skip_frames(gen, 10)
    # animate_with_plot(gen, active, title(0.7))

    path = Path("data/ps.txt")
    ms, hs = record_infected_fraction_against_p(
        np.arange(0.55, 0.701, 0.005), sim, max_iter=1000
    )
    save_data(measurements=ms, path=path, headers=hs)

    for p in (0.6, 0.625, 0.65):
        path = Path(f"data/sp{p}.txt")
        ms, hs = survival_probability(p, sim)
        save_data(measurements=ms, path=path, headers=hs)


def display():
    sim = ContactProcess((50, 50)).randomize()
    gen = sim.run(0.5)
    animate(gen)

    path = Path("data/ps.txt")
    display_infected_fraction_against_p(path)

    fig, ax = plt.subplots()
    display_survival_probability(Path(f"data/sp0.6.txt"), ax=ax)
    display_survival_probability(Path(f"data/sp0.625.txt"), ax=ax)
    display_survival_probability(Path(f"data/sp0.65.txt"), ax=ax)
    ax.legend()
    plt.show()


def main():
    if not Path("data/").exists():
        print("running preparation script")
        prepare()
    display()


if __name__ == "__main__":
    main()
