from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from pvyzmgs.animation import animate, animate_with_plot, no_animation
from pvyzmgs.data import read_array, read_data, save_array, save_data
from pvyzmgs.plotting import display_grid
from pvyzmgs.resampling import bootstrap, jackknife
from pvyzmgs.simulation import Simulation, progress_bar, record_measurements, run_until


class SIRS(Simulation):
    """
    SIRS model cellular automaton in 2 dimensions.
    """

    SUSCEPTIBLE = 0
    INFECTED = 1
    RECOVERED = 2

    def randomize(self):
        """
        Randomize the contents of the simulation grid.
        """
        states = [self.SUSCEPTIBLE, self.INFECTED, self.RECOVERED]
        self.grid = np.random.choice(states, self.grid.shape)
        return self

    def update(self):
        """
        Update cells according to SIRS model rules.
        """
        sweep_coords = np.random.randint(0, self.grid.shape, size=(self.grid.size, 2))
        change_state = np.random.random(size=self.grid.size)
        for (i, j), r in zip(sweep_coords, change_state):
            i -= 1
            j -= 1
            state = self.grid[i, j]

            if state == self.SUSCEPTIBLE:
                # check if von Neumann neighborhood contains an infected cell
                if 1 in (
                    self.grid[i - 1, j],
                    self.grid[i + 1, j],
                    self.grid[i, j - 1],
                    self.grid[i, j + 1],
                ):
                    if r < self.p1:
                        self.grid[i, j] = self.INFECTED
            elif state == self.INFECTED:
                if r < self.p2:
                    self.grid[i, j] = self.RECOVERED
            elif state == self.RECOVERED:
                if r < self.p3:
                    self.grid[i, j] = self.SUSCEPTIBLE

    def run(self, p1, p2, p3, *, steps=10_000, eq=0):
        """
        Run an SIRS simulation.
        """
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

        for _ in range(eq):
            self.update()
            # note: equilibration does not increment elapsed
        return super().run(steps)

    @property
    def susceptible(self):
        """
        Get total number of susceptible cells.
        """
        return np.sum(self.grid == self.SUSCEPTIBLE)

    @property
    def infected(self):
        """
        Get total number of infected cells.
        """
        return np.sum(self.grid == self.INFECTED)

    @property
    def recovered(self):
        """
        Get total number of recovered cells.
        """
        return np.sum(self.grid == self.RECOVERED)


def record_phase_diagram(p1s, p3s, sim, p2=0.5, max_iter=1_000):
    """
    Record a phase diagram across a fixed p2 plane.
    """
    infected_fraction = []
    infected_variance = []
    for p1 in p1s:
        infected_fraction_row = []
        infected_variance_row = []
        for p3 in p3s:
            # track history of infected population
            history = []
            infected = lambda s: s.infected

            # run simulation
            gen = sim.clear().randomize().run(p1, p2, p3, steps=max_iter, eq=100)
            gen = run_until(gen, lambda s: s.infected == 0)
            gen = record_measurements(gen, infected, record=history)
            gen = progress_bar(gen, desc=f"{(p1, p2, p3)}", total=max_iter)
            no_animation(gen)

            # record infected fraction quantities
            history = np.array(history).T[0]
            q = np.mean(history) / sim.grid.size * sim.elapsed / max_iter
            infected_fraction_row.append(q)
            q = np.var(history) / sim.grid.size * sim.elapsed / max_iter
            infected_variance_row.append(q)
        infected_fraction.append(infected_fraction_row)
        infected_variance.append(infected_variance_row)
    return np.array(infected_fraction), np.array(infected_variance)


def record_phase_diagram_slice(p1s, sim, p2=0.5, p3=0.5, max_iter=10_000):
    """
    Record a slice of a phase diagram across a fixed p2 & p3 line.
    """
    measurements = []
    headers = (
        ", ".join(
            (
                "p1",
                "Infected Fraction",
                "Infected Fraction Error",
                "Infected Fraction Variance",
                "Infected Fraction Variance (Bootstrap Resampling)",
                "Infected Fraction Variance (Jackknife Resampling)",
            )
        ),
    )
    for p1 in p1s:
        # track history of infected population
        history = []
        infected = lambda s: s.infected

        # run simulation
        gen = sim.clear().randomize().run(p1, p2, p3, steps=max_iter, eq=100)
        gen = run_until(gen, lambda s: s.infected == 0)
        gen = record_measurements(gen, infected, record=history)
        gen = progress_bar(gen, desc=f"{(p1, p2, p3)}", total=max_iter)
        no_animation(gen)

        # record quantities
        history = np.array(history).T[0]

        inffrac = np.mean(history) / sim.grid.size * sim.elapsed / max_iter
        d_inffrac = np.std(history) / sim.grid.size * sim.elapsed / max_iter

        infected_variance = lambda x: np.var(x) / sim.grid.size * sim.elapsed / max_iter
        infvar = infected_variance(history)
        d_infvar_b = bootstrap(infected_variance, history)
        d_infvar_j = jackknife(infected_variance, history)

        measurements.append((p1, inffrac, d_inffrac, infvar, d_infvar_b, d_infvar_j))
    return measurements, headers


def display_phase_diagram_slice(path, errors="bootstrap"):
    """
    Display a slice of the phase diagram across a fixed p2 & p3 line.
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
        ax.set_xlabel("p1")
        ax.set_ylabel("Infected fraction")
        plt.show()

        fig, ax = plt.subplots()
        if errors == "bootstrap":
            ax.errorbar(p1, infvar, yerr=d_infvar_b, fmt=fmt)
        elif errors == "jackknife":
            ax.errorbar(p1, infvar, yerr=d_infvar_j, fmt=fmt)
        else:
            raise ValueError('Errors must be "bootstrap" or "jackknife".')
        ax.set_xlabel("p1")
        ax.set_ylabel("Infected fraction variance")
        plt.show()


class Immunity(SIRS):
    """
    SIRS model cellular automaton with immunity in 2 dimensions.
    """

    IMMUNE = 3

    def randomize(self, immune_fraction=0):
        """
        Randomize the contents of the simulation grid.
        Make a certain fraction of cells permanently immune.
        """
        states = [self.SUSCEPTIBLE, self.INFECTED, self.RECOVERED]
        self.grid = np.random.choice(states, self.grid.shape)

        immune_mask = np.random.choice(
            [True, False], self.grid.shape, p=(immune_fraction, 1 - immune_fraction)
        )
        self.grid[immune_mask] = self.IMMUNE
        return self


def immune_fraction(sim, N, p1=0.5, p2=0.5, p3=0.5, max_iter=10_000):
    """
    Record the average infected fraction against immune fraction.
    """
    measurements = []
    headers = (
        ", ".join(("Immune Fraction", "Infected Fraction", "Infected Fraction Error")),
    )

    for fraction in np.linspace(0, 1, N):
        # track history of infected population
        history = []
        infected = lambda s: s.infected

        # run simulation
        gen = (
            sim.clear()
            .randomize(immune_fraction=fraction)
            .run(p1, p2, p3, steps=max_iter, eq=100)
        )
        gen = run_until(gen, lambda s: s.infected == 0)
        gen = record_measurements(gen, infected, record=history)
        gen = progress_bar(gen, desc=f"{(p1, p2, p3)}", total=max_iter)
        no_animation(gen)

        # record quantities
        history = np.array(history).T[0]

        inffrac = np.mean(history) / sim.grid.size * sim.elapsed / max_iter
        d_inffrac = np.std(history) / sim.grid.size * sim.elapsed / max_iter

        measurements.append((fraction, inffrac, d_inffrac))
    return measurements, headers


def display_immune_fraction_plot(path):
    """
    Display average infected fraction against immune fraction.
    """
    fraction, inffrac, d_inffrac = read_data(path)

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
        ax.errorbar(fraction, inffrac, yerr=d_inffrac, fmt=fmt)
        ax.set_xlabel("Immune fraction")
        ax.set_ylabel("Infected fraction")
        plt.show()


def prepare():
    sim = SIRS((50, 50))

    gen = sim.randomize().run(0.5, 0.5, 0.5)
    animate(gen)

    p1s = np.linspace(0, 1, 6)
    p3s = np.linspace(0, 1, 6)
    inffrac, infvar = record_phase_diagram(p1s, p3s, sim)
    print(inffrac, infvar)
    save_array(inffrac, Path("data/inffrac_5x5.txt"))
    save_array(infvar, Path("data/infvar_5x5.txt"))

    p1s = np.linspace(0, 1, 11)
    measurements, headers = record_phase_diagram_slice(p1s, sim, max_iter=1_000)
    save_data(measurements, path=Path("data/p1s_rough.txt"), headers=headers)

    sim = Immunity((50, 50))
    measurements, headers = immune_fraction(sim, 10, max_iter=1_000)
    save_data(measurements, path=Path("data/immune.txt"), headers=headers)


def display():
    sim = SIRS((50, 50)).randomize()

    # Suitable parameters
    animate(
        sim.run(0.7, 0.7, 0.7), title=lambda s: f"Dynamic Equilibrium, it={s.elapsed}"
    )
    animate(sim.run(0.8, 0.1, 0.012), title=lambda s: f"Waves, it={s.elapsed}")
    animate(sim.run(0.5, 0.6, 0.1), title=lambda s: f"Absorbing State, it={s.elapsed}")

    # Phase diagram (p2 = 0.5)
    inffrac = read_array(Path("data/inffrac_5x5.txt"))
    display_grid(inffrac, cmap="inferno")

    infvar = read_array(Path("data/infvar_5x5.txt"))
    display_grid(infvar, cmap="inferno")

    # Cut of variance along (p1 = p2 = 0.5)
    display_phase_diagram_slice(Path("data/p1s_rough.txt"))

    # Minimal immune fraction
    display_immune_fraction_plot(Path("data/immune.txt"))


def test():
    sim = SIRS((50, 50))

    gen = sim.randomize().run(0.5, 0.5, 0.5, steps=1000)
    no_animation(progress_bar(gen, total=1000, desc="test run with p1=p2=p3=0.5"))

    p1s = np.linspace(0, 1, 6)
    p3s = np.linspace(0, 1, 6)
    inffrac, infvar = record_phase_diagram(p1s, p3s, sim, max_iter=100)
    print(inffrac, infvar)
    save_array(inffrac, Path("test/inffrac.txt"), overwrite=True)
    save_array(infvar, Path("test/infvar.txt"), overwrite=True)

    p1s = np.linspace(0, 1, 11)
    measurements, headers = record_phase_diagram_slice(p1s, sim, max_iter=100)
    save_data(
        measurements, path=Path("test/p1s_rough.txt"), headers=headers, overwrite=True
    )

    sim = Immunity((50, 50))
    measurements, headers = immune_fraction(sim, 10, max_iter=100)
    save_data(
        measurements, path=Path("test/immune.txt"), headers=headers, overwrite=True
    )
    sim = SIRS((50, 50)).randomize()

    # Suitable parameters
    no_animation(
        progress_bar(sim.run(0.7, 0.7, 0.7, steps=100), total=100, desc="dynamic eq")
    )
    no_animation(
        progress_bar(sim.run(0.8, 0.1, 0.012, steps=100), total=100, desc="waves")
    )
    no_animation(
        progress_bar(sim.run(0.5, 0.6, 0.1, steps=100), total=100, desc="absorbing")
    )

    # Phase diagram (p2 = 0.5)
    inffrac = read_array(Path("test/inffrac.txt"))
    display_grid(inffrac, cmap="inferno")

    infvar = read_array(Path("test/infvar.txt"))
    display_grid(infvar, cmap="inferno")

    # Cut of variance along (p1 = p2 = 0.5)
    display_phase_diagram_slice(Path("test/p1s_rough.txt"))

    # Minimal immune fraction
    display_immune_fraction_plot(Path("test/immune.txt"))


def main():
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        plt.ion()
        print("running test script")
        test()
        return

    if not Path("data/").exists():
        print("running preparation script")
        prepare()
    display()


if __name__ == "__main__":
    main()
