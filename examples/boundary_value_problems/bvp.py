from abc import abstractmethod
from functools import partial
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import laplace as _laplace
from scipy.optimize import minimize_scalar

from pvyzmgs.animation import animate, animate_with_plot, no_animation
from pvyzmgs.data import read_array, read_data, save_array, save_data
from pvyzmgs.simulation import Simulation, progress_bar, run_until, skip_frames

laplacian = partial(_laplace, mode="wrap")


class BoundaryCondition(Simulation):
    @abstractmethod
    def run(self, steps, bc=lambda g: None):
        """
        Create a generator to yield simulation frames as the grid is updated.
        This must be implemented in a subclass, ending with `return super().run(bc, steps)`.
        Custom class variables used in the update method must be initialized here.

        Boundary condition is a function applied to the grid.
        """
        for _ in range(steps):
            self.update()
            bc(self.grid)
            self.elapsed += 1
            yield self

    @staticmethod
    def none(shape=None):
        return lambda g: None

    @staticmethod
    def null_edge(shape):
        """
        Boundary condition of null values around the edge.
        """
        inner_shape = np.array(shape) - 2
        stencil = np.pad(np.ones(inner_shape), 1, "constant", constant_values=0)

        def boundary_condition(arr):
            arr *= stencil

        return boundary_condition


class UseRho(Simulation):
    def __init__(self, rho):
        """
        Create a simulation grid off an external grid variable rho.
        """
        self.rho = rho
        super().__init__(rho.shape)


class Convergence(Simulation):
    @abstractmethod
    def convergent_value(self):
        """
        Value that will be measured for convergence.
        This function must be defined in a subclass.
        """
        pass

    @abstractmethod
    def update(self):
        """
        Save last value before update.
        """
        self.last_total = self.convergent_value()

    @property
    def total_delta(self):
        """
        Calculate the change in value from the last iteration.
        """
        return abs(self.convergent_value() - self.last_total)


class PoissonJacobi(Convergence, UseRho, BoundaryCondition):
    def convergent_value(self):
        """
        Sum of "phi" across simulation grid.
        """
        return np.sum(self.grid)

    def update(self):
        """
        Update grid according to the Jacobi relaxation algorithm.
        """
        super().update()
        self.grid += (
            1 / (2 * self.dim) * (laplacian(self.grid) + self.dx**2 * self.rho)
        )

    def run(self, dx=1, steps=10_000, bc=BoundaryCondition.none()):
        """
        Run the relaxation algorithm.
        """
        self.dx = dx
        return super().run(steps=steps, bc=bc)


class PoissonGaussSeidel(PoissonJacobi):
    def __init__(self, rho):
        """
        Create a simulation grid off an external grid variable rho.
        Store even and odd indices for more efficient computation.
        """
        super().__init__(rho)
        self.evens = (
            np.sum(np.meshgrid(*[np.arange(i) for i in rho.shape]), axis=0) % 2 == 0
        )
        self.odds = (
            np.sum(np.meshgrid(*[np.arange(i) for i in rho.shape]), axis=0) % 2 == 1
        )

    def update(self):
        """
        Update grid according to the Gauss-Seidel algorithm.
        """
        super().update()
        temp = np.zeros(self.grid.shape)
        temp[self.evens] = (
            1
            / (2 * self.dim)
            * (
                laplacian(self.grid)
                + (2 * self.dim) * self.grid
                + self.dx**2 * self.rho
            )[self.evens]
        )
        temp[self.odds] = (
            1 / (2 * self.dim) * (laplacian(temp) + self.dx**2 * self.rho)[self.odds]
        )
        self.grid += temp - self.grid


class PoissonSOR(PoissonGaussSeidel):
    def update(self):
        """
        Update grid according to the successive over-relaxation algorithm.
        """
        super().update()
        temp = np.zeros(self.grid.shape)
        temp[self.evens] = (
            1
            / (2 * self.dim)
            * (
                laplacian(self.grid)
                + (2 * self.dim) * self.grid
                + self.dx**2 * self.rho
            )[self.evens]
        )
        temp[self.odds] = (
            1 / (2 * self.dim) * (laplacian(temp) + self.dx**2 * self.rho)[self.odds]
        )
        self.grid += self.w * (temp - self.grid)

    def run(self, w, steps=10_000, bc=BoundaryCondition.none(), **kwargs):
        """
        Run the successive over-relaxation algorithm.
        """
        self.w = w
        return super().run(steps=steps, bc=bc, **kwargs)


class WireBoundary(PoissonSOR):
    """
    Use "rho" as the z-component of the current J.
    """

    def __init__(self, jz):
        """
        Create a simulation grid off an external grid variable jz.
        """
        super().__init__(jz)
        self.jz = jz


def record_magnetic_field(name, tol=1e-5, bc=BoundaryCondition.none(), max_iter=10_000):
    """
    Record the magnetic field around a wire.
    """
    # setup
    jz = np.zeros((50, 50, 50))
    jz[25, 25] = 1
    bc = BoundaryCondition.null_edge(jz.shape)

    # run simulation
    sim = WireBoundary(jz)
    gen = sim.run(w=2, steps=max_iter, bc=bc)
    gen = run_until(gen, lambda s: s.total_delta < tol)
    gen = progress_bar(gen, total=max_iter)
    gen = skip_frames(gen, 50)
    animate(gen, lambda s: s.grid[:, :, 25], lambda s: s.total_delta)

    dazdx, dazdy, dazdz = np.gradient(sim.grid, edge_order=2)
    Bx = dazdy
    By = -dazdx

    save_array(sim.grid[:, :, 25], Path(f"bvp/a_{name}.txt"))
    save_array(np.array([Bx[:, :, 25], By[:, :, 25]]), Path(f"bvp/bfield_{name}.txt"))


def display_magnetic_field(name):
    """
    Display the magnetic field around a wire, as well as the drop-off in strength with distance.
    """
    a = read_array(Path(f"bvp/a_{name}.txt"))
    Bx, By = read_array(Path(f"bvp/bfield_{name}.txt"))

    plt.imshow(a, cmap="inferno")
    plt.colorbar()

    norm = np.sqrt(Bx**2 + By**2)
    plt.quiver(
        By / norm, Bx / norm, angles="xy", scale_units="xy", scale=1, color="white"
    )
    plt.show()
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
        plt.plot(a[25, 25:], fmt)
        plt.xlabel("Distance from center")
        plt.ylabel("Vector potential strength")
        plt.show()

        plt.plot(norm[25, 26:], fmt)
        plt.xlabel("Distance from center")
        plt.ylabel("B field strength")
        plt.show()


def optimal_w(sim: PoissonSOR, tol, bc=BoundaryCondition.none(), max_iter=10_000):
    """
    Use Scipy's optimization routines to find an optimal relaxation parameter.
    """

    def f(w):
        print(f"Relaxation: {w}")
        sim.clear()
        gen = sim.run(w, steps=max_iter, bc=bc)
        gen = run_until(gen, lambda s: s.total_delta < 1e-5)
        no_animation(gen)
        return sim.elapsed

    res = minimize_scalar(f, bounds=[1, 2], tol=tol)
    print(res)
    return res.x


def prepare():
    record_magnetic_field("test")


def display():
    rho = np.zeros((50, 50))
    rho[25, 25] = 1
    zslice = lambda s: s.grid

    bc = BoundaryCondition.null_edge(rho.shape)
    tol = 1e-5

    sim = PoissonJacobi(rho)
    gen = sim.run(bc=bc)
    gen = run_until(gen, lambda s: s.total_delta < tol)
    gen = skip_frames(gen, 50)
    animate(gen, zslice, cmap="flag")

    sim = PoissonGaussSeidel(rho)
    gen = sim.run(bc=bc)
    gen = run_until(gen, lambda s: s.total_delta < tol)
    gen = skip_frames(gen, 50)
    animate(gen, zslice, cmap="flag")

    sim = PoissonSOR(rho)
    gen = sim.run(w=2, bc=bc)
    gen = run_until(gen, lambda s: s.total_delta < tol)
    gen = skip_frames(gen, 50)
    animate(gen, zslice, cmap="flag")

    # Wire fields
    display_magnetic_field("test")

    # Optimal relaxation parameter
    rho = np.zeros((50, 50))
    rho[25, 25] = 1
    bc = BoundaryCondition.null_edge(rho.shape)
    sim = PoissonSOR(rho)
    w_opt = optimal_w(sim, 0.001, bc=bc)


def test():
    rho = np.zeros((50, 50))
    rho[25, 25] = 1
    zslice = lambda s: s.grid

    bc = BoundaryCondition.null_edge(rho.shape)
    tol = 1e-4

    sim = PoissonJacobi(rho)
    gen = sim.run(bc=bc)
    gen = run_until(gen, lambda s: s.total_delta < tol)
    gen = skip_frames(gen, 50)
    no_animation(progress_bar(gen))

    sim = PoissonGaussSeidel(rho)
    gen = sim.run(bc=bc)
    gen = run_until(gen, lambda s: s.total_delta < tol)
    gen = skip_frames(gen, 50)
    no_animation(progress_bar(gen))

    sim = PoissonSOR(rho)
    gen = sim.run(w=2, bc=bc)
    gen = run_until(gen, lambda s: s.total_delta < tol)
    gen = skip_frames(gen, 50)
    no_animation(progress_bar(gen))

    # Optimal relaxation parameter
    rho = np.zeros((50, 50))
    rho[25, 25] = 1
    bc = BoundaryCondition.null_edge(rho.shape)
    sim = PoissonSOR(rho)
    w_opt = optimal_w(sim, 0.01, bc=bc)


def main():
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        plt.ion()
        print("running test script")
        test()
        return

    if not Path("bvp/").exists():
        print("running preparation script")
        prepare()
    display()


if __name__ == "__main__":
    main()
