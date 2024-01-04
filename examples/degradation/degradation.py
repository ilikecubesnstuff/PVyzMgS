from abc import abstractmethod
from functools import partial
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import laplace as _laplace
from scipy.optimize import minimize_scalar

from pvyzmgs.animation import animate, animate_with_cb, animate_with_plot, no_animation
from pvyzmgs.data import read_array, read_data, save_array, save_data
from pvyzmgs.simulation import Simulation, progress_bar, run_until, skip_frames

laplacian = partial(_laplace, mode="wrap")


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


class DegradationSimulation(Convergence, UseRho):
    def convergent_value(self):
        return self.avg_phi

    def update(self):
        super().update()
        phi = self.grid
        phi += self.dt * (
            self.D / self.dx**2 * laplacian(phi) + self.rho - self.k * phi
        )

    def run(self, D, k, *, dx=1, dt=1, **kwargs):
        self.D = D
        self.k = k
        self.dx = dx
        self.dt = dt
        return super().run(**kwargs)

    @property
    def avg_phi(self):
        return np.mean(self.grid)


class AdvectedDegradationSimulation(DegradationSimulation):
    def update(self):
        super().update()
        phi = self.grid

        axes = [np.arange(n) for n in phi.shape]
        x, y = np.meshgrid(*axes)
        gradphix = np.roll(phi, +1, axis=1) - np.roll(phi, -1, axis=1)
        phi += self.dt * (
            self.D / self.dx**2 * laplacian(phi)
            + self.rho
            - self.k * phi
            + self.v0 / (2 * self.dx) * np.sin(2 * np.pi * y / 50) * gradphix
        )

    def run(self, v0, *args, **kwargs):
        self.v0 = v0
        return super().run(*args, **kwargs)


def run_animation(sigma, k, D, v0, dx, dt, max_iter=10_000):
    shape = (50, 50)
    axes = [np.arange(n) for n in shape]
    coords = np.meshgrid(*axes)
    d_sq = sum((n / 2 - coord) ** 2 for n, coord in zip(shape, coords))
    rho = np.exp(-d_sq / sigma**2)
    # rho = np.sin(-2*d_sq/sigma**2)**2

    # plt.imshow(rho, cmap='inferno')
    # plt.colorbar()
    # plt.show()

    sim = AdvectedDegradationSimulation(rho).randomize(0.5, 0.1)
    title = lambda s: f"{s.elapsed=}, {s.avg_phi=:2f}"

    gen = sim.run(v0, D, k, dx=dx, dt=dt, steps=max_iter)
    gen = progress_bar(gen, total=max_iter)
    gen = run_until(gen, lambda s: s.total_delta < 1e-6)
    gen = skip_frames(gen, 100)
    # animate_with_cb(gen, title=title)
    animate(gen, title=title)

    # d = np.sqrt(d_sq)
    # data_phi = sim.grid.flatten()
    # data_r = d.flatten()
    # data_phi = data_phi[data_r > 0]
    # data_r = data_r[data_r > 0]
    # plt.plot(data_r, data_phi, 'kx')
    # plt.show()

    # from scipy.optimize import curve_fit

    # def f(x, a, n):
    #     return a * x**-n

    # (a, n), pcov = curve_fit(f, data_r, data_phi)
    # print(a, n, pcov)

    # xs = np.linspace(0, np.max(data_r), 100)
    # ys = f(xs, a, n)

    # plt.plot(data_r, data_phi, 'kx')
    # plt.plot(xs, ys, 'r--')
    # plt.show()


def main():
    run_animation(sigma=10, k=0.01, D=1, v0=0.5, dx=1, dt=0.2)


if __name__ == "__main__":
    main()
