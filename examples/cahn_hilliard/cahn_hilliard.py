from functools import partial

import numpy as np
from scipy.ndimage import laplace as _laplace

from pvyzmgs.animation import animate, animate_with_plot, no_animation
from pvyzmgs.simulation import Simulation, skip_frames

laplacian = partial(_laplace, mode="wrap")


class CahnHilliardSimulation(Simulation):
    def update(self):
        """
        Step forward in time using forward-Euler scheme for the Cahn-Hilliard equation.
        """
        phi = self.grid
        mu = -self.a * phi + self.b * phi**3 - self.k / self.dx**2 * laplacian(phi)
        phi += self.M * self.dt / self.dx**2 * laplacian(mu)

    def run(self, M, k, a=0.1, b=None, *, dx=1, dt=1, steps=100_000):
        """
        Evolve system according to a forward-Euler scheme for the Cahn-Hilliard equation.
        """
        self.M = M
        self.k = k
        self.a = a
        self.b = b or a
        self.dx = dx
        self.dt = dt
        return super().run(steps=steps)

    @property
    def free_energy(self):
        """
        Calculate the free energy of the system.
        """
        phi = self.grid
        gradphi_x, gradphi_y = np.gradient(phi, edge_order=2)
        gradphi_sq = gradphi_x**2 + gradphi_y**2
        return np.sum(
            -self.a / 2 * phi**2 + self.a / 4 * phi**4 + self.k / 2 * gradphi_sq
        )


def main():
    sim = CahnHilliardSimulation()
    free_energy = lambda s: s.free_energy

    # Spinodal decomposition + free energy
    gen = sim.randomize(0).run(0.1, 0.1, 0.1, 0.1)
    gen = skip_frames(gen, 500, speedup=1.02)
    animate_with_plot(gen, free_energy, vmin=-1, vmax=1)

    # Droplet growth + free energy
    gen = sim.randomize(0.5).run(0.1, 0.1, 0.1, 0.1)
    gen = skip_frames(gen, 500, speedup=1.02)
    animate_with_plot(gen, free_energy, vmin=-1, vmax=1)


if __name__ == "__main__":
    main()
