from functools import partial

import numpy as np
from scipy.ndimage import laplace as _laplace
from tqdm import tqdm

from pvyzmgs.animation import animate, animate_with_cb, animate_with_plot
from pvyzmgs.simulation import Simulation

laplacian = partial(_laplace, mode="wrap")


class DiffusionSimulation(Simulation):
    def update(self):
        self.grid += self.D * self.dt / self.dx**2 * laplacian(self.grid)

    def run(self, D, dt=1, dx=1, steps=10_000, **kwargs):
        self.D = D
        self.dt = dt
        self.dx = dx
        return super().run(steps, **kwargs)


def main():
    sim = DiffusionSimulation().randomize()
    title_func = lambda s: f"elapsed={s.elapsed}"

    gen = sim.run(0.2)
    gen = tqdm(gen).__iter__()
    animate(gen, title=title_func)


if __name__ == "__main__":
    main()
