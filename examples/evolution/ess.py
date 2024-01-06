from abc import abstractmethod
from collections import Counter, defaultdict
from pathlib import Path
from random import randint, random

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from pvyzmgs.animation import no_animation
from pvyzmgs.data import read_array, read_data, save_array, save_data
from pvyzmgs.simulation import Simulation, progress_bar, record_measurements, run_until


class TeamVsSolo(Simulation):
    SOLO = 0
    TEAMWORK = 1

    def __init__(self, capacity=100, start=None, team_frac=None) -> None:
        """
        Create a simulation with a starting population.
        """
        self.capacity = capacity
        if start and team_frac:
            self.reset(start, team_frac)

    @classmethod
    def from_array(cls, *args, **kwargs):
        raise NotImplemented

    def randomize(self, *args, **kwargs):
        raise NotImplemented

    def reset(self, start=100, team_frac=0.5):
        """
        Reset the population.
        """
        self.population = [
            self.TEAMWORK if i / start < team_frac else self.SOLO for i in range(start)
        ]
        self.elapsed = 0
        return self

    def update(self):
        """
        Update population via "reproduction".
        """
        # shuffle population
        population = sorted(self.population, key=lambda _: random())
        trees = defaultdict(list)
        for blob in population:
            trees[randint(1, self.capacity)].append(blob)

        self.population = []
        for tree, blobs in trees.items():
            if len(blobs) == 1:
                (blob,) = blobs  # unpack blob
                self.population += [blob, blob]
                continue

            a, b, *_ = blobs
            new, extra = divmod(self.matrix[a][b], 1)
            self.population += [a] * (int(new) + (random() < extra))
            new, extra = divmod(self.matrix[b][a], 1)
            self.population += [b] * (int(new) + (random() < extra))

    def run(self, reward_matrix, steps, eq=0):
        """
        Run simulation using a reward matrix.
        """
        self.matrix = reward_matrix
        for _ in range(eq):
            self.update()
        return super().run(steps)

    @property
    def population_fractions(self):
        c = Counter(self.population)
        return [v / c.total() for v in c.values()]

    @property
    def solo(self):
        return sum(blob == self.SOLO for blob in self.population)

    @property
    def teamwork(self):
        return sum(blob == self.TEAMWORK for blob in self.population)


def compute_rainbow_run(p1, p2, sim, steps, savepath):
    histories = []
    for i in tqdm(range(1, steps)):
        # print(i/steps)
        sim.reset(start=steps, team_frac=i / steps)
        reward_matrix = [[2 - p1, 0.5], [1.5, 1 - p2]]
        its = 100
        history = []
        gen = sim.run(reward_matrix, its, eq=0)
        gen = run_until(gen, lambda s: s.solo == 0 or s.teamwork == 0)
        # gen = progress_bar(gen, total=its, leave=False)
        gen = record_measurements(
            gen, lambda s: s.solo, lambda s: s.teamwork, record=history
        )
        no_animation(gen)

        fractions = [b / (a + b) for a, b in history]
        if len(history) < its:
            fractions += ([0] if sim.teamwork == 0 else [1]) * (its - len(history))
        histories.append(fractions)

    save_data(histories, savepath, overwrite=True)


def display_rainbow_run(path):
    data = read_data(path).T
    cmap = mpl.colormaps["viridis"]
    steps = len(data)
    for i, line in enumerate(data):
        # print((i + 1)/steps)
        plt.plot(line, color=cmap((i + 1) / steps), linewidth=0.8)
    plt.xlabel("iteration")
    plt.ylabel("teamwork fraction")
    plt.show()


def heatmap(p1s, p2s, sim):
    frac_avg = np.zeros((p1s.size, p2s.size))
    frac_var = np.zeros((p1s.size, p2s.size))
    for i, p1 in tqdm(enumerate(p1s), total=p1s.size):
        for j, p2 in tqdm(enumerate(p2s), total=p2s.size, leave=False):
            reward_matrix = np.array([[2 - p1, 0.5], [1.5, 1 - p2]])
            its = 200
            history = []

            sim.reset(sim.capacity)
            gen = sim.run(reward_matrix, its, eq=50)
            gen = run_until(gen, lambda s: s.solo == 0 or s.teamwork == 0)
            gen = record_measurements(
                gen, lambda s: s.solo, lambda s: s.teamwork, record=history
            )
            # gen = progress_bar(gen, total=its, desc=f'{p1 = }, {p2 = }', leave=False)
            no_animation(gen)

            fractions = [b / (a + b) for a, b in history]
            if len(history) < its:
                fractions += ([0] if sim.teamwork == 0 else [1]) * (its - len(history))
            frac_avg[i, j] = np.mean(fractions)
            frac_var[i, j] = np.var(fractions)
    return frac_avg, frac_var


def multirun_heatmap(p1s, p2s, sim):
    frac_avg = np.zeros((p1s.size, p2s.size))
    frac_var = np.zeros((p1s.size, p2s.size))
    for i, p1 in tqdm(enumerate(p1s), total=p1s.size):
        for j, p2 in tqdm(enumerate(p2s), total=p2s.size, leave=False):
            reward_matrix = ((2 - p1, 0.5), (1.5, 1 - p2))
            fractions = []
            for _ in range(20):
                history = []
                its = 100

                sim.reset(sim.capacity)
                gen = sim.run(reward_matrix, 1, eq=20)
                gen = run_until(gen, lambda s: s.solo == 0 or s.teamwork == 0)
                gen = record_measurements(
                    gen, lambda s: s.solo, lambda s: s.teamwork, record=history
                )
                # gen = progress_bar(gen, total=its, desc=f'{p1 = }, {p2 = }', leave=False)
                no_animation(gen)

                a = sim.solo
                b = sim.teamwork
                fractions.append(b / (a + b))

            frac_avg[i, j] = np.mean(fractions)
            frac_var[i, j] = np.var(fractions)
    return frac_avg, frac_var


def main():
    # sim = TeamVsSolo(capacity=1000)
    # cmap = mpl.colormaps['inferno']

    # steps = 1000
    # histories = []
    # for i in tqdm(range(1, steps)):
    #     # print(i/steps)
    #     sim.reset(start=1000, team_frac=i/steps)
    #     reward_matrix = [
    #         [7/4, 2/4],
    #         [6/4, 3/4]
    #     ]
    #     its = 200
    #     history = []
    #     gen = sim.run(reward_matrix, its)
    #     # gen = progress_bar(gen, total=steps)
    #     gen = record_measurements(gen, lambda s: s.solo, lambda s: s.teamwork, record=history)
    #     no_animation(gen)

    #     fractions = [a / (a + b) for a, b in history]
    #     histories.append(fractions)

    # save_data(histories, Path('data/fractions_1k.txt'), overwrite=True)

    # data = read_data(Path('data/fractions_1k.txt')).T
    # steps = len(data)
    # for i, line in enumerate(data):
    #     # print((i + 1)/steps)
    #     plt.plot(line, color=cmap(1 - (i + 1)/steps), linewidth=0.8)
    # plt.xlabel('iteration')
    # plt.ylabel('teamwork fraction')
    # plt.show()

    # sim = TeamVsSolo(capacity=1000)
    # p1s = np.linspace(1/4, 4/4, 30 + 1)
    # p2s = np.linspace(1/4, 4/4, 30 + 1)
    # extent = (1/4, 4/4, 1/4, 4/4)

    sim = TeamVsSolo(capacity=2000)
    p1s = np.linspace(0 / 4, 4 / 4, 50 + 1)
    p2s = np.linspace(0 / 4, 4 / 4, 50 + 1)
    extent = (0 / 4, 4 / 4, 0 / 4, 4 / 4)
    frac_avg, frac_var = heatmap(p1s, p2s, sim)

    save_array(frac_avg, Path("data/frac_avg_fixed.txt"), overwrite=True)
    save_array(frac_var, Path("data/frac_var_fixed.txt"), overwrite=True)

    frac_avg = read_array(Path("data/frac_avg_fixed.txt"))
    plt.imshow(frac_avg.T, origin="lower", extent=extent)
    plt.title("Teamwork Fraction")
    plt.xlabel("fighting penalty")
    plt.ylabel("cooperation penalty")
    plt.colorbar()
    plt.show()

    frac_var = read_array(Path("data/frac_var_fixed.txt"))
    plt.imshow(frac_var.T, origin="lower", extent=extent, vmax=np.mean(frac_var))
    plt.title("Teamwork Fraction Variance")
    plt.xlabel("fighting penalty")
    plt.ylabel("cooperation penalty")
    plt.colorbar()
    # plt.show()

    # sim = TeamVsSolo(capacity=1000)
    # compute_rainbow_run(3/4, 1/4, sim, 1000, Path('data/rainbow_75_25.txt'))
    # compute_rainbow_run(1/4, 3/4, sim, 1000, Path('data/rainbow_25_75.txt'))
    # compute_rainbow_run(3/4, 3/4, sim, 1000, Path('data/rainbow_75_75.txt'))

    # display_rainbow_run(Path('data/rainbow_25_75.txt'))
    # display_rainbow_run(Path('data/rainbow_75_25.txt'))
    # display_rainbow_run(Path('data/rainbow_75_75.txt'))

    # sim = TeamVsSolo(capacity=100)
    # p1s = np.linspace(0/4, 4/4, 20 + 1)
    # p2s = np.linspace(0/4, 4/4, 20 + 1)
    # extent = (0/4, 4/4, 0/4, 4/4)
    # frac_avg, frac_var = multirun_heatmap(p1s, p2s, sim)

    # save_array(frac_avg, Path('data/multi_frac_avg_coarse.txt'), overwrite=True)
    # save_array(frac_var, Path('data/multi_frac_var_coarse.txt'), overwrite=True)

    # frac_avg = read_array(Path('data/multi_frac_avg_coarse.txt'))
    # frac_var = read_array(Path('data/multi_frac_var_coarse.txt'))

    # plt.imshow(frac_avg.T, origin='lower', extent=extent)
    # plt.title('Teamwork Fraction')
    # plt.xlabel('fighting penalty')
    # plt.ylabel('cooperation penalty')
    # plt.colorbar()
    # plt.show()

    # plt.imshow(np.log(frac_var.T), origin='lower', extent=extent)
    # plt.title('Teamwork Fraction Variance (log scale)')
    # plt.xlabel('fighting penalty')
    # plt.ylabel('cooperation penalty')
    # plt.colorbar()
    # plt.show()


if __name__ == "__main__":
    main()
