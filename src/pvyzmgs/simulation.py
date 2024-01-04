from abc import abstractmethod

import numpy as np
from tqdm import tqdm


class Simulation:
    def __init__(self, shape=(100, 100)) -> None:
        """
        Create a simulation grid given a tuple of lengths across each dimension.
        """
        self.dim = len(shape)
        self.grid = np.zeros(shape)
        self.elapsed = 0

    @classmethod
    def from_array(cls, arr):
        """
        Create a simulation grid from an existing numpy array.
        """
        inst = cls(shape=arr.shape)
        inst.grid = arr
        return inst

    def randomize(self, value=0, spread=0.1, *, distribution=np.random.normal):
        """
        Randomize the contents of the simulation grid.
        """
        self.grid = distribution(value, spread, size=self.grid.shape)
        return self

    def clear(self):
        """
        Reset simulation grid to zeros and set elapsed to zero.
        """
        self.grid = np.zeros(self.grid.shape)
        self.elapsed = 0
        return self

    @abstractmethod
    def update(self):
        """
        Apply an update algorithm to the simulation grid at each iteration.
        This must be implemented in a subclass.
        """
        pass

    @abstractmethod
    def run(self, steps):
        """
        Create a generator to yield simulation frames as the grid is updated.
        This must be implemented in a subclass, ending with `return super().run(steps)`.
        Custom class variables used in the update method must be initialized here.
        """
        for _ in range(steps):
            self.update()
            self.elapsed += 1
            yield self


# simulation generator utilities


def record_measurements(gen, *measures, record):
    """
    Record measurements by appending to a collection.
    """
    for state in gen:
        measurements = tuple(measure(state) for measure in measures)
        record.append(measurements)
        yield state


def progress_bar(gen, **kwargs):
    """
    Print a tqdm progress bar for the simulation.
    """
    gen = tqdm(gen, **kwargs).__iter__()
    yield from gen


def filter_frames(gen, /, pred):
    """
    Filter simulation states using a predicate.

    Usage:
        filter_frame(gen, pred=lambda s: s.elapsed%2 == 0)
            Yield every second state.
    """
    for state in gen:
        if pred(state):
            yield state


def skip_frames(gen, /, n, speedup=1):
    """
    Skip to every 'n'th simulation state. Optionally increase 'n' by a speedup factor for each yielded state.
    The speedup factor only affects the number of states skipped, not the speed of the simulation.

    Usage:
        skip_frames(gen, n=4, speedup=1.15)
            Yield every 4th state at the beginning, with a 15% speedup each state.
    """
    last = 0
    for state in gen:
        if (state.elapsed - last) % int(n) == 0:
            last = state.elapsed
            n *= speedup
            yield state


def run_until(gen, /, pred):
    """
    Stop the simulation if a given predicate is met.

    Usage:
        run_until(gen, pred=lambda s: sum(s.grid) == 0)
            Stop the simulation if the grid becomes entirely 0.
    """
    for state in gen:
        yield state
        if pred(state):
            break
