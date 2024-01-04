import matplotlib.pyplot as plt
import numpy as np


def display_grid(arr, **imshow_kwargs):
    plt.imshow(arr, origin="lower", **imshow_kwargs)
    plt.colorbar()
    plt.show()


def plot_measurements(path):
    data = np.loadtxt(path)
    plt.plot(data)
    plt.show()
