import logging

import numpy as np

logger = logging.getLogger("mvp." + __name__)


def save_data(measurements, path, overwrite=False, headers=tuple()):
    if path.is_dir():
        path /= "data.txt"
    parent = path.parent
    if not parent.exists():
        logger.warning(f"directory {parent} does not exist - creating folders.")
        parent.mkdir(parents=True)
    if path.exists() and not overwrite:
        new_name = path.stem + "_copy"
        logger.warning(
            f"file at {path} already exists - writing to {new_name} instead ({overwrite=})"
        )
        path = parent / f"{new_name}.txt"
    logger.info(f"writing lattice states to {path}")
    with path.open("w+") as f:
        for header in headers:
            f.write("# " + str(header) + "\n")
        for line in measurements:
            f.write(", ".join(map(str, line)) + "\n")


def read_data(path, print_headers=True):
    with path.open("r") as f:
        for line in f.readlines():
            if line.startswith("# ") and print_headers:
                print(line[2:])
    data = np.loadtxt(path, delimiter=",")
    return data.T


def save_array(arr, path, overwrite=False):
    save_data([arr.flatten()], path=path, headers=(arr.shape,), overwrite=overwrite)


def read_array(path):
    with path.open("r") as f:
        header = f.readline()
        shape = eval(header[2:])
    arr = np.loadtxt(path, delimiter=",").reshape(shape)
    return arr
