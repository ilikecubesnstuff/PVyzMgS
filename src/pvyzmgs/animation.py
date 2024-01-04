import logging

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as fanim
from mpl_toolkits.axes_grid1 import make_axes_locatable

logger = logging.getLogger("mvp." + __name__)


def no_animation(gen) -> None:
    """
    Exhaust a simulation without generating an animation.
    """
    for _ in gen:
        pass


def animate(
    gen,
    component=lambda s: s.grid,
    title=lambda s: f"{s.elapsed}",
    path=None,
    **imshow_kwargs,
):
    """
    Animate a component of the simulation.
    """
    state = next(gen)

    fig, ax = plt.subplots()
    im = ax.imshow(component(state), **imshow_kwargs)
    plt.colorbar(im)

    def update(state):
        ax.set_title(f"{title(state)}")
        im.set_data(component(state))

    anim = fanim(fig, update, gen, interval=33, save_count=300)
    if path:
        if path.exists():
            logger.warning(f"file {path} already exists - overwriting animation")
            path.unlink()
        anim.save(path, fps=30)
    else:
        plt.show()
    return fig


def animate_with_cb(
    gen,
    component=lambda s: s.grid,
    title=lambda s: f"{s.elapsed}",
    path=None,
    **imshow_kwargs,
):
    """
    Animate a component of the simulation with an animated colorbar.
    """
    state = next(gen)

    fig, ax = plt.subplots()
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", "5%", "5%")

    im = ax.imshow(component(state), **imshow_kwargs)
    cb = fig.colorbar(im, cax=cax)

    def update(state):
        ax.cla()
        cax.cla()
        ax.set_title(f"{title(state)}")
        im = ax.imshow(state.grid, **imshow_kwargs)
        fig.colorbar(im, cax=cax)

    anim = fanim(fig, update, gen, interval=33, save_count=300)
    if path:
        if path.exists():
            logger.warning(f"file {path} already exists - overwriting animation")
            path.unlink()
        anim.save(path, fps=30)
    else:
        plt.show()
    return fig


def animate_with_plot(
    gen,
    func,
    component=lambda s: s.grid,
    title=lambda s: f"{s.elapsed}",
    path=None,
    **imshow_kwargs,
):
    """
    Animate a component of the simulation with an animated line graph of recorded measurements of a quantity over time.
    """
    state = next(gen)

    fig = plt.figure(figsize=(12, 3))
    ax = plt.subplot2grid((1, 3), (0, 0), 1, 1)
    line_ax = plt.subplot2grid((1, 3), (0, 1), 1, 2)
    im = ax.imshow(state.grid, **imshow_kwargs)
    xdata, ydata = [], []
    (line,) = line_ax.plot(xdata, ydata)

    def update(state):
        ax.set_title(f"{title(state)}")
        line_ax.set_title(f"value={func(state)}")

        xdata.append(state.elapsed)
        ydata.append(func(state))

        line_ax.set_xlim(min(xdata), max(xdata) + 1)
        line_ax.set_ylim(min(ydata), max(ydata) + 1)
        line_ax.figure.canvas.draw()

        line.set_data(xdata, ydata)
        im.set_data(state.grid)
        return im, line

    anim = fanim(fig, update, gen, interval=33, save_count=300)
    if path:
        if path.exists():
            logger.warning(f"file {path} already exists - overwriting animation")
            path.unlink()
        anim.save(path, fps=30)
    else:
        plt.show()
    return fig
