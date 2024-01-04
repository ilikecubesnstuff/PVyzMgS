def apply_pattern(grid, pattern, position):
    I, J = position
    pattern = pattern.replace(" ", "")
    for i, line in enumerate(pattern.split("\n")):
        for j, char in enumerate(line):
            if char != "_":
                grid[I + i][J + j] = not grid[I + i][J + j]


glider_nw = """
**_
*_*
*__
"""

glider_ne = """
_**
*_*
__*
"""

glider_sw = """
*__
*_*
**_
"""

glider = glider_se = """
__*
*_*
_**
"""

line_10 = """
**********
"""

r_pentomino = """
**_
_**
_*_
"""

line_7 = "*******"

lwss = """
*__*_
____*
*___*
_****
"""

gosper_glider_gun = """
________________________*___________
______________________*_*___________
____________**______**____________**
___________*___*____**____________**
**________*_____*___**______________
**________*___*_**____*_*___________
__________*_____*_______*___________
___________*___*____________________
____________**______________________
"""

eater = """
**__
*_*_
__*_
__**
"""


def main():
    from gol import GOL

    from pvyzmgs.animation import animate

    sim = GOL((50, 50))
    title_func = (
        lambda s: f"elapsed={s.elapsed}, evolving={s.evolving}, alive={s.alive}"
    )

    sim.clear()
    apply_pattern(sim.grid, glider, (0, 0))
    animate(sim.run(100), title=title_func)

    sim.clear()
    apply_pattern(sim.grid, line_10, (20, 20))
    animate(sim.run(100), title=title_func)

    sim.clear()
    apply_pattern(sim.grid, line_7, (20, 20))
    animate(sim.run(100), title=title_func)

    sim.clear()
    apply_pattern(sim.grid, gosper_glider_gun, (10, 0))
    apply_pattern(sim.grid, eater, (36, 40))
    animate(sim.run(10_000), title=title_func)


if __name__ == "__main__":
    main()
