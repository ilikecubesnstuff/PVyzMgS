# PVyzMgS Examples

## Ising Model

Simulation of a square lattice of spin-up or spin-down states (see [Ising model](https://en.wikipedia.org/wiki/Ising_model) on Wikipedia), including scripts for ferromagnetic and antiferromagnetic interactions.
Phase transitions are explored with both [Glauber dynamics](https://en.wikipedia.org/wiki/Glauber_dynamics) (random flips) and [Kawasaki dynamics](https://link.springer.com/chapter/10.1007/978-3-319-24777-9_18) (random swaps).

## Game of Life

Simulation of a square lattice of "dead" or "alive" states, updating using the rules of the [Game of Life](https://en.wikipedia.org/wiki/Conway's_Game_of_Life).
Includes presets for popular starting configurations ([gliders](https://conwaylife.com/wiki/Spaceship), [oscillators](https://conwaylife.com/wiki/Oscillator), [still-life](https://conwaylife.com/wiki/Still_life)), and measures the speed of gliders.

## SIRS Model

Simulation of a square lattice with 3 states (susceptible, infectious, and recovered, see [SIRS model](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology)).
Also includes simulations with a 4th state of "immune", and displays phase transitions between the different types of states (absorbing states, waves, and dynamic equilibria).

## Cahn Hilliard Simulation

Models phase separation with the [Cahn-Hilliard Equation](https://en.wikipedia.org/wiki/Cahn%E2%80%93Hilliard_equation).
An even mix produces [spinodal decomposition](https://en.wikipedia.org/wiki/Spinodal_decomposition) while uneven mixes produce [ostwald ripening](https://en.wikipedia.org/wiki/Ostwald_ripening).

## Boundary Value Problems

General classes aiding in the solving of boundary value problems are included here, but the specific problem being solved is modelling the electromagnetic fields around a charge and a current-carrying wire.
Convergence speed is optimized with the [Gauss-Seidel algorithm](https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method) and [successive over-relaxation](https://en.wikipedia.org/wiki/Successive_over-relaxation).

## Contact Process

A [contact process](https://en.wikipedia.org/wiki/Contact_process_(mathematics)) model is simulated on a square lattice.

## Degradation

A degradation process model is simulated on a square lattice.

## Rock, Paper, Scissors

Simulation of a square lattice with [rock, paper, and scissors](https://en.wikipedia.org/wiki/Rock_paper_scissors) states.
Analyses the dynamics of the minority fraction with both deterministic and random update rules.
