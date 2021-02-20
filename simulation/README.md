# Simulation of RNA transport in muscle fibers

<img align="right" src="img/03-L-C-P.gif" alt="03-L-C-P" width=300 border="1">

## Description
We developed a discrete-time Markov chain (DTMC)-based simulation to investigate how the different motion states available to RNAs affect their steady-state distribution in the myofiber cytoplasm. In the simulation, the following RNA transport states are modeled:

1. **low-mobility (L)**: a slow diffusion state with a low range of diffusion coefficients (median _D_: 3.9 x 10<sup>-4</sup> µm<sup>2</sup>/s)
2. **high-mobility (H)**: a fast diffusion state with a high range of diffusion coefficients (median _D_: 6.5 x 10<sup>-3</sup> µm<sup>2</sup>/s)
3. **crawling (C)**: a slow directed transport state in which RNAs inch along in the same direction (median speed: 0.11 μm/s)
4. **processive (P)**: a fast directed transport state in which RNAs travel quickly in the same direction (median speed: 0.63 μm/s)

In a C2C12 myotube model of muscle fibers, we observed RNAs moving in each of these states, and we measured diffusion coefficients and directed transport distances for each particle. The distributions of these measurements were smoothed and sampled to provide estimates for motion parameters in the simulation.

## Instructions

## Examples