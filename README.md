# Bandwith Optimization

Network Optimization – ETH Zürich Spring 2025
Optimization Methods for Engineers (227-0087-00L)

This project applies advanced optimization techniques to solve a real-world network flow optimization problem. Developed as part of the ETH Zürich course Optimization Methods for Engineers (Spring Semester 2025), the implementation explores both exact and approximate methods to optimize routing cost and bandwidth under various constraints.

# Data

<https://sndlib.put.poznan.pl/home.action>


# Usage Inscructions

> run main.py with the following folder hierarchy for correct "import" usage
>```
> src/
> ├── main.py
> ├── algorithms/
> ├── inputs/
> ├── types/
> ├── utilities/
> ```

- run `utilities.parsing.parse_network()` to parse all input data: returns `Network`
- run `algorithms.integer_programming.run_IP(Network)` to optimize network: returns `ResultIP`
- run `algorithms.genetic_algorithm.run_GA(Network)` to optimize network: returns `ResultGA`

# QUICK NOTES

- both IP and GA seem to be working
- GA seems to be worse than IP right now
- GA needs to be optimized
- RUNTIMES of both algorithms are still needed
- GA code still has some hardcoded values, like 5 shortest paths per demand, or the parameters like population count, nr of generations
