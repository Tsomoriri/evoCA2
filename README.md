# Particle Swarm Optimization for Traveling Salesman Problem

This project implements and analyzes various Particle Swarm Optimization (PSO) algorithms for solving the Traveling Salesman Problem (TSP). The study compares different PSO variants and explores the impact of various hyperparameters on their performance.

## Algorithms Implemented

1. Standard PSO
2. Adaptive PSO (APSO)
3. Binary PSO (HPSO)
4. Spatial PSO (SPSO)
5. Differential Evolution PSO (DEPSO)
6. Predator-Prey PSO (PPPSO)
7. Random Sampling (for comparison)
8. Stochastic Hill Climber (for comparison)

## Experiments

The project includes several experiments to analyze the performance of these algorithms:

1. Population Size
2. Inertia Weight
3. Acceleration Coefficients
4. Wmin and Wmax (for APSO)
5. Neighborhood Size (for SPSO)
6. Crossover Rate and Mutation Rate (for DEPSO)
7. Fear Factor (for PPPSO)
8. Maximum Iterations

## TSP Instances

The algorithms are tested on three TSP instances of varying sizes:

- a29 (29 cities)
- a280 (280 cities)
- fl1400 (1400 cities)

## Key Findings

- The optimal population size varies across algorithms, with PSO and DEPSO performing best at 60 particles, while APSO, HPSO, SPSO, and PPPSO benefit from larger populations of 100 particles.
- Inertia weight significantly impacts performance, with optimal values varying by algorithm (e.g., 1.0 for PSO and HPSO, 0.4 for SPSO).
- Acceleration coefficients play a crucial role in balancing exploration and exploitation, with optimal values often around 2.0 for cognitive and 1.5-2.0 for social components.
- Adaptive mechanisms in APSO show promise in dynamically adjusting search behavior.
- Neighborhood size in SPSO affects performance differently based on problem size, with smaller neighborhoods (3) better for larger instances and larger neighborhoods (9) for smaller instances.
- DEPSO's performance is sensitive to crossover and mutation rates, with optimal values of 0.8 and 0.2 respectively.
- The fear factor in PPPSO needs to be tuned based on problem size, with lower values (0.2) for smaller instances and higher values (0.4) for larger ones.



## Future Work

- Implement more advanced PSO variants
- Explore larger TSP instances
- Investigate multi-objective optimization approaches
- Develop hybrid algorithms combining PSO with other metaheuristics

## References

1. Engelbrecht, A.P. (2007). Computational Intelligence: An Introduction. John Wiley & Sons.
2. [Include other relevant references from the report]

## Author

Sushen Yadav

