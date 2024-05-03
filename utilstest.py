import math
import os
import math
import time
import tqdm
import random
import multiprocessing

import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from math import sqrt, tanh, exp
from joblib import Parallel, delayed

"""
/*********************************************************************
 *  Data processing for TSP instances
 *
 * ProcessData class
 *
 * Reads a TSP file, calculates the distance matrix, and saves it to a CSV file.
 * Also provides a method to load the distance matrix from the CSV file.
 *
 * Member Variables:
 * - tsp_file_path (str): Path to the TSP file.
 * - csv_file_path (str): Path to the CSV file for storing the distance matrix.
 * - data (list): List of city coordinates read from the TSP file.
 * - distance_matrix (list): 2D list representing the distance matrix.
 * - tsp_instance_list (list): List containing the loaded TSP instance.
 *
 * Member Functions:
 * - read_tsp_file(): Reads the TSP file and extracts the city coordinates.
 * - calculate_distance_matrix(): Calculates the distance matrix based on the city coordinates.
 * - save_distance_matrix_to_csv(): Saves the distance matrix to a CSV file.
 * - load_distance_matrix_from_csv(): Loads the distance matrix from the CSV file.
 * - process(): Performs the complete processing of the TSP file.
 * - get_tsp_instance_list(): Returns the loaded TSP instance list.
 * - name(): Returns the name of the TSP instance based on the TSP file name.
 *********************************************************************/

"""


class ProcessData:
    def __init__(self, tsp_file_path, csv_file_path):
        self.tsp_file_path = tsp_file_path
        self.csv_file_path = csv_file_path
        self.data = None
        self.distance_matrix = None
        self.tsp_instance_list = None

    def read_tsp_file(self):
        """
        Reads the TSP file and extracts the city coordinates.

        Postconditions:
        - self.data is populated with the city coordinates.
        """
        with open(self.tsp_file_path, 'r') as file:
            lines = file.readlines()
            self.data = []
            for line in lines:
                if line.strip().startswith('NODE_COORD_SECTION'):
                    break
            for line in lines[lines.index(line) + 1:]:
                if line.strip() == 'EOF':
                    break
                node_id, x, y = map(float, line.strip().split())
                self.data.append([x, y])

    def calculate_distance_matrix(self):
        """
        Calculates the distance matrix based on the city coordinates.

        Preconditions:
        - self.data must be populated with city coordinates.

        Postconditions:
        - self.distance_matrix is populated with the calculated distances.
        """
        n = len(self.data)
        self.distance_matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = self.data[i]
                x2, y2 = self.data[j]
                distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                self.distance_matrix[i][j] = self.distance_matrix[j][i] = distance

    def save_distance_matrix_to_csv(self):
        """
        Saves the distance matrix to a CSV file.

        Preconditions:
        - self.distance_matrix must be populated.
        """
        df_distance = pl.DataFrame(self.distance_matrix)
        df_distance.write_csv(self.csv_file_path)

    def load_distance_matrix_from_csv(self):
        """
        Loads the distance matrix from the CSV file.

        Postconditions:
        - self.tsp_instance_list is populated with the loaded distance matrix.
        """
        df_d = pl.read_csv(self.csv_file_path)
        self.tsp_instance_list = df_d.to_numpy().tolist()

    def process(self):
        """
        Performs the complete processing of the TSP file.

        Postconditions:
        - The TSP file is read, distance matrix is calculated and saved to CSV.
        - The distance matrix is loaded from CSV into self.tsp_instance_list.
        """
        self.read_tsp_file()
        self.calculate_distance_matrix()
        self.save_distance_matrix_to_csv()
        self.load_distance_matrix_from_csv()

    def get_tsp_instance_list(self):
        """
        Returns the loaded TSP instance list.

        Returns:
        - self.tsp_instance_list: List containing the loaded TSP instance.
        """
        self.load_distance_matrix_from_csv()
        return self.tsp_instance_list

    def name(self):
        """
        Returns the name of the TSP instance based on the TSP file name.

        Returns:
        - tsp_name (str): Name of the TSP instance.
        """
        tsp_name = os.path.basename(self.tsp_file_path).split('.')[0]
        return tsp_name


# # Usage example
# tsp_file_path = 'a280.tsp'
# csv_file_path = 'a280_distance_matrix.csv'
#
# data_processor = ProcessData(tsp_file_path, csv_file_path)
'''
/*****************************************************************************
* BasePSO abstract base class
*
* Represents the base class for Particle Swarm Optimization algorithms.
*
* Class Variables:
* - Particle: Inner class representing a particle in the swarm.
*
* Member Variables:
* - tsp_instance (list): 2D list representing the TSP instance.
* - population_size (int): Size of the swarm population.
* - max_iterations (int): Maximum number of iterations for the optimization.
* - w (float): Inertia weight.
* - c1 (float): Cognitive coefficient.
* - c2 (float): Social coefficient.
* - gbest_solution (list): Global best solution found by the swarm.
* - gbest_fitness (float): Fitness value of the global best solution.
*
* Member Functions (Abstract):
* - create_particle(): Creates a new particle for the swarm.
* - calculate_fitness(solution): Calculates the fitness value of a solution.
* - update_velocity(particle): Updates the velocity of a particle.
* - update_position(particle): Updates the position of a particle.
*
Member Functions:
* - optimize(): Performs the PSO optimization
*****************************************************************************/
'''


class BasePSO(ABC):
    class Particle:
        def __init__(self, solution, fitness):
            self.solution = solution
            self.fitness = fitness
            self.pbest_solution = solution
            self.pbest_fitness = fitness
            self.velocity = [0] * len(solution)

    def __init__(self, tsp_instance, population_size, max_iterations, w, c1, c2):
        self.tsp_instance = tsp_instance
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.gbest_solution = None
        self.gbest_fitness = float('inf')

    @abstractmethod
    def create_particle(self):
        pass

    @abstractmethod
    def calculate_fitness(self, solution):
        pass

    def update_velocity(self, particle):
        """
        Updates the velocity of a particle.

        Parameters:
        - particle (Particle): The particle to update the velocity for.

        Postconditions:
        - The velocity of the particle is updated based on PSO equations.
        """
        for i in range(len(particle.velocity)):
            r1 = random.random()
            r2 = random.random()
            cognitive_velocity = self.c1 * r1 * \
                (particle.pbest_solution[i] - particle.solution[i])
            social_velocity = self.c2 * r2 * \
                (self.gbest_solution[i] - particle.solution[i])
            particle.velocity[i] = self.w * particle.velocity[i] + \
                cognitive_velocity + social_velocity

    @abstractmethod
    def update_position(self, particle):
        pass

    def optimize(self):
        """
        Performs the PSO optimization.

        Returns:
        - gbest_solution (list): Global best solution found by the swarm.
        - gbest_fitness (float): Fitness value of the global best solution.
        - convergence_data (list): List of best fitness values at each iteration.
        - runtime (float): Total runtime of the optimization.
        """
        start_time = time.time()
        swarm = [self.create_particle() for _ in range(self.population_size)]

        # Initialize gbest_solution and gbest_fitness
        self.gbest_solution = swarm[0].solution
        self.gbest_fitness = swarm[0].fitness
        for particle in swarm[1:]:
            if particle.fitness < self.gbest_fitness:
                self.gbest_solution = particle.solution
                self.gbest_fitness = particle.fitness
        convergence_data = []
        for iteration in range(self.max_iterations):
            for particle in swarm:
                self.update_velocity(particle)
                self.update_position(particle)

                # Update gbest_solution and gbest_fitness
                if particle.fitness < self.gbest_fitness:
                    self.gbest_solution = particle.solution
                    self.gbest_fitness = particle.fitness
            convergence_data.append(self.gbest_fitness)
        end_time = time.time()
        runtime = end_time - start_time
        return self.gbest_solution, self.gbest_fitness, convergence_data, runtime


""" /*************************************************************************
    * StochasticHillClimber class for solving TSP using the Stochastic Hill Climber algorithm.
    *
    * Douglas Wilhelm Harder
    * 2023-05-02
    *
    * This class implements the Stochastic Hill Climber algorithm for solving the Traveling Salesman Problem (TSP).
    * It starts with a random solution and iteratively tries to improve it by swapping two cities in the solution.
    * If the new solution is better, it becomes the new best solution. The algorithm stops when the maximum number
    * of iterations is reached or when the solution does not improve for a specified number of iterations.
    *
    * Member Variables:
    *     tsp_instance (list): The TSP instance represented as a distance matrix.
    *     max_iterations (int): The maximum number of iterations for the optimization process.
    *     max_stagnation (int): The maximum number of iterations without improvement before terminating the search.
    *
    * Member Functions:
    *     __init__(self, tsp_instance, max_iterations, max_stagnation)
    *         Initializes the StochasticHillClimber class with the given parameters.
    *
    *     optimize(self)
    *         Runs the optimization process and returns the best solution found, its fitness, convergence data, and runtime.
    ************************************************************************/
"""


class StochasticHillClimber:
    def __init__(self, tsp_instance, max_iterations, max_stagnation):
        self.tsp_instance = tsp_instance
        self.max_iterations = max_iterations
        self.max_stagnation = max_stagnation

    def optimize(self):
        """
        Runs the optimization process and returns the best solution found, its fitness, convergence data, and runtime.

        Returns:
            tuple: A tuple containing the best solution, its fitness, convergence data, and runtime.
        """
        iteration = 0
        stagnation_counter = 0
        num_cities = len(self.tsp_instance)
        # Initialize with a random solution
        best_solution = list(range(num_cities))
        random.shuffle(best_solution)  # Shuffle the initial solution
        best_fitness = PSO.calculate_fitness(best_solution, self.tsp_instance)
        convergence_data = []
        start_time = time.time()

        
        while iteration < self.max_iterations:
            new_solution = list(best_solution)

            if len(new_solution) >= 2:
                i, j = random.sample(range(len(new_solution)), 2)
                new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
                new_fitness = PSO.calculate_fitness(
                    new_solution, self.tsp_instance)

                if new_fitness < best_fitness:
                    best_solution = new_solution
                    best_fitness = new_fitness
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

                convergence_data.append(best_fitness)
            else:
                # Handle the case where the length of new_solution is less than 2
                # In this case, there is no way to improve the solution by swapping cities
                # Therefore, we can simply continue to the next iteration
                stagnation_counter += 1
                convergence_data.append(best_fitness)

            iteration += 1
        
        runtime = time.time() - start_time
        return best_solution, best_fitness, convergence_data, runtime


""" /*************************************************************************
    * RandomSampling class for solving TSP using random sampling.
    *
    * Douglas Wilhelm Harder
    * 2023-05-02
    *
    * This class implements a random sampling approach for solving the Traveling Salesman Problem (TSP).
    * It generates a specified number of random solutions and returns the best solution found among them.
    *
    * Member Variables:
    *     tsp_instance (list): The TSP instance represented as a distance matrix.
    *     num_samples (int): The number of random samples to generate.
    *
    * Member Functions:
    *     __init__(self, tsp_instance, num_samples)
    *         Initializes the RandomSampling class with the given parameters.
    *
    *     optimize(self)
    *         Generates random solutions and returns the best solution found, its fitness, convergence data, and runtime.
    ************************************************************************/
"""


class RandomSampling:
    def __init__(self, tsp_instance, num_samples):
        self.tsp_instance = tsp_instance
        self.num_samples = num_samples

    def optimize(self):
        """
        Generates random solutions and returns the best solution found, its fitness, convergence data, and runtime.

        Returns:
            tuple: A tuple containing the best solution, its fitness, convergence data, and runtime.
        """
        best_solution = None
        best_fitness = float('inf')
        runtime = 0
        convergence_data = []

        for _ in range(self.num_samples):
            solution = list(range(len(self.tsp_instance)))
            random.shuffle(solution)
            fitness = PSO.calculate_fitness(solution, self.tsp_instance)

            if fitness < best_fitness:
                best_solution = solution
                best_fitness = fitness

            convergence_data.append(best_fitness)

        return best_solution, best_fitness, convergence_data, runtime


"""
/*********************************************************************
* PSO class
*
* Represents the Particle Swarm Optimization algorithm for solving 
* the TSP.
* Inherits from BasePSO.
*
* Member Functions:
* - create_particle(): Creates a new particle for the swarm.
* - calculate_fitness(solution, tsp_instance): Calculates the fitness
*   value of a solution.
* - update_position(particle): Updates the position of a particle.
*********************************************************************/
"""


class PSO(BasePSO):
    def create_particle(self):
        """
        Creates a new particle for the swarm.

        Returns:
        - particle (Particle): The created particle with random solution and fitness.
        """
        solution = random.sample(
            range(1, len(self.tsp_instance) + 1), len(self.tsp_instance))
        fitness = self.calculate_fitness(solution, self.tsp_instance)
        return BasePSO.Particle(solution, fitness)

    @staticmethod
    def calculate_fitness(solution, tsp_instance):
        """
        Calculates the fitness value of a solution.

        Parameters:
        - solution (list): The solution to calculate the fitness for.
        - tsp_instance (list): 2D list representing the TSP instance.

        Returns:
        - total_distance (float): The total distance of the solution.
        """
        total_distance = 0
        for i in range(len(solution)):
            city1 = solution[i] - 1  # Adjust the indexing to start from 0
            # Adjust the indexing to start from 0
            city2 = solution[(i + 1) % len(solution)] - 1
            total_distance += tsp_instance[city1][city2]
        return total_distance

    def update_position(self, particle):
        """
        Updates the position of a particle.

        Parameters:
        - particle (Particle): The particle to update the position for.

        Postconditions:
        - The position of the particle is updated based on the velocity.
        - The particle's solution, fitness, pbest_solution, and pbest_fitness are updated if a better solution is found.
        """
        new_solution = particle.solution[:]
        for i in range(len(new_solution)):
            if random.random() < math.tanh(abs(particle.velocity[i])):
                j = random.randint(0, len(new_solution) - 1)
                new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
        new_fitness = self.calculate_fitness(new_solution, self.tsp_instance)
        if new_fitness < particle.fitness:
            particle.solution = new_solution
            particle.fitness = new_fitness
            if new_fitness < particle.pbest_fitness:
                particle.pbest_solution = new_solution
                particle.pbest_fitness = new_fitness


"""
    *********************************************************************
    * APSO class for solving TSP using Adaptive Particle Swarm Optimization.
    *
    * This class extends the PSO class and implements the Adaptive PSO (APSO) algorithm for solving the
    * Traveling Salesman Problem (TSP). It introduces an adaptive inertia weight that varies over the
    * course of the optimization process.
    *
    * Member Variables:
    *     w_min (float): The minimum value of the inertia weight.
    *     w_max (float): The maximum value of the inertia weight.
    *
    * Member Functions (Mutators):
    *     __init__(self, tsp_instance, population_size, max_iterations, w_min, w_max, c1, c2)
    *         Initializes the APSO class with the given parameters.
    *
    *     update_velocity(self, particle, iteration)
    *         Updates the velocity of a particle based on its personal best, the global best, and the adaptive inertia weight.
    *
    *     optimize(self)
    *         Runs the optimization process for the specified number of iterations and returns the best solution found.
    *********************************************************************/
"""


class APSO(PSO):
    def __init__(self, tsp_instance, population_size, max_iterations, w_min, w_max, c1, c2):
        super().__init__(tsp_instance, population_size, max_iterations, None, c1, c2)
        self.w_min = w_min
        self.w_max = w_max

    def update_velocity(self, particle, iteration):
        """
        Updates the velocity of a particle based on its personal best, the global best, and the adaptive inertia weight.

        Parameters:
            particle (Particle): The particle whose velocity needs to be updated.
            iteration (int): The current iteration of the optimization process.
        """
        # Calculate the adaptive inertia weight
        w = self.w_max - (self.w_max - self.w_min) * \
            (iteration / self.max_iterations)
        for i in range(len(particle.velocity)):
            r1 = random.random()
            r2 = random.random()
            cognitive_velocity = self.c1 * r1 * \
                (particle.pbest_solution[i] - particle.solution[i])
            social_velocity = self.c2 * r2 * \
                (self.gbest_solution[i] - particle.solution[i])
            particle.velocity[i] = w * particle.velocity[i] + \
                cognitive_velocity + social_velocity

    def optimize(self):
        """
        Runs the optimization process for the specified number of iterations and returns the best solution found.

        Returns:
            A tuple containing the best solution, its fitness, the convergence data, and the runtime.
        """
        start_time = time.time()
        swarm = [self.create_particle() for _ in range(self.population_size)]

        # Initialize gbest_solution and gbest_fitness
        self.gbest_solution = swarm[0].solution
        self.gbest_fitness = swarm[0].fitness
        for particle in swarm[1:]:
            if particle.fitness < self.gbest_fitness:
                self.gbest_solution = particle.solution
                self.gbest_fitness = particle.fitness
        convergence_data = []
        for iteration in range(self.max_iterations):
            for particle in swarm:
                self.update_velocity(particle, iteration)
                self.update_position(particle)

                # Update gbest_solution and gbest_fitness
                if particle.fitness < self.gbest_fitness:
                    self.gbest_solution = particle.solution
                    self.gbest_fitness = particle.fitness
            convergence_data.append(self.gbest_fitness)
        end_time = time.time()
        runtime = end_time - start_time
        return self.gbest_solution, self.gbest_fitness, convergence_data, runtime


""" /*************************************************************************
    * DiscretePSO class for solving TSP using Discrete Particle Swarm Optimization.
    *
    * This class extends the PSO class and implements the Discrete PSO algorithm for solving the
    * Traveling Salesman Problem (TSP). It adapts the continuous PSO algorithm to work with discrete
    * solutions by using a probabilistic approach for updating particle positions.
    *
    * Member Functions (Mutators):
    *     create_particle(self)
    *         Creates a new particle by generating a random solution and calculating its fitness.
    *
    *     update_velocity(self, particle)
    *         Updates the velocity of a particle based on its personal best and the global best.
    *
    *     update_position(self, particle)
    *         Updates the position of a particle using a probabilistic approach based on the velocity.
    ************************************************************************/
"""


class DiscretePSO(PSO):
    def create_particle(self):
        """
        Creates a new particle by generating a random solution and calculating its fitness.

        Returns:
            A new particle with a random solution and its corresponding fitness.
        """
        solution = random.sample(
            range(1, len(self.tsp_instance) + 1), len(self.tsp_instance))
        fitness = self.calculate_fitness(solution, self.tsp_instance)
        return BasePSO.Particle(solution, fitness)

    def update_velocity(self, particle):
        """
        Updates the velocity of a particle based on its personal best and the global best.

        Parameters:
            particle (Particle): The particle whose velocity needs to be updated.
        """
        for i in range(len(particle.velocity)):
            r1 = random.random()
            r2 = random.random()
            cognitive_velocity = self.c1 * r1 * \
                (particle.pbest_solution[i] - particle.solution[i])
            social_velocity = self.c2 * r2 * \
                (self.gbest_solution[i] - particle.solution[i])
            particle.velocity[i] = math.ceil(
                self.w * particle.velocity[i] + cognitive_velocity + social_velocity)

    def update_position(self, particle):
        """
        Updates the position of a particle using a probabilistic approach based on the velocity.

        Parameters:
            particle (Particle): The particle whose position needs to be updated.
        """
        new_solution = particle.solution[:]
        for i in range(len(new_solution)):
            if random.random() < 1 / (1 + math.exp(-abs(particle.velocity[i]))):
                j = random.randint(0, len(new_solution) - 1)
                new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
        new_fitness = self.calculate_fitness(new_solution, self.tsp_instance)
        if new_fitness < particle.fitness:
            particle.solution = new_solution
            particle.fitness = new_fitness
            if new_fitness < particle.pbest_fitness:
                particle.pbest_solution = new_solution
                particle.pbest_fitness = new_fitness


""" /*************************************************************************
    * SpatialPSO class for solving TSP using Spatial Particle Swarm Optimization.
    *
    * This class extends the BasePSO class and implements the Spatial PSO algorithm for solving the
    * Traveling Salesman Problem (TSP). It introduces a neighborhood concept where each particle interacts
    * with a subset of the swarm based on spatial proximity.
    *
    * Member Variables:
    *     neighborhood_size (int): The size of the neighborhood for each particle.
    *
    * Member Functions (Mutators):
    *     __init__(self, tsp_instance, population_size, max_iterations, w, c1, c2, neighborhood_size)
    *         Initializes the SpatialPSO class with the given parameters.
    *
    *     create_particle(self)
    *         Creates a new particle by generating a random solution and calculating its fitness.
    *
    *     calculate_fitness(self, solution)
    *         Calculates the fitness (total distance) of a given solution.
    *
    *     update_position(self, particle)
    *         Updates the position of a particle based on its velocity and applies a swap operation for exploration.
    *
    *     sigmoid(self, x)
    *         Calculates the sigmoid function value for a given input.
    *
    *     optimize(self)
    *         Runs the optimization process for the specified number of iterations and returns the best solution found.
    *
    *     find_neighbors(self, particle, swarm)
    *         Finds the neighboring particles for a given particle based on Euclidean distance.
    *
    *     euclidean_distance(self, solution1, solution2)
    *         Calculates the Euclidean distance between two solutions.
    ************************************************************************/
"""


class SpatialPSO(BasePSO):
    def __init__(self, tsp_instance, population_size, max_iterations, w, c1, c2, neighborhood_size):
        super().__init__(tsp_instance, population_size, max_iterations, w, c1, c2)
        self.neighborhood_size = neighborhood_size

    def create_particle(self):
        """
        Creates a new particle by generating a random solution and calculating its fitness.

        Returns:
            A new particle with a random solution and its corresponding fitness.
        """
        solution = random.sample(
            range(len(self.tsp_instance)), len(self.tsp_instance))
        fitness = self.calculate_fitness(solution)
        return self.Particle(solution, fitness)

    def calculate_fitness(self, solution):
        """
        Calculates the fitness (total distance) of a given solution.

        Parameters:
            solution (list): A list representing the order of cities in the solution.

        Returns:
            The total distance of the solution.
        """
        total_distance = 0
        for i in range(len(solution)):
            city1 = solution[i]
            city2 = solution[(i + 1) % len(solution)]
            total_distance += self.tsp_instance[city1][city2]
        return total_distance

    def update_position(self, particle):
        """
        Updates the position of a particle based on its velocity and applies a swap operation for exploration.

        Parameters:
            particle (Particle): The particle whose position needs to be updated.
        """
        new_solution = particle.solution[:]
        pbest_indices = {city: i for i,
                         city in enumerate(particle.pbest_solution)}
        swap_prob = max(
            0.3 * (1 - (self.current_iteration / self.max_iterations)), 0.1)

        for i in range(len(particle.solution)):
            if random.random() < self.calculate_tanh(particle.velocity[i]):
                # Swap the current city with the city in pbest
                j = pbest_indices[particle.solution[i]]
                new_solution[i], new_solution[j] = new_solution[j], new_solution[i]

            # Introduce a swap operation to explore the search space
            if random.random() < swap_prob:
                j = random.randint(0, len(new_solution) - 1)
                new_solution[i], new_solution[j] = new_solution[j], new_solution[i]

        particle.solution = new_solution
        particle.fitness = self.calculate_fitness(particle.solution)

        if particle.fitness < particle.pbest_fitness:
            particle.pbest_solution = particle.solution
            particle.pbest_fitness = particle.fitness

    def calculate_tanh(self, x):
        """
        Calculates the tanh function value for a given input.

        Parameters:
            x (float): The input value.

        Returns:
            The tanh function value.
        """
        return (1 + tanh(x)) / 4

    def optimize(self):
        """
        Runs the optimization process for the specified number of iterations and returns the best solution found.

        Returns:
            A tuple containing the best solution, its fitness, the convergence data, and the runtime.
        """
        start_time = time.time()
        swarm = [self.create_particle() for _ in range(self.population_size)]

        for particle in swarm:
            particle.neighbors = self.find_neighbors(particle, swarm)

        self.gbest_solution = min(swarm, key=lambda p: p.fitness).solution
        self.gbest_fitness = min(swarm, key=lambda p: p.fitness).fitness

        convergence_data = []
        for self.current_iteration in range(self.max_iterations):
            for particle in swarm:
                self.update_velocity(particle)
                self.update_position(particle)

                if particle.fitness < self.gbest_fitness:
                    self.gbest_solution = particle.solution
                    self.gbest_fitness = particle.fitness

            convergence_data.append(self.gbest_fitness)
        end_time = time.time()
        runtime = end_time - start_time
        return self.gbest_solution, self.gbest_fitness, convergence_data, runtime

    def find_neighbors(self, particle, swarm):
        """
        Finds the neighboring particles for a given particle based on Euclidean distance.

        Parameters:
            particle (Particle): The particle for which neighbors need to be found.
            swarm (list): The list of particles in the swarm.

        Returns:
            A list of neighboring particles.
        """
        distances = [(other_particle, self.euclidean_distance(particle.solution, other_particle.solution))
                     for other_particle in swarm if other_particle != particle]
        distances.sort(key=lambda x: x[1])
        return [p for p, _ in distances[:self.neighborhood_size]]

    def euclidean_distance(self, solution1, solution2):
        """
        Calculates the Euclidean distance between two solutions.

        Parameters:
            solution1 (list): The first solution.
            solution2 (list): The second solution.

        Returns:
            The Euclidean distance between the two solutions.
        """
        return sqrt(sum((coord1 - coord2) ** 2 for coord1, coord2 in zip(solution1, solution2)))


"""
    /*****************************************************************
    * DEPSO class for solving TSP using Differential Evolution Particle Swarm Optimization.
    *
    *
    * This class extends the BasePSO class and implements the Differential Evolution PSO (DEPSO) algorithm
    * for solving the Traveling Salesman Problem (TSP). It incorporates the crossover and mutation operators
    * from Differential Evolution into the PSO algorithm.
    *
    * Member Variables:
    *     cr (float): The crossover probability.
    *     f (float): The scaling factor for the mutation operator.
    *
    * Member Functions (Mutators):
    *     __init__(self, tsp_instance, population_size, max_iterations, w, c1, c2, cr, f)
    *         Initializes the DEPSO class with the given parameters.
    *
    *     create_particle(self)
    *         Creates a new particle by generating a random solution and calculating its fitness.
    *
    *     calculate_fitness(self, solution)
    *         Calculates the fitness (total distance) of a given solution.
    *
    *     update_position(self, particle)
    *         Updates the position of a particle using the Differential Evolution crossover and mutation operators.
    *
    *     optimize(self)
    *         Runs the optimization process for the specified number of iterations and returns the best solution found.
    *****************************************************************/
"""


class DEPSO(BasePSO):
    def __init__(self, tsp_instance, population_size, max_iterations, w, c1, c2, cr, f):
        super().__init__(tsp_instance, population_size, max_iterations, w, c1, c2)
        self.cr = cr
        self.f = f

    def create_particle(self):
        """
        Creates a new particle by generating a random solution and calculating its fitness.

        Returns:
            A new particle with a random solution and its corresponding fitness.
        """
        solution = random.sample(
            range(len(self.tsp_instance)), len(self.tsp_instance))
        fitness = self.calculate_fitness(solution)
        return self.Particle(solution, fitness)

    def calculate_fitness(self, solution):
        """
        Calculates the fitness (total distance) of a given solution.

        Parameters:
            solution (list): A list representing the order of cities in the solution.

        Returns:
            The total distance of the solution.
        """
        total_distance = 0
        for i in range(len(solution)):
            city1 = solution[i]
            city2 = solution[(i + 1) % len(solution)]
            total_distance += self.tsp_instance[city1][city2]
        return total_distance

    def update_position(self, particle):
        """
        Updates the position of a particle using the Differential Evolution crossover and mutation operators.

        Parameters:
            particle (Particle): The particle whose position needs to be updated.
        """
        new_solution = particle.solution[:]

        # Select three random particles from the swarm
        r1, r2, r3 = random.sample(range(self.population_size), 3)

        for j in range(len(particle.solution)):
            if random.random() < self.cr or j == random.randint(0, len(particle.solution) - 1):
                # Apply the DE crossover operator
                new_city_index = int(
                    self.swarm[r1].solution[j] + self.f * (self.swarm[r2].solution[j] - self.swarm[r3].solution[j]))
                new_city_index = new_city_index % len(particle.solution)

                # Find the city at the new index
                new_city = particle.solution[new_city_index]

                # Find the current position of the new city in the solution
                current_city_index = new_solution.index(new_city)

                # Swap the cities to maintain the TSP rules
                new_solution[j], new_solution[current_city_index] = new_solution[current_city_index], new_solution[j]

        new_fitness = self.calculate_fitness(new_solution)
        if new_fitness < particle.fitness:
            particle.solution = new_solution
            particle.fitness = new_fitness

            if particle.fitness < particle.pbest_fitness:
                particle.pbest_solution = particle.solution
                particle.pbest_fitness = particle.fitness

    def optimize(self):
        """
        Runs the optimization process for the specified number of iterations and returns the best solution found.

        Returns:
            A tuple containing the best solution, its fitness, the convergence data, and the runtime.
        """
        start_time = time.time()
        self.swarm = [self.create_particle()
                      for _ in range(self.population_size)]

        # Initialize gbest_solution and gbest_fitness
        self.gbest_solution = self.swarm[0].solution
        self.gbest_fitness = self.swarm[0].fitness

        for particle in self.swarm[1:]:
            if particle.fitness < self.gbest_fitness:
                self.gbest_solution = particle.solution
                self.gbest_fitness = particle.fitness

        convergence_data = []
        for iteration in range(self.max_iterations):
            for particle in self.swarm:
                self.update_velocity(particle)
                self.update_position(particle)

                # Update gbest_solution and gbest_fitness
                if particle.fitness < self.gbest_fitness:
                    self.gbest_solution = particle.solution
                    self.gbest_fitness = particle.fitness

            convergence_data.append(self.gbest_fitness)

        end_time = time.time()
        runtime = end_time - start_time

        return self.gbest_solution, self.gbest_fitness, convergence_data, runtime


"""
/*********************************************************************
* PredatorPreyPSO class for solving TSP using Predator-Prey Particle Swarm Optimization.
*
*
* This class extends the BasePSO class and implements the Predator-Prey PSO algorithm for solving the
* Traveling Salesman Problem (TSP). It introduces a predator particle that influences the behavior of
* the swarm particles.
*
* Member Variables:
*    fear_factor (float): The fear factor that determines the influence of the predator on the swarm particles.
*    predator_velocity (list): The velocity of the predator particle.
*
* Member Functions (Mutators):
*    __init__(self, tsp_instance, population_size, max_iterations, w, c1, c2, fear_factor)
*        Initializes the PredatorPreyPSO class with the given parameters.
*
*    create_particle(self)
*        Creates a new particle by generating a random solution and calculating its fitness.
*
*    calculate_fitness(self, solution)
*        Calculates the fitness (total distance) of a given solution.
*
*    update_velocity(self, particle, predator)
*        Updates the velocity of a particle based on its personal best, global best, and the predator's influence.
*
*    update_position(self, particle)
*        Updates the position of a particle based on its velocity and applies a swap operation for exploration.
*
*    sigmoid(self, x)
*        Calculates the sigmoid function value for a given input.
*
*    optimize(self)
*        Runs the optimization process for the specified number of iterations and returns the best solution found.
    ******************************************************************/
"""


class PredatorPreyPSO(BasePSO):
    def __init__(self, tsp_instance, population_size, max_iterations, w, c1, c2, fear_factor):
        super().__init__(tsp_instance, population_size, max_iterations, w, c1, c2)
        self.fear_factor = fear_factor
        self.predator_velocity = [0] * len(tsp_instance)

    def create_particle(self):
        """
        Creates a new particle by generating a random solution and calculating its fitness.

        Returns:
            A new particle with a random solution and its corresponding fitness.
        """
        solution = random.sample(
            range(len(self.tsp_instance)), len(self.tsp_instance))
        fitness = self.calculate_fitness(solution)
        return self.Particle(solution, fitness)

    def calculate_fitness(self, solution):
        """
        Calculates the fitness (total distance) of a given solution.

        Parameters:
            solution (list): A list representing the order of cities in the solution.

        Returns:
            The total distance of the solution.
        """
        total_distance = 0
        for i in range(len(solution)):
            city1 = solution[i]
            city2 = solution[(i + 1) % len(solution)]
            total_distance += self.tsp_instance[city1][city2]
        return total_distance

    def update_velocity(self, particle, predator):
        """
        Updates the velocity of a particle based on its personal best, global best, and the predator's influence.

        Parameters:
            particle (Particle): The particle whose velocity needs to be updated.
            predator (Particle): The predator particle.
        """
        for i in range(len(particle.velocity)):
            r1 = random.random()
            r2 = random.random()
            cognitive_velocity = self.c1 * r1 * \
                (particle.pbest_solution[i] - particle.solution[i])
            social_velocity = self.c2 * r2 * \
                (self.gbest_solution[i] - particle.solution[i])
            predator_velocity = self.fear_factor * \
                (particle.solution[i] - predator.solution[i])
            particle.velocity[i] = self.w * particle.velocity[i] + \
                cognitive_velocity + social_velocity + predator_velocity

        self.predator_velocity = [
            self.fear_factor * (particle.solution[i] - predator.solution[i]) for i in range(len(particle.solution))]

    def update_position(self, particle):
        """
        Updates the position of a particle based on its velocity and applies a swap operation for exploration.

        Parameters:
            particle (Particle): The particle whose position needs to be updated.
        """
        new_solution = particle.solution[:]
        pbest_indices = {city: i for i,
                         city in enumerate(particle.pbest_solution)}
        swap_prob = max(
            0.3 * (1 - (self.current_iteration / self.max_iterations)), 0.1)

        for i in range(len(particle.solution)):
            if random.random() < self.calculate_tanh(particle.velocity[i]):
                # Swap the current city with the city in pbest
                j = pbest_indices[particle.solution[i]]
                new_solution[i], new_solution[j] = new_solution[j], new_solution[i]

            # Introduce a swap operation to explore the search space
            if random.random() < swap_prob:
                j = random.randint(0, len(new_solution) - 1)
                new_solution[i], new_solution[j] = new_solution[j], new_solution[i]

        particle.solution = new_solution
        particle.fitness = self.calculate_fitness(particle.solution)

        if particle.fitness < particle.pbest_fitness:
            particle.pbest_solution = particle.solution
            particle.pbest_fitness = particle.fitness

    def calculate_tanh(self, x):
        """
        Calculates the tanh function value for a given input.

        Parameters:
            x (float): The input value.

        Returns:
            The tanh function value.
        """
        return (1 + tanh(x)) / 2

    def optimize(self):
        """
        Runs the optimization process for the specified number of iterations and returns the best solution found.

        Returns:
            A tuple containing the best solution, its fitness, the convergence data, and the runtime.
        """
        start_time = time.time()
        self.swarm = [self.create_particle()
                      for _ in range(self.population_size)]
        predator = self.create_particle()
        self.gbest_solution = min(self.swarm, key=lambda p: p.fitness).solution
        self.gbest_fitness = min(self.swarm, key=lambda p: p.fitness).fitness
        convergence_data = []

        for self.current_iteration in range(self.max_iterations):
            for particle in self.swarm:
                self.update_velocity(particle, predator)
                self.update_position(particle)
                if particle.fitness < self.gbest_fitness:
                    self.gbest_solution = particle.solution
                    self.gbest_fitness = particle.fitness
            convergence_data.append(self.gbest_fitness)
            predator.solution = self.gbest_solution
            predator.velocity = self.predator_velocity

        end_time = time.time()
        runtime = end_time - start_time
        return self.gbest_solution, self.gbest_fitness, convergence_data, runtime


'''
/***************************************************************************************
*                                                                                     
*    PrettyPlotting class for creating visualizations of TSP optimization results.    
*                                                                                                                                                           
*    This class provides various methods to create plots and visualizations for       
*    analyzing the results of TSP optimization experiments. It includes methods for   
*    convergence plots, performance comparison heatmaps, tour length vs. runtime      
*    plots, and line plots for different hyperparameters.                             
*                                                                                     
*    Member Functions:                                                                
*    __init__(self)                                                              
*        Initializes the PrettyPlotting class.                                   
*                                                                                     
*    convergence_plot(self, data, algorithms, problem_instance)                  
*        Creates a convergence plot for the specified algorithms and problem     
*        instance.                                                                
*    plot_data(self, data, tsp_instance_name, iterations=100)                     
*        Plots the convergence data for all algorithms.                           
*                                                                                     
*    box_plot_tour_lengths(self, data, algorithms, tsp_instance_name)            
*        Creates a box plot of the best tour lengths for each algorithm.          
*                                                                                     
*    performance_heatmap(self, data, algorithms, instances=100)                  
*        Creates a performance comparison matrix heatmap for the algorithms.      
*                                                                                     
*    tour_length_vs_runtime(self, data)                                          
*        Plots the tour length vs. runtime for each run.                          
*                                                                                     
*    line_plot_population_size(self, data, tsp_instance_name)                    
*        Creates a line plot of the best tour length vs. population size for     
*        each algorithm.                                                          
*                                                                                     
*    plot_runtime_population_size(self, data, tsp_instance_name)                 
*        Plots the average runtime vs. population size for each algorithm.       
*                                                                                     
*    line_plot_inertia_weight(self, data, tsp_instance_name)                     
*        Creates a line plot of the best tour length vs. inertia weight for      
*        PSO algorithms.                                                          
*                                                                                     
*    line_plot_acceleration_coefficients(self, data, tsp_instance_name)          
*        Creates a line plot of the best tour length vs. acceleration coefficient
*        combinations for PSO algorithms.                                         
*                                                                                     
*    convergence_plot_random_hyperparameters(self, data, tsp_instance_name)      
*        Creates a convergence plot with random hyperparameters for all          
*        algorithms.                                                              
*                                                                                     
*    line_plot_neighborhood_size(self, data, tsp_instance_name)                   
*        Creates a line plot of the best tour length vs. neighborhood size for   
*        the SPSO algorithm.                                                      
*                                                                                     
*    line_plot_cr_f(self, data, tsp_instance_name)                               
*        Creates a line plot of the best tour length vs. CR and F combinations    
*        for the DEPSO algorithm.                                                 
*                                                                                     
*    line_plot_fear_factor(self, data, tsp_instance_name)                        
*        Creates a line plot of the best tour length vs. fear factor for the      
*        PPPSO algorithm.                                                         
*                                                                                     
*    convergence_plot_all_algorithms(self, data, tsp_instance_name)              
*        Creates a convergence plot for all algorithms.                           
*                                                                                     
*    line_plot_w_min_w_max(self, data, tsp_instance_name)                        
*        Creates a line plot of the best tour length vs. w_min and w_max          
*        combinations for the APSO algorithm.                                     
*                                                                                 
*    line_plot_max_iterations(self, data, tsp_instance_name)                     
*        Creates a line plot of the best tour length vs. maximum iterations for  
*        each algorithm.                                                          
*                                                                                     
***************************************************************************************/

'''


class PrettyPlotting:
    def __init__(self):
        """
        Initializes the PrettyPlotting class.

        Sets the style and font scale for the plots using seaborn.
        """
        sns.set_theme(style='ticks', palette='muted', font_scale=1)

    def convergence_plot(self, data, algorithms, problem_instance):
        """
        Creates a convergence plot for the specified algorithms and problem instance.

        Parameters:
            data (dict): Dictionary containing the convergence data for each algorithm.
            algorithms (list): List of algorithm names.
            problem_instance (str): Name of the problem instance.

        Postconditions:
            Displays the convergence plot.
        """
        plt.figure(figsize=(8, 6))
        for algorithm in algorithms:
            for run_data in data[algorithm]:
                iterations = range(1, len(run_data['convergence_data']) + 1)
                best_tour_lengths = run_data['convergence_data']
                plt.plot(iterations, best_tour_lengths,
                         label=f"{algorithm} - Run {data[algorithm].index(run_data) + 1}")
        plt.xlabel('Number of Iterations')
        plt.ylabel('Best Tour Length')
        plt.title('Convergence Plot - Problem Instance: ')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_data(self, data, tsp_instance_name, iterations=100):
        """
        Plots the convergence data for all algorithms.

        Parameters:
            data (dict): Dictionary containing the convergence data for each algorithm.
            tsp_instance_name (str): Name of the TSP instance.
            iterations (int): Number of iterations to plot. Default is 100.

        Postconditions:
            Displays the plot of convergence data for all algorithms.
        """
        count = iterations
        plt.figure(figsize=(10, 10))

        # Define a dictionary to map algorithms to colors
        color_map = {
            'PSO': 'blue',
            'APSO': 'green',
            'HPSO': 'red',
            'Random Sampling': 'purple',
            'Stochastic Hill Climber': 'orange',
            'spso': 'cyan',
            'depso': 'magenta',
            'pppso': 'black'
        }

        for algorithm in data.keys():
            # Get the color for the current algorithm
            color = color_map[algorithm]
            for run_data in data[algorithm]:
                iterations = np.linspace(1, count, count)
                plt.plot(
                    iterations, run_data['convergence_data'], color=color, label=f'{algorithm}')

        # Remove duplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(),
                   loc='upper right', fontsize='small')

        plt.xlabel('Iterations')
        plt.ylabel('Fitness')
        plt.suptitle(f'TSP Instance: {tsp_instance_name}', fontsize=14)
        # Adjust the plot layout to accommodate the suptitle
        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))

        plt.show()

    def box_plot_tour_lengths(self, data, algorithms, tsp_instance_name):
        """
        Creates a box plot of the best tour lengths for each algorithm.

        Parameters:
            data (dict): Dictionary containing the best tour lengths for each algorithm.
            algorithms (list): List of algorithm names.
            tsp_instance_name (str): Name of the TSP instance.

        Postconditions:
            Displays the box plot of best tour lengths for each algorithm.
        """
        plt.figure(figsize=(10, 6))
        tour_lengths = []
        for algorithm in algorithms:
            algorithm_data = data[algorithm]
            if algorithm == 'HPSO':
                tour_lengths.append([run['best_fitness']
                                    for run in algorithm_data])
            else:
                # Assuming the same data structure for other algorithms
                tour_lengths.append([run['best_fitness']
                                    for run in algorithm_data])

        # Create the boxplot
        bp = plt.boxplot(tour_lengths, labels=algorithms)

        # Rotate x-labels by 45 degrees
        plt.xticks(rotation=45, ha='right')

        # Adjust the bottom margin to prevent overlapping
        plt.subplots_adjust(bottom=0.2)

        plt.ylabel('Best Tour Length')
        plt.title('Distribution of Best Tour Lengths')
        plt.suptitle(f'TSP Instance: {tsp_instance_name}', fontsize=14)
        # Adjust the plot layout to accommodate the suptitle
        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))

    def performance_heatmap(self, data, algorithms, instances=100):
        """
        Creates a performance comparison matrix heatmap for the algorithms.

        Parameters:
            data (dict): Dictionary containing the best tour lengths for each algorithm.
            algorithms (list): List of algorithm names.
            instances (int): Number of problem instances. Default is 100.

        Postconditions:
            Displays the performance comparison matrix heatmap.
        """
        # Create a dictionary to store performance ranks
        performance_ranks = {}

        # Iterate over instances
        for instance in instances:
            # Sort algorithms by best tour length for this instance
            sorted_algos = sorted(algorithms, key=lambda algo: min(
                data[algo][instance]['best_fitness']))

            # Assign ranks based on the sorted order
            for rank, algo in enumerate(sorted_algos, start=1):
                performance_ranks.setdefault(algo, {})[instance] = rank

        # Create a DataFrame from the performance_ranks dictionary
        df = pd.DataFrame.from_dict(performance_ranks, orient='index')

        # Create the heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(df, annot=True, cmap='YlGnBu')
        plt.xlabel('Problem Instances')
        plt.ylabel('Algorithms')
        plt.title('Performance Comparison Matrix')

    def tour_length_vs_runtime(self, data):
        """
        Plots the tour length vs. runtime for each run.

        Parameters:
            data (dict): Dictionary containing the best tour lengths and runtimes for each algorithm.

        Postconditions:
            Displays the plot of tour length vs. runtime.
        """
        plt.figure(figsize=(10, 6))

        best_fitnesses = []
        runtimes = []
        for algo_data in data.values():
            for run_data in algo_data:
                best_fitnesses.append(run_data['best_fitness'])
                runtimes.append(run_data['runtime'])

        # Plot a point for each run
        plt.scatter(runtimes, best_fitnesses, label='cities', alpha=0.5)

        plt.xlabel('Runtime')
        plt.ylabel('Best Tour Length')
        plt.title('Tour Length vs. Runtime')
        plt.legend()

    def line_plot_population_size(self, data, tsp_instance_name):
        """
        Creates a line plot of the best tour length vs. population size for each algorithm.

        Parameters:
            data (dict): Dictionary containing the best tour lengths for each population size and algorithm.
            tsp_instance_name (str): Name of the TSP instance.

        Postconditions:
            Displays the line plot of best tour length vs. population size for each algorithm.
        """
        plt.figure(figsize=(10, 6))
        population_sizes = []
        pso_best_lengths = []
        hpso_best_lengths = []
        apso_best_lengths = []
        random_sampling_best_lengths = []
        stochastic_hill_climber_best_lengths = []
        spso_best_lengths = []
        depso_best_lengths = []
        pppso_best_lengths = []

        for pop_size, runs in data.items():
            population_sizes.append(pop_size)
            pso_best_lengths.append(
                min(run['best_fitness'] for run in runs['PSO']))
            hpso_best_lengths.append(
                min(run['best_fitness'] for run in runs['HPSO']))
            apso_best_lengths.append(
                min(run['best_fitness'] for run in runs['APSO']))
            random_sampling_best_lengths.append(
                min(run['best_fitness'] for run in runs['Random Sampling']))
            stochastic_hill_climber_best_lengths.append(
                min(run['best_fitness'] for run in runs['Stochastic Hill Climber']))
            spso_best_lengths.append(
                min(run['best_fitness'] for run in runs['spso']))
            depso_best_lengths.append(
                min(run['best_fitness'] for run in runs['depso']))
            pppso_best_lengths.append(
                min(run['best_fitness'] for run in runs['pppso']))

        plt.plot(population_sizes, pso_best_lengths, marker='o', label='PSO')
        plt.plot(population_sizes, hpso_best_lengths, marker='o', label='HPSO')
        plt.plot(population_sizes, apso_best_lengths, marker='o', label='APSO')
        plt.plot(population_sizes, random_sampling_best_lengths,
                 marker='o', label='Random Sampling')
        plt.plot(population_sizes, stochastic_hill_climber_best_lengths,
                 marker='o', label='Stochastic Hill Climber')
        plt.plot(population_sizes, spso_best_lengths, marker='o', label='SPSO')
        plt.plot(population_sizes, depso_best_lengths,
                 marker='o', label='DEPSO')
        plt.plot(population_sizes, pppso_best_lengths,
                 marker='o', label='PPPSO')

        plt.xlabel('Population Size')
        plt.ylabel('Best Tour Length')
        plt.title('Impact of Population Size on Best Tour Length')
        plt.legend()
        plt.suptitle(f'TSP Instance: {tsp_instance_name}', fontsize=14)
        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
        plt.show()

    def plot_runtime_population_size(self, data, tsp_instance_name):
        """
        Plots the average runtime vs. population size for each algorithm.

        Parameters:
            data (dict): Dictionary containing the runtimes for each population size and algorithm.
            tsp_instance_name (str): Name of the TSP instance.

        Postconditions:
            Displays the plot of average runtime vs. population size for each algorithm.
        """
        plt.figure(figsize=(10, 6))

        population_sizes = list(data.keys())
        algorithms = ['PSO', 'HPSO', 'APSO', 'spso', 'depso',
                      'pppso', 'Random Sampling', 'Stochastic Hill Climber']
        runtimes = {algo: [] for algo in algorithms}

        for pop_size in population_sizes:
            runs = data[pop_size]
            for algo in algorithms:
                if algo in runs:
                    avg_runtime = sum(run['runtime']
                                      for run in runs[algo]) / len(runs[algo])
                    runtimes[algo].append(avg_runtime)
                else:
                    # If the algorithm is not present, append 0 as runtime
                    runtimes[algo].append(0)

        for algo, runtime_values in runtimes.items():
            plt.plot(population_sizes, runtime_values, marker='o', label=algo)

        plt.xlabel('Population Size')
        plt.ylabel('Average Runtime (seconds)')
        plt.title('Runtime vs. Population Size')
        plt.legend()
        plt.grid(True)
        plt.suptitle(f'TSP Instance: {tsp_instance_name}', fontsize=14)
        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
        plt.show()

    def line_plot_inertia_weight(self, data, tsp_instance_name):
        """
        Creates a line plot of the best tour length vs. inertia weight for PSO algorithms.

        Parameters:
            data (dict): Dictionary containing the best tour lengths for each inertia weight and PSO algorithm.
            tsp_instance_name (str): Name of the TSP instance.

        Postconditions:
            Displays the line plot of best tour length vs. inertia weight for PSO algorithms.
        """
        plt.figure(figsize=(10, 6))
        inertia_weights = []
        pso_best_lengths = []
        hpso_best_lengths = []
        spso_best_lengths = []

        for inertia_weight, runs in data.items():
            inertia_weights.append(inertia_weight)
            pso_best_lengths.append(
                min(run['best_fitness'] for run in runs['PSO']))
            hpso_best_lengths.append(
                min(run['best_fitness'] for run in runs['HPSO']))
            spso_best_lengths.append(
                min(run['best_fitness'] for run in runs['spso']))

        plt.plot(inertia_weights, pso_best_lengths, marker='o', label='PSO')
        plt.plot(inertia_weights, hpso_best_lengths, marker='o', label='HPSO')
        plt.plot(inertia_weights, spso_best_lengths, marker='o', label='SPSO')
        plt.xlabel('Inertia Weight')
        plt.ylabel('Best Tour Length')
        plt.title('Impact of Inertia Weight on Best Tour Length (PSO)')
        plt.legend()
        plt.suptitle(f'TSP Instance: {tsp_instance_name}', fontsize=14)
        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
        plt.show()

    def line_plot_acceleration_coefficients(self, data, tsp_instance_name):
        """
        Creates a line plot of the best tour length vs. acceleration coefficient combinations for PSO algorithms.

        Parameters:
           data (dict): Dictionary containing the best tour lengths for each acceleration coefficient combination and PSO algorithm.
           tsp_instance_name (str): Name of the TSP instance.

        Postconditions:
           Displays the line plot of best tour length vs. acceleration coefficient combinations for PSO algorithms.
        """
        plt.figure(figsize=(10, 6))
        pso_best_lengths = []
        hpso_best_lengths = []
        apso_best_lengths = []
        spso_best_lengths = []
        depso_best_lengths = []
        pppso_best_lengths = []

        for coefficients, runs in data.items():
            c1, c2 = coefficients
            pso_best_lengths.append(
                min(run['best_fitness'] for run in runs['PSO']))
            hpso_best_lengths.append(
                min(run['best_fitness'] for run in runs['HPSO']))
            apso_best_lengths.append(
                min(run['best_fitness'] for run in runs['APSO']))
            spso_best_lengths.append(
                min(run['best_fitness'] for run in runs['spso']))
            depso_best_lengths.append(
                min(run['best_fitness'] for run in runs['depso']))
            pppso_best_lengths.append(
                min(run['best_fitness'] for run in runs['pppso']))

        plt.plot(range(len(data)), pso_best_lengths, marker='o', label='PSO')
        plt.plot(range(len(data)), hpso_best_lengths, marker='o', label='HPSO')
        plt.plot(range(len(data)), apso_best_lengths, marker='o', label='APSO')
        plt.plot(range(len(data)), spso_best_lengths, marker='o', label='SPSO')
        plt.plot(range(len(data)), depso_best_lengths,
                 marker='o', label='DEPSO')
        plt.plot(range(len(data)), pppso_best_lengths,
                 marker='o', label='PPPSO')
        plt.xlabel('Acceleration Coefficient Combination')
        plt.ylabel('Best Tour Length')
        plt.title('Impact of Acceleration Coefficients on Best Tour Length (PSO)')
        plt.xticks(range(len(data)), [f'c1={c1}, c2={c2}' 
                                      for c1, c2 in data.keys()], rotation=45)
        plt.legend()
        plt.suptitle(f'TSP Instance: {tsp_instance_name}', fontsize=14)
        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
        plt.show()

    def convergence_plot_random_hyperparameters(self, data, tsp_instance_name):
        """
        Creates a convergence plot with random hyperparameters for all algorithms.

        Parameters:
           data (dict): Dictionary containing the convergence data for each algorithm.
           tsp_instance_name (str): Name of the TSP instance.

        Postconditions:
           Displays the convergence plot with random hyperparameters for all algorithms.
        """
        plt.figure(figsize=(10, 6))
        iterations = range(1, len(data['PSO'][0]['convergence_data']) + 1)

        pso_convergence_data = [run['convergence_data'] for run in data['PSO']]
        pso_mean_convergence = np.mean(pso_convergence_data, axis=0)
        hpso_convergence_data = [run['convergence_data']
                                 for run in data['HPSO']]
        hpso_mean_convergence = np.mean(hpso_convergence_data, axis=0)
        apso_convergence_data = [run['convergence_data']
                                 for run in data['APSO']]
        apso_mean_convergence = np.mean(apso_convergence_data, axis=0)
        random_sampling_convergence_data = [
            run['convergence_data'] for run in data['Random Sampling']]
        random_sampling_mean_convergence = np.mean(
            random_sampling_convergence_data, axis=0)
        stochastic_hill_climber_convergence_data = [
            run['convergence_data'] for run in data['Stochastic Hill Climber']]
        stochastic_hill_climber_convergence_data = np.array(
            stochastic_hill_climber_convergence_data)
        stochastic_hill_climber_mean_convergence = np.mean(
            stochastic_hill_climber_convergence_data, axis=0)

        spso_convergence_data = [run['convergence_data']
                                 for run in data['spso']]
        spso_mean_convergence = np.mean(spso_convergence_data, axis=0)
        depso_convergence_data = [run['convergence_data']
                                  for run in data['depso']]
        depso_mean_convergence = np.mean(depso_convergence_data, axis=0)
        pppso_convergence_data = [run['convergence_data']
                                  for run in data['pppso']]
        pppso_mean_convergence = np.mean(pppso_convergence_data, axis=0)

        plt.plot(iterations, pso_mean_convergence, label='PSO')
        plt.plot(iterations, hpso_mean_convergence, label='HPSO')
        plt.plot(iterations, apso_mean_convergence, label='APSO')
        plt.plot(iterations, random_sampling_mean_convergence,
                 label='Random Sampling')
        plt.plot(iterations, stochastic_hill_climber_mean_convergence,
                 label='Stochastic Hill Climber')
        plt.plot(iterations, spso_mean_convergence, label='SPSO')
        plt.plot(iterations, depso_mean_convergence, label='DEPSO')
        plt.plot(iterations, pppso_mean_convergence, label='PPPSO')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Best Tour Length')
        plt.title('Convergence Plot')
        plt.legend()
        plt.suptitle(f'TSP Instance: {tsp_instance_name}', fontsize=14)
        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
        plt.show()

    def line_plot_neighborhood_size(self, data, tsp_instance_name):
        """
        Creates a line plot of the best tour length vs. neighborhood size for the SPSO algorithm.

        Parameters:
           data (dict): Dictionary containing the best tour lengths for each neighborhood size and the SPSO algorithm.
           tsp_instance_name (str): Name of the TSP instance.

        Postconditions:
           Displays the line plot of best tour length vs. neighborhood size for the SPSO algorithm.
        """
        plt.figure(figsize=(10, 6))
        neighborhood_sizes = []
        spso_best_lengths = []

        for neighborhood_size, runs in data.items():
            neighborhood_sizes.append(neighborhood_size)
            spso_best_lengths.append(
                min(run['best_fitness'] for run in runs['spso']))

        plt.plot(neighborhood_sizes, spso_best_lengths,
                 marker='o', label='SPSO')
        plt.xlabel('Neighborhood Size')
        plt.ylabel('Best Tour Length')
        plt.title('Impact of Neighborhood Size on Best Tour Length (SPSO)')
        plt.legend()
        plt.suptitle(f'TSP Instance: {tsp_instance_name}', fontsize=14)
        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
        plt.show()

    def line_plot_cr_f(self, data, tsp_instance_name):
        """
        Creates a line plot of the best tour length vs. CR and F combinations for the DEPSO algorithm.

        Parameters:
           data (dict): Dictionary containing the best tour lengths for each CR and F combination and the DEPSO algorithm.
           tsp_instance_name (str): Name of the TSP instance.

        Postconditions:
           Displays the line plot of best tour length vs. CR and F combinations for the DEPSO algorithm.
        """
        plt.figure(figsize=(10, 6))
        depso_best_lengths = []

        for (cr, f), runs in data.items():
            depso_best_lengths.append(
                min(run['best_fitness'] for run in runs['depso']))

        plt.plot(range(len(data)), depso_best_lengths,
                 marker='o', label='DEPSO')
        plt.xlabel('CR and F Combination')
        plt.ylabel('Best Tour Length')
        plt.title('Impact of CR and F on Best Tour Length (DEPSO)')
        plt.xticks(range(len(data)), [f'CR={cr}, F={f}'
                                       for cr, f in data.keys()], rotation=45)
        plt.legend()
        plt.suptitle(f'TSP Instance: {tsp_instance_name}', fontsize=14)
        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
        plt.show()

    def line_plot_fear_factor(self, data, tsp_instance_name):
        """
        Creates a line plot of the best tour length vs. fear factor for the PPPSO algorithm.

        Parameters:
           data (dict): Dictionary containing the best tour lengths for each fear factor and the PPPSO algorithm.
           tsp_instance_name (str): Name of the TSP instance.

        Postconditions:
           Displays the line plot of best tour length vs. fear factor for the PPPSO algorithm.
        """
        plt.figure(figsize=(10, 6))
        fear_factors = []
        pppso_best_lengths = []

        for fear_factor, runs in data.items():
            fear_factors.append(fear_factor)
            pppso_best_lengths.append(
                min(run['best_fitness'] for run in runs['pppso']))

        plt.plot(fear_factors, pppso_best_lengths, marker='o', label='PPPSO')
        plt.xlabel('Fear Factor')
        plt.ylabel('Best Tour Length')
        plt.title('Impact of Fear Factor on Best Tour Length (PPPSO)')
        plt.legend()
        plt.suptitle(f'TSP Instance: {tsp_instance_name}', fontsize=14)
        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
        plt.show()

    def convergence_plot_all_algorithms(self, data, tsp_instance_name):
        """
        Creates a convergence plot for all algorithms.

        Parameters:
           data (dict): Dictionary containing the convergence data for each algorithm.
           tsp_instance_name (str): Name of the TSP instance.

        Postconditions:
           Displays the convergence plot for all algorithms.
        """
        plt.figure(figsize=(10, 6))
        iterations = range(1, len(data['PSO'][0]['convergence_data']) + 1)

        for algorithm in data:
            convergence_data = [run['convergence_data']
                                for run in data[algorithm]]
            mean_convergence = np.mean(convergence_data, axis=0)
            plt.plot(iterations, mean_convergence, label=algorithm)

        plt.xlabel('Number of Iterations')
        plt.ylabel('Best Tour Length')
        plt.title('Convergence Plot - All Algorithms')
        plt.legend()
        plt.suptitle(f'TSP Instance: {tsp_instance_name}', fontsize=14)
        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
        plt.show()

    def line_plot_w_min_w_max(self, data, tsp_instance_name):
        """
        Creates a line plot of the best tour length vs. w_min and w_max combinations for the APSO algorithm.

        Parameters:
           data (dict): Dictionary containing the best tour lengths for each w_min and w_max combination and the APSO algorithm.
           tsp_instance_name (str): Name of the TSP instance.

        Postconditions:
           Displays the line plot of best tour length vs. w_min and w_max combinations for the APSO algorithm.
        """
        plt.figure(figsize=(10, 6))
        apso_best_lengths = []

        for (w_min, w_max), runs in data.items():
            apso_best_lengths.append(
                min(run['best_fitness'] for run in runs['APSO']))

        x_ticks_labels = ['w_min={}, w_max={}'.format(w_min, w_max) for w_min, w_max in data.keys()]
        plt.plot(range(len(data)), apso_best_lengths, marker='o', label='APSO')
        plt.xlabel('w_min and w_max Combination')
        plt.ylabel('Best Tour Length')
        plt.title('Impact of w_min and w_max on Best Tour Length (APSO)')
        plt.xticks(range(len(data)), x_ticks_labels, rotation=45)
        plt.legend()
        plt.suptitle(f'TSP Instance: {tsp_instance_name}', fontsize=14)
        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
        plt.show()

    def line_plot_max_iterations(self, data, tsp_instance_name):
        """
        Creates a line plot of the best tour length vs. maximum iterations for each algorithm.

        Parameters:
           data (dict): Dictionary containing the best tour lengths for each maximum iteration value and algorithm.
           tsp_instance_name (str): Name of the TSP instance.

        Postconditions:
           Displays the line plot of best tour length vs. maximum iterations for each algorithm.
        """
        plt.figure(figsize=(10, 6))
        max_iterations = list(data.keys())
        algorithms = data[max_iterations[0]].keys()

        for algo in algorithms:
            best_lengths = []
            for iterations in max_iterations:
                runs = data[iterations][algo]
                best_lengths.append(min(run['best_fitness'] for run in runs))

            plt.plot(max_iterations, best_lengths, marker='o', label=algo)

        plt.xlabel('Maximum Iterations')
        plt.ylabel('Best Tour Length')
        plt.title('Impact of Maximum Iterations on Best Tour Length')
        plt.legend()
        plt.suptitle(f'TSP Instance: {tsp_instance_name}', fontsize=14)
        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
        plt.show()


"""
/***************************************************************************************
 * Experiment class
 *
 * Represents an experiment for running and evaluating different PSO algorithms on a TSP instance.
 *
 * Member Variables:
 * - tsp_instance (list): 2D list representing the TSP instance.
 * - tsp_instance_name (str): Name of the TSP instance.
 * - num_runs (int): Number of runs for each algorithm.
 * - population_size (int): Size of the swarm population.
 * - max_iterations (int): Maximum number of iterations for the optimization.
 * - w (float): Inertia weight.
 * - c1 (float): Cognitive coefficient.
 * - c2 (float): Social coefficient.
 * - w_min (float): Minimum inertia weight for APSO.
 * - w_max (float): Maximum inertia weight for APSO.
 * - max_stagnation (int): Maximum number of stagnation iterations for Hill Climber.
 * - neighborhood_size (int): Size of the neighborhood for SPSO.
 * - cr (float): Crossover probability for DEPSO.
 * - f (float): Mutation factor for DEPSO.
 * - fear_factor (float): Fear factor for PPPSO.
 * - algorithms (list): List of algorithm names to run in the experiment.
 * - data (dict): Dictionary to store the results of each algorithm.
 * - plotter (PrettyPlotting): Object for plotting the results.
 *
 * Member Functions:
 * - run(algorithms=None): Runs the specified algorithms (or all algorithms if None) on the TSP instance.
 * - run_algorithm(algorithm): Runs a single algorithm on the TSP instance for multiple runs.
 * - _run_single_run(algorithm): Runs a single run of an algorithm on the TSP instance.
 * - run_population_size_experiments(population_sizes): Runs experiments with different population sizes.
 * - run_population_size_experiment(pop_size): Runs an experiment with a specific population size.
 * - run_inertia_weight_experiments(inertia_weights): Runs experiments with different inertia weights.
 * - run_inertia_weight_experiment(inertia_weight): Runs an experiment with a specific inertia weight.
 * - run_acceleration_coefficients_experiments(coefficient_combinations): Runs experiments with 
 *   different acceleration coefficient combinations.
 * - run_acceleration_coefficients_experiment(c1, c2): Runs an experiment with specific acceleration coefficients.
 * - run_neighborhood_size_experiments(neighborhood_sizes): Runs experiments with different neighborhood sizes for SPSO.
 * - run_neighborhood_size_experiment(neighborhood_size): Runs an experiment with a specific neighborhood size for SPSO.
 * - run_cr_f_experiments(cr_f_combinations): Runs experiments with different CR and F values for DEPSO.
 * - run_cr_f_experiment(cr, f): Runs an experiment with specific CR and F values for DEPSO.
 * - run_fear_factor_experiments(fear_factors): Runs experiments with different fear factors for PPPSO.
 * - run_fear_factor_experiment(fear_factor): Runs an experiment with a specific fear factor for PPPSO.
 * - run_w_min_w_max_experiments(w_min_w_max_combinations): Runs experiments with different w_min and w_max values for APSO.
 * - run_w_min_w_max_experiment(w_min, w_max): Runs an experiment with specific w_min and w_max values for APSO.
 * - run_best_hyperparameter_experiments(): Runs experiments with the best hyperparameters for each algorithm.
 * - run_all_algorithms_convergence(): Runs all algorithms and plots their convergence.
 * - max_iterations_experiments(max_iterations_list): Runs experiments with different maximum iterations.
 * - run_max_iterations_experiment(max_iterations): Runs an experiment with a specific maximum iterations value.
********************************************************************************************************/
"""


class Experiment:
    def __init__(self, tsp_instance, num_runs=10, population_size=50, max_iterations=10, w=0.8, c1=2, c2=2, w_min=0.1, w_max=1.9, max_stagnation=10, neighborhood_size=5, cr=0.5, f=0.5, fear_factor=0.5):
        self.tsp_instance = tsp_instance.get_tsp_instance_list()
        self.tsp_instance_name = tsp_instance.name()
        self.num_runs = num_runs
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.w_min = w_min
        self.w_max = w_max
        self.max_stagnation = max_stagnation
        self.neighborhood_size = neighborhood_size
        self.cr = cr
        self.f = f
        self.fear_factor = fear_factor
        self.algorithms = ['PSO', 'APSO', 'HPSO', 'Random Sampling',
                           'Stochastic Hill Climber', 'spso', 'depso', 'pppso']
        self.data = {algo: [] for algo in self.algorithms}
        self.plotter = PrettyPlotting()

    def run(self, algorithms=None):
        """
        Runs the specified algorithms (or all algorithms if None) on the TSP instance.

        Parameters:
        - algorithms (list, optional): List of algorithm names to run. If None, all algorithms are run.

        Postconditions:
        - The results of each algorithm are stored in self.data.
        """
        if algorithms is None:
            algorithms = self.algorithms

        
        results = [self.run_algorithm(algo) for algo in algorithms]

        for algo, data in zip(algorithms, results):
            self.data[algo] = data

    def run_algorithm(self, algorithm):
        """
        Runs a single algorithm on the TSP instance for multiple runs.

        Parameters:
        - algorithm (str): Name of the algorithm to run.

        Returns:
        - data (list): List of dictionaries containing the results of each run.
        """
        data = Parallel(n_jobs=-1, backend='threading')(
        delayed(self._run_single_run)(algorithm) for _ in tqdm.tqdm(range(self.num_runs),desc=f'Running {algorithm}'))


        return data

    def _run_single_run(self, algorithm):
        """
        Runs a single run of an algorithm on the TSP instance.

        Parameters:
        - algorithm (str): Name of the algorithm to run.

        Returns:
        - result (dict): Dictionary containing the results of the run.
        """
        if algorithm == 'PSO':
            pso = PSO(self.tsp_instance, population_size=self.population_size, max_iterations=self.max_iterations,
                      w=self.w, c1=self.c1, c2=self.c2)
            best_solution, best_fitness, convergence_data, runtime = pso.optimize()
        elif algorithm == 'APSO':
            apso = APSO(self.tsp_instance, population_size=self.population_size, max_iterations=self.max_iterations,
                        w_min=self.w_min, w_max=self.w_max, c1=self.c1, c2=self.c2)
            best_solution, best_fitness, convergence_data, runtime = apso.optimize()
        elif algorithm == 'HPSO':
            hpso = DiscretePSO(self.tsp_instance, population_size=self.population_size,
                               max_iterations=self.max_iterations, w=self.w, c1=self.c1, c2=self.c2)
            best_solution, best_fitness, convergence_data, runtime = hpso.optimize()
        elif algorithm == 'Random Sampling':
            random_sampling = RandomSampling(
                self.tsp_instance, num_samples=self.max_iterations)
            best_solution, best_fitness, convergence_data, runtime = random_sampling.optimize()
        elif algorithm == 'Stochastic Hill Climber':
            stochastic_hill_climber = StochasticHillClimber(self.tsp_instance, max_iterations=self.max_iterations,
                                                            max_stagnation=self.max_stagnation)
            best_solution, best_fitness, convergence_data, runtime = stochastic_hill_climber.optimize()
        elif algorithm == 'spso':
            spso = SpatialPSO(self.tsp_instance, population_size=self.population_size,
                              max_iterations=self.max_iterations, w=self.w, c1=self.c1, c2=self.c2,
                              neighborhood_size=self.neighborhood_size)
            best_solution, best_fitness, convergence_data, runtime = spso.optimize()
        elif algorithm == 'depso':
            deps = DEPSO(self.tsp_instance, population_size=self.population_size, max_iterations=self.max_iterations,
                         w=self.w, c1=self.c1, c2=self.c2, cr=self.cr, f=self.f)
            best_solution, best_fitness, convergence_data, runtime = deps.optimize()
        elif algorithm == 'pppso':
            pppso = PredatorPreyPSO(self.tsp_instance, population_size=self.population_size,
                                    max_iterations=self.max_iterations, w=self.w, c1=self.c1, c2=self.c2,
                                    fear_factor=self.fear_factor)
            best_solution, best_fitness, convergence_data, runtime = pppso.optimize()

        return {'best_solution': best_solution, 'best_fitness': best_fitness, 'convergence_data': convergence_data,
                'runtime': runtime}

    def run_population_size_experiments(self, population_sizes):
        """
        Runs experiments with different population sizes.

        Parameters:
        - population_sizes (list): List of population sizes to experiment with.

        Postconditions:
        - The results are plotted using the plotter object.
        """
        with multiprocessing.Pool() as pool:
            results = pool.map(
                self.run_population_size_experiment, population_sizes)
        data = {pop_size: result for pop_size,
                result in zip(population_sizes, results)}
        self.plotter.line_plot_population_size(
            data, tsp_instance_name=self.tsp_instance_name)
        self.plotter.plot_runtime_population_size(
            data, tsp_instance_name=self.tsp_instance_name)

    def run_population_size_experiment(self, pop_size):
        """
        Runs an experiment with a specific population size.

        Parameters:
        - pop_size (int): Population size to use in the experiment.

        Returns:
        - self.data (dict): Dictionary containing the results of the experiment.
        """
        self.population_size = pop_size
        self.run()
        return self.data

    def run_inertia_weight_experiments(self, inertia_weights):
        """
        Runs experiments with different inertia weights.

        Parameters:
            inertia_weights (list): List of inertia weights to experiment with.

        Postconditions:
            The results are plotted using the plotter object.
        """
        results = Parallel(n_jobs=-1, backend='multiprocessing')(
            delayed(self.run_inertia_weight_experiment)(inertia_weight) for inertia_weight in inertia_weights)
        data = {inertia_weight: result for inertia_weight,
                result in zip(inertia_weights, results)}
        self.plotter.line_plot_inertia_weight(
            data, tsp_instance_name=self.tsp_instance_name)

    def run_inertia_weight_experiment(self, inertia_weight):
        """
        Runs an experiment with a specific inertia weight.

        Parameters:
            inertia_weight (float): Inertia weight to use in the experiment.

        Returns:
            self.data (dict): Dictionary containing the results of the experiment.
        """
        self.w = inertia_weight
        self.run(['PSO', 'HPSO', 'spso'])
        return self.data

    def run_acceleration_coefficients_experiments(self, coefficient_combinations):
        """
        Runs experiments with different acceleration coefficient combinations.

        Parameters:
            coefficient_combinations (list): List of tuples representing acceleration coefficient combinations (c1, c2).

        Postconditions:
            The results are plotted using the plotter object.
        """
        results = Parallel(n_jobs=-1, backend='multiprocessing')(
            delayed(self.run_acceleration_coefficients_experiment)(c1, c2) for c1, c2 in coefficient_combinations)
        data = {(c1, c2): result for (c1, c2), result in zip(
            coefficient_combinations, results)}
        self.plotter.line_plot_acceleration_coefficients(
            data, tsp_instance_name=self.tsp_instance_name)

    def run_acceleration_coefficients_experiment(self, c1, c2):
        """
        Runs an experiment with specific acceleration coefficients.

        Parameters:
            c1 (float): Cognitive acceleration coefficient.
            c2 (float): Social acceleration coefficient.

        Returns:
            self.data (dict): Dictionary containing the results of the experiment.
        """
        self.c1 = c1
        self.c2 = c2
        self.run(['PSO', 'HPSO', 'APSO', 'spso', 'depso', 'pppso'])
        return self.data

    def run_neighborhood_size_experiments(self, neighborhood_sizes):
        """
        Runs experiments with different neighborhood sizes for the SPSO algorithm.

        Parameters:
            neighborhood_sizes (list): List of neighborhood sizes to experiment with.

        Postconditions:
            The results are plotted using the plotter object.
        """
        results = Parallel(n_jobs=-1, backend='multiprocessing')(
            delayed(self.run_neighborhood_size_experiment)(neighborhood_size) for neighborhood_size in
            neighborhood_sizes)
        data = {neighborhood_size: result for neighborhood_size,
                result in zip(neighborhood_sizes, results)}
        self.plotter.line_plot_neighborhood_size(
            data, tsp_instance_name=self.tsp_instance_name)

    def run_neighborhood_size_experiment(self, neighborhood_size):
        """
        Runs an experiment with a specific neighborhood size for the SPSO algorithm.

        Parameters:
            neighborhood_size (int): Neighborhood size to use in the experiment.

        Returns:
            self.data (dict): Dictionary containing the results of the experiment.
        """
        self.neighborhood_size = neighborhood_size
        self.run(['spso'])
        return self.data

    def run_cr_f_experiments(self, cr_f_combinations):
        """
        Runs experiments with specific crossover probability and mutation factor for the DEPSO algorithm.

        Parameters:
            cr (float): Crossover probability.
            f (float): Mutation factor.

        Returns:
            self.data (dict): Dictionary containing the results of the experiment.
        """

        results = Parallel(n_jobs=-1, backend='multiprocessing')(
            delayed(self.run_cr_f_experiment)(cr, f) for cr, f in cr_f_combinations)
        data = {(cr, f): result for (cr, f),
                result in zip(cr_f_combinations, results)}
        self.plotter.line_plot_cr_f(
            data, tsp_instance_name=self.tsp_instance_name)

    def run_cr_f_experiment(self, cr, f):
        """
        Runs an experiment with specific crossover probability and mutation factor for the DEPSO algorithm.

        Parameters:
            cr (float): Crossover probability.
            f (float): Mutation factor.

        Returns:
            self.data (dict): Dictionary containing the results of the experiment.
        """
        self.cr = cr
        self.f = f
        self.run(['depso'])
        return self.data

    def run_fear_factor_experiments(self, fear_factors):
        """
        Runs experiments with different fear factors for the PPPSO algorithm.

        Parameters:
            fear_factors (list): List of fear factors to experiment with.

        Postconditions:
            The results are plotted using the plotter object.
        """
        results = Parallel(n_jobs=-1, backend='multiprocessing')(
            delayed(self.run_fear_factor_experiment)(fear_factor) for fear_factor in fear_factors)
        data = {fear_factor: result for fear_factor,
                result in zip(fear_factors, results)}
        self.plotter.line_plot_fear_factor(
            data, tsp_instance_name=self.tsp_instance_name)

    def run_fear_factor_experiment(self, fear_factor):
        """
        Runs an experiment with a specific fear factor for the PPPSO algorithm.

        Parameters:
        fear_factor (float): Fear factor to use in the experiment.

         Returns:
        self.data (dict): Dictionary containing the results of the experiment.
        """
        self.fear_factor = fear_factor
        self.run(['pppso'])
        return self.data

    def run_w_min_w_max_experiments(self, w_min_w_max_combinations):
        """
        Runs experiments with different minimum and maximum inertia weight combinations for the APSO algorithm.

        Parameters:
            w_min_w_max_combinations (list): List of tuples representing minimum and maximum inertia weight combinations (w_min, w_max).

        Postconditions:
            The results are plotted using the plotter object.
        """
        results = Parallel(n_jobs=-1, backend='multiprocessing')(
            delayed(self.run_w_min_w_max_experiment)(w_min, w_max) for w_min, w_max in w_min_w_max_combinations)
        data = {(w_min, w_max): result for (w_min, w_max),
                result in zip(w_min_w_max_combinations, results)}
        self.plotter.line_plot_w_min_w_max(
            data, tsp_instance_name=self.tsp_instance_name)

    def run_w_min_w_max_experiment(self, w_min, w_max):
        """
        Runs an experiment with specific minimum and maximum inertia weights for the APSO algorithm.

        Parameters:
            w_min (float): Minimum inertia weight.
            w_max (float): Maximum inertia weight.

        Returns:
            self.data (dict): Dictionary containing the results of the experiment.
        """
        self.w_min = w_min
        self.w_max = w_max
        self.run(['APSO'])
        return self.data

    def run_best_hyperparameter_experiments(self):
        """
        Runs experiments with the best hyperparameters for all algorithms.

        Postconditions:
            The results are plotted using the plotter object.
        """
        self.run()
        self.plotter.convergence_plot_random_hyperparameters(
            self.data, tsp_instance_name=self.tsp_instance_name)

    def run_all_algorithms_convergence(self):
        """
        Runs all algorithms and plots their convergence.

        Postconditions:
            The results are plotted using the plotter object.
        """
        self.run()
        self.plotter.convergence_plot_all_algorithms(
            self.data, tsp_instance_name=self.tsp_instance_name)

    def max_iterations_experiments(self, max_iterations_list):
        """
        Runs experiments with different maximum iteration values.

        Parameters:
            max_iterations_list (list): List of maximum iteration values to experiment with.

        Postconditions:
            The results are plotted using the plotter object.
        """
        results = Parallel(n_jobs=-1, backend='multiprocessing')(
            delayed(self.run_max_iterations_experiment)(max_iterations) for max_iterations in max_iterations_list)
        data = {max_iterations: result for max_iterations,
                result in zip(max_iterations_list, results)}
        self.plotter.line_plot_max_iterations(
            data, tsp_instance_name=self.tsp_instance_name)

    def run_max_iterations_experiment(self, max_iterations):
        """
        Runs an experiment with a specific maximum iteration value.

        Parameters:
            max_iterations (int): Maximum number of iterations to use in the experiment.

        Returns:
            self.data (dict): Dictionary containing the results of the experiment.
        """
        self.max_iterations = max_iterations
        self.run()
        return self.data
