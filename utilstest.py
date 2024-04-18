import math
import polars as pl
import os
import math
import time
from abc import ABC, abstractmethod
import random
from math import sqrt,tanh
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import multiprocessing
import concurrent.futures
'''
********************************************************************************************************
'''
class ProcessData:
    def __init__(self, tsp_file_path, csv_file_path):
        self.tsp_file_path = tsp_file_path
        self.csv_file_path = csv_file_path
        self.data = None
        self.distance_matrix = None
        self.tsp_instance_list = None

    def read_tsp_file(self):
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
        n = len(self.data)
        self.distance_matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = self.data[i]
                x2, y2 = self.data[j]
                distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                self.distance_matrix[i][j] = self.distance_matrix[j][i] = distance

    def save_distance_matrix_to_csv(self):
        df_distance = pl.DataFrame(self.distance_matrix)
        df_distance.write_csv(self.csv_file_path)

    def load_distance_matrix_from_csv(self):
        df_d = pl.read_csv(self.csv_file_path)
        self.tsp_instance_list = df_d.to_numpy().tolist()

    def process(self):
        self.read_tsp_file()
        self.calculate_distance_matrix()
        self.save_distance_matrix_to_csv()
        self.load_distance_matrix_from_csv()

    def get_tsp_instance_list(self):
        self.load_distance_matrix_from_csv()
        return self.tsp_instance_list

    def name(self):
        tsp_name = os.path.basename(self.tsp_file_path).split('.')[0]
        return tsp_name


# # Usage example
# tsp_file_path = 'a280.tsp'
# csv_file_path = 'a280_distance_matrix.csv'
#
# data_processor = ProcessData(tsp_file_path, csv_file_path)
'''
********************************************************************************************************
'''


class StochasticHillClimber:
    def __init__(self, tsp_instance, max_iterations, max_stagnation):
        self.tsp_instance = tsp_instance
        self.max_iterations = max_iterations
        self.max_stagnation = max_stagnation

    def optimize(self):
        iteration = 0
        stagnation_counter = 0
        num_cities = len(self.tsp_instance)
        best_solution = list(range(num_cities))  # Initialize with a random solution
        random.shuffle(best_solution)  # Shuffle the initial solution
        best_fitness = PSO.calculate_fitness(best_solution, self.tsp_instance)
        convergence_data = []
        start_time = time.time()

        while iteration < self.max_iterations and stagnation_counter < self.max_stagnation:
            new_solution = list(best_solution)

            if len(new_solution) >= 2:
                i, j = random.sample(range(len(new_solution)), 2)
                new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
                new_fitness = PSO.calculate_fitness(new_solution, self.tsp_instance)

                if new_fitness < best_fitness:
                    best_solution = new_solution
                    best_fitness = new_fitness
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

                convergence_data.append(best_fitness)
            else:
                # Handle the case where the length of new_solution is less than 2
                pass

            iteration += 1

        runtime = time.time() - start_time
        return best_solution, best_fitness, convergence_data, runtime


class RandomSampling:
    def __init__(self, tsp_instance, num_samples):
        self.tsp_instance = tsp_instance
        self.num_samples = num_samples

    def optimize(self):
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
        for i in range(len(particle.velocity)):
            r1 = random.random()
            r2 = random.random()
            cognitive_velocity = self.c1 * r1 * (particle.pbest_solution[i] - particle.solution[i])
            social_velocity = self.c2 * r2 * (self.gbest_solution[i] - particle.solution[i])
            particle.velocity[i] = self.w * particle.velocity[i] + cognitive_velocity + social_velocity

    @abstractmethod
    def update_position(self, particle):
        pass

    def optimize(self):
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


class PSO(BasePSO):
    def create_particle(self):
        solution = random.sample(range(1, len(self.tsp_instance) + 1), len(self.tsp_instance))
        fitness = self.calculate_fitness(solution, self.tsp_instance)
        return BasePSO.Particle(solution, fitness)

    @staticmethod
    def calculate_fitness(solution, tsp_instance):
        total_distance = 0
        for i in range(len(solution)):
            city1 = solution[i] - 1  # Adjust the indexing to start from 0
            city2 = solution[(i + 1) % len(solution)] - 1  # Adjust the indexing to start from 0
            total_distance += tsp_instance[city1][city2]
        return total_distance

    def update_position(self, particle):
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


class APSO(PSO):
    def __init__(self, tsp_instance, population_size, max_iterations, w_min, w_max, c1, c2):
        super().__init__(tsp_instance, population_size, max_iterations, None, c1, c2)
        self.w_min = w_min
        self.w_max = w_max

    def update_velocity(self, particle, iteration):
        w = self.w_max - (self.w_max - self.w_min) * (iteration / self.max_iterations)
        for i in range(len(particle.velocity)):
            r1 = random.random()
            r2 = random.random()
            cognitive_velocity = self.c1 * r1 * (particle.pbest_solution[i] - particle.solution[i])
            social_velocity = self.c2 * r2 * (self.gbest_solution[i] - particle.solution[i])
            particle.velocity[i] = w * particle.velocity[i] + cognitive_velocity + social_velocity

    def optimize(self):
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


class DiscretePSO(PSO):
    def create_particle(self):
        solution = random.sample(range(1, len(self.tsp_instance) + 1), len(self.tsp_instance))
        fitness = self.calculate_fitness(solution, self.tsp_instance)
        return BasePSO.Particle(solution, fitness)

    def update_velocity(self, particle):
        for i in range(len(particle.velocity)):
            r1 = random.random()
            r2 = random.random()
            cognitive_velocity = self.c1 * r1 * (particle.pbest_solution[i] - particle.solution[i])
            social_velocity = self.c2 * r2 * (self.gbest_solution[i] - particle.solution[i])
            particle.velocity[i] = math.ceil(self.w * particle.velocity[i] + cognitive_velocity + social_velocity)

    def update_position(self, particle):
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


class SpatialPSO(BasePSO):
    def __init__(self, tsp_instance, population_size, max_iterations, w, c1, c2, neighborhood_size):
        super().__init__(tsp_instance, population_size, max_iterations, w, c1, c2)
        self.neighborhood_size = neighborhood_size

    def create_particle(self):
        solution = random.sample(range(len(self.tsp_instance)), len(self.tsp_instance))
        fitness = self.calculate_fitness(solution)
        return self.Particle(solution, fitness)

    def calculate_fitness(self, solution):
        total_distance = 0
        for i in range(len(solution)):
            city1 = solution[i]
            city2 = solution[(i + 1) % len(solution)]
            total_distance += self.tsp_instance[city1][city2]
        return total_distance

    def update_position(self, particle):
        new_solution = particle.solution[:]
        for i in range(len(particle.solution)):
            if random.random() < self.sigmoid(particle.velocity[i]):
                # Swap the current city with the city in pbest
                j = particle.pbest_solution.index(particle.solution[i])
                new_solution[i], new_solution[j] = new_solution[j], new_solution[i]

        particle.solution = new_solution
        particle.fitness = self.calculate_fitness(particle.solution)
        if particle.fitness < particle.pbest_fitness:
            particle.pbest_solution = particle.solution
            particle.pbest_fitness = particle.fitness

    def sigmoid(self, x):
        return (1 + tanh(x)) / 2

    def optimize(self):
        start_time = time.time()
        swarm = [self.create_particle() for _ in range(self.population_size)]

        for particle in swarm:
            particle.neighbors = self.find_neighbors(particle, swarm)

        self.gbest_solution = min(swarm, key=lambda p: p.fitness).solution
        self.gbest_fitness = min(swarm, key=lambda p: p.fitness).fitness

        convergence_data = []
        for iteration in range(self.max_iterations):
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
        distances = [(other_particle, self.euclidean_distance(particle.solution, other_particle.solution))
                     for other_particle in swarm if other_particle != particle]
        distances.sort(key=lambda x: x[1])
        return [p for p, _ in distances[:self.neighborhood_size]]

    def euclidean_distance(self, solution1, solution2):
        return sqrt(sum((coord1 - coord2) ** 2 for coord1, coord2 in zip(solution1, solution2)))


class DEPSO(BasePSO):
    def __init__(self, tsp_instance, population_size, max_iterations, w, c1, c2, cr, f):
        super().__init__(tsp_instance, population_size, max_iterations, w, c1, c2)
        self.cr = cr
        self.f = f

    def create_particle(self):
        solution = random.sample(range(len(self.tsp_instance)), len(self.tsp_instance))
        fitness = self.calculate_fitness(solution)
        return self.Particle(solution, fitness)

    def calculate_fitness(self, solution):
        total_distance = 0
        for i in range(len(solution)):
            city1 = solution[i]
            city2 = solution[(i + 1) % len(solution)]
            total_distance += self.tsp_instance[city1][city2]
        return total_distance

    def update_position(self, particle):
        new_solution = particle.solution[:]
        # Select two random particles from the swarm
        r1, r2 = random.sample(range(self.population_size), 2)
        for i in range(len(particle.solution)):
            if random.random() < self.cr:
                # Find a city that is present in both solutions
                common_cities = set(particle.solution) & set(self.swarm[r1].solution) & set(self.swarm[r2].solution)
                if common_cities:
                    city = random.choice(list(common_cities))
                    j1 = self.swarm[r1].solution.index(city)
                    j2 = self.swarm[r2].solution.index(city)

                    # Apply the DE formula to get the new city index
                    new_index = (j1 + int(self.f * (j2 - j1))) % len(particle.solution)

                    # Find the city at the new index in the current solution
                    new_city = particle.solution[new_index]

                    # Swap the current city with the new city
                    current_index = particle.solution.index(city)
                    new_solution[current_index] = new_city
                    new_solution[new_index] = city

        new_fitness = self.calculate_fitness(new_solution)
        if new_fitness < particle.fitness:
            particle.solution = new_solution
            particle.fitness = new_fitness

            if particle.fitness < particle.pbest_fitness:
                particle.pbest_solution = particle.solution
                particle.pbest_fitness = particle.fitness

    def optimize(self):
        start_time = time.time()
        self.swarm = [self.create_particle() for _ in range(self.population_size)]

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


class PredatorPreyPSO(BasePSO):
    def __init__(self, tsp_instance, population_size, max_iterations, w, c1, c2, fear_factor):
        super().__init__(tsp_instance, population_size, max_iterations, w, c1, c2)
        self.fear_factor = fear_factor

    def create_particle(self):
        solution = random.sample(range(len(self.tsp_instance)), len(self.tsp_instance))
        fitness = self.calculate_fitness(solution)
        return self.Particle(solution, fitness)

    def calculate_fitness(self, solution):
        total_distance = 0
        for i in range(len(solution)):
            city1 = solution[i]
            city2 = solution[(i + 1) % len(solution)]
            total_distance += self.tsp_instance[city1][city2]
        return total_distance

    def update_velocity(self, particle, predator):
        for i in range(len(particle.velocity)):
            r1 = random.random()
            r2 = random.random()
            cognitive_velocity = self.c1 * r1 * (particle.pbest_solution[i] - particle.solution[i])
            social_velocity = self.c2 * r2 * (self.gbest_solution[i] - particle.solution[i])
            predator_velocity = self.fear_factor * (particle.solution[i] - predator.solution[i])
            particle.velocity[i] = self.w * particle.velocity[
                i] + cognitive_velocity + social_velocity + predator_velocity

    def update_position(self, particle):
        new_solution = particle.solution[:]
        for i in range(len(particle.solution)):
            if random.random() < self.sigmoid(particle.velocity[i]):
                # Swap the current city with the city in pbest
                j = particle.pbest_solution.index(particle.solution[i])
                new_solution[i], new_solution[j] = new_solution[j], new_solution[i]

        particle.solution = new_solution
        particle.fitness = self.calculate_fitness(particle.solution)
        if particle.fitness < particle.pbest_fitness:
            particle.pbest_solution = particle.solution
            particle.pbest_fitness = particle.fitness

    def sigmoid(self, x):
        return (1 + tanh(x)) / 2

    def optimize(self):
        start_time = time.time()
        self.swarm = [self.create_particle() for _ in range(self.population_size)]
        predator = self.create_particle()

        self.gbest_solution = min(self.swarm, key=lambda p: p.fitness).solution
        self.gbest_fitness = min(self.swarm, key=lambda p: p.fitness).fitness

        convergence_data = []
        for iteration in range(self.max_iterations):
            for particle in self.swarm:
                self.update_velocity(particle, predator)
                self.update_position(particle)

                if particle.fitness < self.gbest_fitness:
                    self.gbest_solution = particle.solution
                    self.gbest_fitness = particle.fitness

            convergence_data.append(self.gbest_fitness)
            predator.solution = self.gbest_solution

        end_time = time.time()
        runtime = end_time - start_time

        return self.gbest_solution, self.gbest_fitness, convergence_data, runtime

'''
********************************************************************************************************
'''


class PrettyPlotting:
    def __init__(self):
        # plt.style.use('seaborn')
        sns.set(style='whitegrid', palette='muted', font_scale=1.2)

    def convergence_plot(self, data, algorithms, problem_instance):
        plt.figure(figsize=(8, 6))
        for algorithm in algorithms:
            for run_data in data[algorithm]:
                iterations = range(1, len(run_data['convergence_data']) + 1)
                best_tour_lengths = run_data['convergence_data']
                plt.plot(iterations, best_tour_lengths,
                         label=f"{algorithm} - Run {data[algorithm].index(run_data) + 1}")
        plt.xlabel('Number of Iterations')
        plt.ylabel('Best Tour Length')
        plt.title(f'Convergence Plot - Problem Instance: ')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_data(self, data, tsp_instance_name, iterations=100):
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
            color = color_map[algorithm]  # Get the color for the current algorithm
            for run_data in data[algorithm]:
                iterations = np.linspace(1, count, count)
                plt.plot(iterations, run_data['convergence_data'], color=color, label=f'{algorithm}')

        # Remove duplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small')

        plt.xlabel('Iterations')
        plt.ylabel('Fitness')
        plt.suptitle(f'TSP Instance: {tsp_instance_name}', fontsize=14)
        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))  # Adjust the plot layout to accommodate the suptitle

        plt.show()

    def box_plot_tour_lengths(self, data, algorithms, tsp_instance_name):
        plt.figure(figsize=(10, 6))
        tour_lengths = []
        for algorithm in algorithms:
            algorithm_data = data[algorithm]
            if algorithm == 'HPSO':
                tour_lengths.append([run['best_fitness'] for run in algorithm_data])
            else:
                # Assuming the same data structure for other algorithms
                tour_lengths.append([run['best_fitness'] for run in algorithm_data])

        # Create the boxplot
        bp = plt.boxplot(tour_lengths, labels=algorithms)

        # Rotate x-labels by 45 degrees
        plt.xticks(rotation=45, ha='right')

        # Adjust the bottom margin to prevent overlapping
        plt.subplots_adjust(bottom=0.2)

        plt.ylabel('Best Tour Length')
        plt.title('Distribution of Best Tour Lengths')
        plt.suptitle(f'TSP Instance: {tsp_instance_name}', fontsize=14)
        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))  # Adjust the plot layout to accommodate the suptitle

    def performance_heatmap(self, data, algorithms, instances=100):
        # Create a dictionary to store performance ranks
        performance_ranks = {}

        # Iterate over instances
        for instance in instances:
            # Sort algorithms by best tour length for this instance
            sorted_algos = sorted(algorithms, key=lambda algo: min(data[algo][instance]['best_fitness']))

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
        plt.figure(figsize=(10, 6))
        population_sizes = []
        pso_best_lengths = []
        hpso_best_lengths = []
        apso_best_lengths = []
        spso_best_lengths = []
        depso_best_lengths = []
        pppso_best_lengths = []

        for pop_size, runs in data.items():
            population_sizes.append(pop_size)
            pso_best_lengths.append(min(run['best_fitness'] for run in runs['PSO']))
            hpso_best_lengths.append(min(run['best_fitness'] for run in runs['HPSO']))
            apso_best_lengths.append(min(run['best_fitness'] for run in runs['APSO']))
            spso_best_lengths.append(min(run['best_fitness'] for run in runs['spso']))
            depso_best_lengths.append(min(run['best_fitness'] for run in runs['depso']))
            pppso_best_lengths.append(min(run['best_fitness'] for run in runs['pppso']))

        plt.plot(population_sizes, pso_best_lengths, marker='o', label='PSO')
        plt.plot(population_sizes, hpso_best_lengths, marker='o', label='HPSO')
        plt.plot(population_sizes, apso_best_lengths, marker='o', label='APSO')
        plt.plot(population_sizes, spso_best_lengths, marker='o', label='SPSO')
        plt.plot(population_sizes, depso_best_lengths, marker='o', label='DEPSO')
        plt.plot(population_sizes, pppso_best_lengths, marker='o', label='PPPSO')

        plt.xlabel('Population Size')
        plt.ylabel('Best Tour Length')
        plt.title('Impact of Population Size on Best Tour Length')
        plt.legend()
        plt.suptitle(f'TSP Instance: {tsp_instance_name}', fontsize=14)
        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
        plt.show()

    def plot_runtime_population_size(self, data, tsp_instance_name):
        plt.figure(figsize=(10, 6))

        population_sizes = list(data.keys())
        algorithms = ['PSO', 'HPSO', 'APSO', 'spso', 'depso', 'pppso']
        runtimes = {algo: [] for algo in algorithms}

        for pop_size in population_sizes:
            runs = data[pop_size]
            for algo in algorithms:
                if algo in runs:
                    avg_runtime = sum(run['runtime'] for run in runs[algo]) / len(runs[algo])
                    runtimes[algo].append(avg_runtime)
                else:
                    runtimes[algo].append(0)  # If the algorithm is not present, append 0 as runtime

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
        plt.figure(figsize=(10, 6))
        inertia_weights = []
        pso_best_lengths = []
        hpso_best_lengths = []
        spso_best_lengths = []

        for inertia_weight, runs in data.items():
            inertia_weights.append(inertia_weight)
            pso_best_lengths.append(min(run['best_fitness'] for run in runs['PSO']))
            hpso_best_lengths.append(min(run['best_fitness'] for run in runs['HPSO']))
            spso_best_lengths.append(min(run['best_fitness'] for run in runs['spso']))

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
        plt.figure(figsize=(10, 6))
        pso_best_lengths = []
        hpso_best_lengths = []
        apso_best_lengths = []
        spso_best_lengths = []
        depso_best_lengths = []
        pppso_best_lengths = []

        for coefficients, runs in data.items():
            c1, c2 = coefficients
            pso_best_lengths.append(min(run['best_fitness'] for run in runs['PSO']))
            hpso_best_lengths.append(min(run['best_fitness'] for run in runs['HPSO']))
            apso_best_lengths.append(min(run['best_fitness'] for run in runs['APSO']))
            spso_best_lengths.append(min(run['best_fitness'] for run in runs['spso']))
            depso_best_lengths.append(min(run['best_fitness'] for run in runs['depso']))
            pppso_best_lengths.append(min(run['best_fitness'] for run in runs['pppso']))

        plt.plot(range(len(data)), pso_best_lengths, marker='o', label='PSO')
        plt.plot(range(len(data)), hpso_best_lengths, marker='o', label='HPSO')
        plt.plot(range(len(data)), apso_best_lengths, marker='o', label='APSO')
        plt.plot(range(len(data)), spso_best_lengths, marker='o', label='SPSO')
        plt.plot(range(len(data)), depso_best_lengths, marker='o', label='DEPSO')
        plt.plot(range(len(data)), pppso_best_lengths, marker='o', label='PPPSO')
        plt.xlabel('Acceleration Coefficient Combination')
        plt.ylabel('Best Tour Length')
        plt.title('Impact of Acceleration Coefficients on Best Tour Length (PSO)')
        plt.xticks(range(len(data)), [f'c1={c1}, c2={c2}' for c1, c2 in data.keys()], rotation=45)
        plt.legend()
        plt.suptitle(f'TSP Instance: {tsp_instance_name}', fontsize=14)
        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
        plt.show()

    def convergence_plot_best_hyperparameters(self, data, tsp_instance_name):
        plt.figure(figsize=(10, 6))
        iterations = range(1, len(data['PSO'][0]['convergence_data']) + 1)

        pso_convergence_data = [run['convergence_data'] for run in data['PSO']]
        pso_mean_convergence = np.mean(pso_convergence_data, axis=0)
        hpso_convergence_data = [run['convergence_data'] for run in data['HPSO']]
        hpso_mean_convergence = np.mean(hpso_convergence_data, axis=0)

        plt.plot(iterations, pso_mean_convergence, label='PSO')
        plt.plot(iterations, hpso_mean_convergence, label='HPSO')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Best Tour Length')
        plt.title('Convergence Plot - Best Hyperparameter Settings')
        plt.legend()
        plt.suptitle(f'TSP Instance: {tsp_instance_name}', fontsize=14)
        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
        plt.show()

    def line_plot_neighborhood_size(self, data, tsp_instance_name):
        plt.figure(figsize=(10, 6))
        neighborhood_sizes = []
        spso_best_lengths = []

        for neighborhood_size, runs in data.items():
            neighborhood_sizes.append(neighborhood_size)
            spso_best_lengths.append(min(run['best_fitness'] for run in runs['spso']))

        plt.plot(neighborhood_sizes, spso_best_lengths, marker='o', label='SPSO')
        plt.xlabel('Neighborhood Size')
        plt.ylabel('Best Tour Length')
        plt.title('Impact of Neighborhood Size on Best Tour Length (SPSO)')
        plt.legend()
        plt.suptitle(f'TSP Instance: {tsp_instance_name}', fontsize=14)
        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
        plt.show()

    def line_plot_cr_f(self, data, tsp_instance_name):
        plt.figure(figsize=(10, 6))
        depso_best_lengths = []

        for (cr, f), runs in data.items():
            depso_best_lengths.append(min(run['best_fitness'] for run in runs['depso']))

        plt.plot(range(len(data)), depso_best_lengths, marker='o', label='DEPSO')
        plt.xlabel('CR and F Combination')
        plt.ylabel('Best Tour Length')
        plt.title('Impact of CR and F on Best Tour Length (DEPSO)')
        plt.xticks(range(len(data)), [f'CR={cr}, F={f}' for cr, f in data.keys()], rotation=45)
        plt.legend()
        plt.suptitle(f'TSP Instance: {tsp_instance_name}', fontsize=14)
        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
        plt.show()

    def line_plot_fear_factor(self, data, tsp_instance_name):
        plt.figure(figsize=(10, 6))
        fear_factors = []
        pppso_best_lengths = []

        for fear_factor, runs in data.items():
            fear_factors.append(fear_factor)
            pppso_best_lengths.append(min(run['best_fitness'] for run in runs['pppso']))

        plt.plot(fear_factors, pppso_best_lengths, marker='o', label='PPPSO')
        plt.xlabel('Fear Factor')
        plt.ylabel('Best Tour Length')
        plt.title('Impact of Fear Factor on Best Tour Length (PPPSO)')
        plt.legend()
        plt.suptitle(f'TSP Instance: {tsp_instance_name}', fontsize=14)
        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
        plt.show()

    def convergence_plot_all_algorithms(self, data, tsp_instance_name):
        plt.figure(figsize=(10, 6))
        iterations = range(1, len(data['PSO'][0]['convergence_data']) + 1)

        for algorithm in data:
            convergence_data = [run['convergence_data'] for run in data[algorithm]]
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
        plt.figure(figsize=(10, 6))
        apso_best_lengths = []

        for (w_min, w_max), runs in data.items():
            apso_best_lengths.append(min(run['best_fitness'] for run in runs['APSO']))

        plt.plot(range(len(data)), apso_best_lengths, marker='o', label='APSO')
        plt.xlabel('w_min and w_max Combination')
        plt.ylabel('Best Tour Length')
        plt.title('Impact of w_min and w_max on Best Tour Length (APSO)')
        plt.xticks(range(len(data)), [f'w_min={w_min}, w_max={w_max}' for w_min, w_max in data.keys()], rotation=45)
        plt.legend()
        plt.suptitle(f'TSP Instance: {tsp_instance_name}', fontsize=14)
        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
        plt.show()

'''
********************************************************************************************************
'''

class Experiment:
    def __init__(self, tsp_instance, num_runs=10, population_size=50, max_iterations=10, w=0.8, c1=2, c2=2, w_min=0.1, w_max=1.9, max_stagnation=10, neighborhood_size=5,cr =0.5,f = 0.5,fear_factor = 0.5):
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
        self.algorithms = ['PSO', 'APSO', 'HPSO', 'Random Sampling', 'Stochastic Hill Climber','spso', 'depso','pppso']
        self.data = {algo: [] for algo in self.algorithms}
        self.plotter = PrettyPlotting()

    def run(self, algorithms=None):
        if algorithms is None:
            algorithms = self.algorithms

        # results = Parallel(n_jobs=-1)(
        #     delayed(self.run_algorithm)(algo) for algo in algorithms)
        results = [self.run_algorithm(algo) for algo in algorithms]

        for algo, data in zip(algorithms, results):
            self.data[algo] = data

    def run_algorithm(self, algorithm):
        data = Parallel(n_jobs=-1,backend='threading')(
            delayed(self._run_single_run)(algorithm) for _ in range(self.num_runs))

        return data

    def _run_single_run(self, algorithm):
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
            random_sampling = RandomSampling(self.tsp_instance, num_samples=self.max_iterations)
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
        with multiprocessing.Pool() as pool:
            results = pool.map(self.run_population_size_experiment, population_sizes)
        data = {pop_size: result for pop_size, result in zip(population_sizes, results)}
        self.plotter.line_plot_population_size(data, tsp_instance_name=self.tsp_instance_name)
        self.plotter.plot_runtime_population_size(data, tsp_instance_name=self.tsp_instance_name)
    def run_population_size_experiment(self, pop_size):
        self.population_size = pop_size
        self.run()
        return self.data
    def run_inertia_weight_experiments(self, inertia_weights):
        results = Parallel(n_jobs=-1, backend='multiprocessing')(
            delayed(self.run_inertia_weight_experiment)(inertia_weight) for inertia_weight in inertia_weights)
        data = {inertia_weight: result for inertia_weight, result in zip(inertia_weights, results)}
        self.plotter.line_plot_inertia_weight(data, tsp_instance_name=self.tsp_instance_name)

    def run_inertia_weight_experiment(self, inertia_weight):
        self.w = inertia_weight
        self.run(['PSO', 'HPSO', 'spso'])
        return self.data

    def run_acceleration_coefficients_experiments(self, coefficient_combinations):
        results = Parallel(n_jobs=-1, backend='multiprocessing')(
            delayed(self.run_acceleration_coefficients_experiment)(c1, c2) for c1, c2 in coefficient_combinations)
        data = {(c1, c2): result for (c1, c2), result in zip(coefficient_combinations, results)}
        self.plotter.line_plot_acceleration_coefficients(data, tsp_instance_name=self.tsp_instance_name)

    def run_acceleration_coefficients_experiment(self, c1, c2):
        self.c1 = c1
        self.c2 = c2
        self.run(['PSO', 'HPSO', 'APSO', 'spso', 'depso', 'pppso'])
        return self.data

    def run_neighborhood_size_experiments(self, neighborhood_sizes):
        results = Parallel(n_jobs=-1, backend='multiprocessing')(
            delayed(self.run_neighborhood_size_experiment)(neighborhood_size) for neighborhood_size in
            neighborhood_sizes)
        data = {neighborhood_size: result for neighborhood_size, result in zip(neighborhood_sizes, results)}
        self.plotter.line_plot_neighborhood_size(data, tsp_instance_name=self.tsp_instance_name)

    def run_neighborhood_size_experiment(self, neighborhood_size):
        self.neighborhood_size = neighborhood_size
        self.run(['spso'])
        return self.data

    def run_cr_f_experiments(self, cr_f_combinations):
        results = Parallel(n_jobs=-1, backend='multiprocessing')(
            delayed(self.run_cr_f_experiment)(cr, f) for cr, f in cr_f_combinations)
        data = {(cr, f): result for (cr, f), result in zip(cr_f_combinations, results)}
        self.plotter.line_plot_cr_f(data, tsp_instance_name=self.tsp_instance_name)

    def run_cr_f_experiment(self, cr, f):
        self.cr = cr
        self.f = f
        self.run(['depso'])
        return self.data

    def run_fear_factor_experiments(self, fear_factors):
        results = Parallel(n_jobs=-1, backend='multiprocessing')(
            delayed(self.run_fear_factor_experiment)(fear_factor) for fear_factor in fear_factors)
        data = {fear_factor: result for fear_factor, result in zip(fear_factors, results)}
        self.plotter.line_plot_fear_factor(data, tsp_instance_name=self.tsp_instance_name)

    def run_fear_factor_experiment(self, fear_factor):
        self.fear_factor = fear_factor
        self.run(['pppso'])
        return self.data

    def run_w_min_w_max_experiments(self, w_min_w_max_combinations):
        results = Parallel(n_jobs=-1, backend='multiprocessing')(
            delayed(self.run_w_min_w_max_experiment)(w_min, w_max) for w_min, w_max in w_min_w_max_combinations)
        data = {(w_min, w_max): result for (w_min, w_max), result in zip(w_min_w_max_combinations, results)}
        self.plotter.line_plot_w_min_w_max(data, tsp_instance_name=self.tsp_instance_name)

    def run_w_min_w_max_experiment(self, w_min, w_max):
        self.w_min = w_min
        self.w_max = w_max
        self.run(['APSO'])
        return self.data

    def run_best_hyperparameter_experiments(self):
        self.run()
        self.plotter.convergence_plot_best_hyperparameters(self.data, tsp_instance_name=self.tsp_instance_name)

    def run_all_algorithms_convergence(self):
        self.run()
        self.plotter.convergence_plot_all_algorithms(self.data, tsp_instance_name=self.tsp_instance_name)
