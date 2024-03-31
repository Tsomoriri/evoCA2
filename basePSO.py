import random
import math
from abc import ABC, abstractmethod


class BasePSO(ABC):
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
        swarm = [self.create_particle() for _ in range(self.population_size)]
        for iteration in range(self.max_iterations):
            for particle in swarm:
                self.update_velocity(particle)
                self.update_position(particle)
        return self.gbest_solution, self.gbest_fitness


class PSO(BasePSO):
    def create_particle(self):
        solution = random.sample(range(1, len(self.tsp_instance) + 1), len(self.tsp_instance))
        fitness = self.calculate_fitness(solution)
        return Particle(solution, fitness)

    def calculate_fitness(self, solution):
        total_distance = 0
        for i in range(len(solution)):
            city1 = solution[i]
            city2 = solution[(i + 1) % len(solution)]
            total_distance += self.tsp_instance[city1 - 1][city2 - 1]
        return total_distance

    def update_position(self, particle):
        new_solution = particle.solution[:]
        for i in range(len(new_solution)):
            if random.random() < math.tanh(abs(particle.velocity[i])):
                j = random.randint(0, len(new_solution) - 1)
                new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
        new_fitness = self.calculate_fitness(new_solution)
        if new_fitness < particle.fitness:
            particle.solution = new_solution
            particle.fitness = new_fitness
            if new_fitness < particle.pbest_fitness:
                particle.pbest_solution = new_solution
                particle.pbest_fitness = new_fitness
                if new_fitness < self.gbest_fitness:
                    self.gbest_solution = new_solution
                    self.gbest_fitness = new_fitness


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
        swarm = [self.create_particle() for _ in range(self.population_size)]
        for iteration in range(self.max_iterations):
            for particle in swarm:
                self.update_velocity(particle, iteration)
                self.update_position(particle)
        return self.gbest_solution, self.gbest_fitness


class BPSO(BasePSO):
    def create_particle(self):
        solution = [random.randint(0, 1) for _ in range(len(self.tsp_instance))]
        fitness = self.calculate_fitness(solution)
        return Particle(solution, fitness)

    def calculate_fitness(self, solution):
        total_distance = 0
        selected_cities = [i + 1 for i, bit in enumerate(solution) if bit == 1]
        for i in range(len(selected_cities)):
            city1 = selected_cities[i]
            city2 = selected_cities[(i + 1) % len(selected_cities)]
            total_distance += self.tsp_instance[city1 - 1][city2 - 1]
        return total_distance

    def update_position(self, particle):
        new_solution = particle.solution[:]
        for i in range(len(new_solution)):
            if random.random() < 1 / (1 + math.exp(-particle.velocity[i])):
                new_solution[i] = 1 - new_solution[i]
        new_fitness = self.calculate_fitness(new_solution)
        if new_fitness < particle.fitness:
            particle.solution = new_solution
            particle.fitness = new_fitness
            if new_fitness < particle.pbest_fitness:
                particle.pbest_solution = new_solution
                particle.pbest_fitness = new_fitness
                if new_fitness < self.gbest_fitness:
                    self.gbest_solution = new_solution
                    self.gbest_fitness = new_fitness


# Usage example
tsp_instance = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]

pso = PSO(tsp_instance, population_size=50, max_iterations=100, w=0.8, c1=2, c2=2)
best_solution, best_fitness = pso.optimize()
print("PSO Best solution:", best_solution)
print("PSO Best fitness:", best_fitness)

apso = APSO(tsp_instance, population_size=50, max_iterations=100, w_min=0.4, w_max=0.9, c1=2, c2=2)
best_solution, best_fitness = apso.optimize()
print("APSO Best solution:", best_solution)
print("APSO Best fitness:", best_fitness)

bpso = BPSO(tsp_instance, population_size=50, max_iterations=100, w=0.8, c1=2, c2=2)
best_solution, best_fitness = bpso.optimize()
print("BPSO Best solution:", [i + 1 for i, bit in enumerate(best_solution) if bit == 1])
print("BPSO Best fitness:", best_fitness)