import numpy as np
import random


class BinaryPSO:
    def __init__(self, distances, num_particles, max_iterations):
        self.distances = distances
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.num_cities = len(distances)
        self.particles = self.initialize_particles()
        self.pbest_positions = self.particles.copy()
        self.pbest_fitness = np.full(num_particles, float('inf'))
        self.gbest_position = None
        self.gbest_fitness = float('inf')

    def initialize_particles(self):
        particles = []
        for _ in range(self.num_particles):
            particle = np.random.binomial(1, 0.5, size=(self.num_cities, self.num_cities))
            particles.append(particle)
        return np.array(particles)

    def calculate_fitness(self, particle):
        tour = self.decode_particle(particle)
        fitness = 0
        for i in range(len(tour) - 1):
            fitness += self.distances[tour[i]][tour[i + 1]]
        fitness += self.distances[tour[-1]][tour[0]]
        return fitness

    def decode_particle(self, particle):
        tour = []
        remaining_cities = list(range(self.num_cities))
        while len(remaining_cities) > 0:
            if len(tour) > 0:
                probabilities = particle[tour[-1], remaining_cities]
            else:
                probabilities = particle[0, remaining_cities]
            probabilities = probabilities.astype(float)  # Convert to float
            prob_sum = np.sum(probabilities)
            if prob_sum > 0:
                probabilities /= prob_sum
            else:
                # If the sum of probabilities is zero, assign equal probabilities
                probabilities = np.ones_like(probabilities) / len(probabilities)
            next_city = np.random.choice(remaining_cities, p=probabilities)
            tour.append(next_city)
            remaining_cities.remove(next_city)
        return tour

    def update_pbest(self):
        for i in range(self.num_particles):
            fitness = self.calculate_fitness(self.particles[i])
            if fitness < self.pbest_fitness[i]:
                self.pbest_positions[i] = self.particles[i]
                self.pbest_fitness[i] = fitness

    def update_gbest(self):
        best_index = np.argmin(self.pbest_fitness)
        if self.pbest_fitness[best_index] < self.gbest_fitness:
            self.gbest_position = self.pbest_positions[best_index]
            self.gbest_fitness = self.pbest_fitness[best_index]

    def update_velocities(self, c1, c2):
        r1 = np.random.rand(self.num_particles, self.num_cities, self.num_cities)
        r2 = np.random.rand(self.num_particles, self.num_cities, self.num_cities)
        velocities = c1 * r1 * (self.pbest_positions - self.particles) + \
                     c2 * r2 * (self.gbest_position - self.particles)
        return velocities

    def update_positions(self, velocities):
        self.particles = np.clip(self.particles + velocities, 0, 1)
        self.particles = np.where(self.particles > np.random.rand(*self.particles.shape), 1, 0)

    def run(self, c1=2, c2=2):
        for _ in range(self.max_iterations):
            self.update_pbest()
            self.update_gbest()
            velocities = self.update_velocities(c1, c2)
            self.update_positions(velocities)

        best_tour = self.decode_particle(self.gbest_position)
        best_distance = self.gbest_fitness
        return best_tour, best_distance


# Example usage
distances = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]

num_particles = 20
max_iterations = 100

pso = BinaryPSO(distances, num_particles, max_iterations)
best_tour, best_distance = pso.run()

print("Best tour:", best_tour)
print("Best distance:", best_distance)