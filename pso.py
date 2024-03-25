import math
import random
import numpy as np

# Read the data from the att48.tsp file
def read_tsp_data(filename):
    with open(filename) as f:
        lines = f.readlines()
        node_coord_start = lines.index('NODE_COORD_SECTION\n')
        node_coord_end = lines.index('EOF\n')
        
        nodes = []
        for line in lines[node_coord_start+1:node_coord_end]:
            node_id, x, y = line.split()
            nodes.append((float(x), float(y)))
        
        return nodes

# Calculate the Euclidean distance between two nodes
def distance(node1, node2):
    return math.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)

# Calculate the total distance of a tour
def total_distance(nodes, tour):
    total_dist = 0
    for i in range(len(tour)):
        node1 = nodes[tour[i]]
        node2 = nodes[tour[(i+1) % len(tour)]]
        total_dist += distance(node1, node2)
    return total_dist

# PSO algorithm
def pso_tsp(nodes, num_particles, max_iterations, c1, c2, w):
    num_nodes = len(nodes)
    
    # Initialize particles' positions and velocities
    particles = np.random.permutation(num_nodes).reshape((num_particles, num_nodes))
    velocities = np.zeros((num_particles, num_nodes))
    
    # Initialize pbest and gbest
    pbest = particles.copy()
    pbest_fitness = np.array([total_distance(nodes, tour) for tour in pbest])
    gbest_index = np.argmin(pbest_fitness)
    gbest = pbest[gbest_index].copy()
    gbest_fitness = pbest_fitness[gbest_index]
    
    for _ in range(max_iterations):
        for i in range(num_particles):
            # Update velocities
            r1 = np.random.rand(num_nodes)
            r2 = np.random.rand(num_nodes)
            velocities[i] = w * velocities[i] + c1 * r1 * (pbest[i] - particles[i]) + c2 * r2 * (gbest - particles[i])
            
            # Update positions
            particles[i] = (particles[i] + velocities[i]).argsort()
            
            # Update pbest and gbest
            fitness = total_distance(nodes, particles[i])
            if fitness < pbest_fitness[i]:
                pbest[i] = particles[i].copy()
                pbest_fitness[i] = fitness
                if fitness < gbest_fitness:
                    gbest = pbest[i].copy()
                    gbest_fitness = fitness
    
    return gbest, gbest_fitness

# Read the data
nodes = read_tsp_data('att48.tsp')

# Set the PSO parameters
num_particles = 50
max_iterations = 1000
c1 = 1.49618
c2 = 1.49618
w = 0.7298

# Run the PSO algorithm
best_tour, best_distance = pso_tsp(nodes, num_particles, max_iterations, c1, c2, w)

# Print the results
print("Best tour:", best_tour)
print("Best distance:", best_distance)
