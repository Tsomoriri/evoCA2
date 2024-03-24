import numpy as np
import pandas as pd
from lstm import LSTM


# Generate sample TSP data
num_nodes = 10
nodes = np.arange(num_nodes)
x = np.random.rand(num_nodes)
y = np.random.rand(num_nodes)
distances = np.zeros((num_nodes, num_nodes))

for i in range(num_nodes):
    for j in range(num_nodes):
        if i != j:
            distances[i, j] = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)

data = pd.DataFrame({'node': nodes, 'x': x, 'y': y})
data = pd.concat([data, pd.DataFrame(distances)], axis=1)

# Prepare the input data
X = data.values

# Initialize LSTM parameters
input_size = X.shape[1]
hidden_size = 64
output_size = num_nodes
learning_rate = 0.01
num_epochs = 100

# Create LSTM instance
lstm = LSTM(input_size, hidden_size, output_size, learning_rate)

# Training loop
for epoch in range(num_epochs):
    # Generate a random TSP solution as the target
    target_solution = np.random.permutation(num_nodes)
    
    # Convert target solution to one-hot encoding
    target_one_hot = np.zeros((num_nodes, num_nodes))
    target_one_hot[np.arange(num_nodes), target_solution] = 1
    
    # Reset hidden and cell states
    h = np.zeros((hidden_size, 1))
    c = np.zeros((hidden_size, 1))
    
    # Forward pass through the LSTM for each node
    for i in range(num_nodes):
        x = X[i].reshape(-1, 1)
        y_true = target_one_hot[i].reshape(-1, 1)
        
        y_pred, h, c = lstm.forward(x, h, c)
        
        # Compute loss
        loss = -np.sum(y_true * np.log(y_pred))
        
        # Backward pass
        dy = y_pred - y_true
        dh_prev, dc_prev = lstm.backward(dy)
    
    # Print the loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

# Generate a new TSP instance for testing
test_nodes = np.arange(num_nodes)
test_x = np.random.rand(num_nodes)
test_y = np.random.rand(num_nodes)
test_distances = np.zeros((num_nodes, num_nodes))

for i in range(num_nodes):
    for j in range(num_nodes):
        if i != j:
            test_distances[i, j] = np.sqrt((test_x[i] - test_x[j])**2 + (test_y[i] - test_y[j])**2)

test_data = pd.DataFrame({'node': test_nodes, 'x': test_x, 'y': test_y})
test_data = pd.concat([test_data, pd.DataFrame(test_distances)], axis=1)

# Prepare the test data
test_X = test_data.values

# Reset hidden and cell states for testing
h = np.zeros((hidden_size, 1))
c = np.zeros((hidden_size, 1))

# Generate TSP solution using the trained LSTM
tsp_solution = []
for i in range(num_nodes):
    x = test_X[i].reshape(-1, 1)
    y_pred, h, c = lstm.forward(x, h, c)
    next_node = np.argmax(y_pred)
    tsp_solution.append(next_node)

print("TSP Solution:", tsp_solution)
