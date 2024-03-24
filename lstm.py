import numpy as np
import pandas as pd

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Hyperbolic tangent activation function and its derivative
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - x ** 2

# LSTM class
class LSTM:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wy = np.random.randn(output_size, hidden_size)
        
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
    
    def forward(self, x, h_prev, c_prev):
        # Concatenate input and previous hidden state
        concat = np.concatenate((x, h_prev), axis=0)
        
        # Compute gate activations
        f = sigmoid(np.dot(self.Wf, concat) + self.bf)
        i = sigmoid(np.dot(self.Wi, concat) + self.bi)
        c_bar = tanh(np.dot(self.Wc, concat) + self.bc)
        o = sigmoid(np.dot(self.Wo, concat) + self.bo)
        
        # Compute cell state and hidden state
        c = f * c_prev + i * c_bar
        h = o * tanh(c)
        
        # Compute output
        y = sigmoid(np.dot(self.Wy, h) + self.by)
        
        # Store activations for backpropagation
        self.x = x
        self.h_prev = h_prev
        self.c_prev = c_prev
        self.f = f
        self.i = i
        self.c_bar = c_bar
        self.o = o
        self.c = c
        self.h = h
        self.y = y
        
        return y, h, c
    
    def backward(self, dy):
        # Compute gradients
        dWy = np.dot(dy, self.h.T)
        dby = dy
        dh = np.dot(self.Wy.T, dy)
        
        dc = dh * self.o * tanh_derivative(self.c)
        dc_bar = dc * self.i
        di = dc * self.c_bar * sigmoid_derivative(self.i)
        df = dc * self.c_prev * sigmoid_derivative(self.f)
        do = dh * tanh(self.c) * sigmoid_derivative(self.o)
        
        concat = np.concatenate((self.x, self.h_prev), axis=0)
        dWf = np.dot(df, concat.T)
        dWi = np.dot(di, concat.T)
        dWc = np.dot(dc_bar, concat.T)
        dWo = np.dot(do, concat.T)
        
        dbf = df
        dbi = di
        dbc = dc_bar
        dbo = do
        
        # Update weights and biases
        self.Wf -= self.learning_rate * dWf
        self.Wi -= self.learning_rate * dWi
        self.Wc -= self.learning_rate * dWc
        self.Wo -= self.learning_rate * dWo
        self.Wy -= self.learning_rate * dWy
        
        self.bf -= self.learning_rate * dbf
        self.bi -= self.learning_rate * dbi
        self.bc -= self.learning_rate * dbc
        self.bo -= self.learning_rate * dbo
        self.by -= self.learning_rate * dby
        
        # Compute gradients for previous hidden and cell states
        dh_prev = np.dot(self.Wf[:, self.input_size:].T, df) + np.dot(self.Wi[:, self.input_size:].T, di) + \
                  np.dot(self.Wc[:, self.input_size:].T, dc_bar) + np.dot(self.Wo[:, self.input_size:].T, do)
        dc_prev = dc * self.f
        
        return dh_prev, dc_prev

# Generate sample data
data = pd.DataFrame(np.random.rand(100, 5), columns=['feature_' + str(i) for i in range(5)])
data['target'] = np.random.randint(0, 2, size=(100,))

# Prepare the input data
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)

# Initialize LSTM parameters
input_size = X.shape[1]
hidden_size = 32
output_size = 1
learning_rate = 0.01
num_epochs = 10

# Create LSTM instance
lstm = LSTM(input_size, hidden_size, output_size, learning_rate)

# Training loop
for epoch in range(num_epochs):
    h = np.zeros((hidden_size, 1))
    c = np.zeros((hidden_size, 1))
    
    for i in range(len(X)):
        x = X[i].reshape(-1, 1)
        y_true = y[i]
        
        # Forward pass
        y_pred, h, c = lstm.forward(x, h, c)
        
        # Compute loss
        loss = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
        
        # Backward pass
        dy = y_pred - y_true
        dh_prev, dc_prev = lstm.backward(dy)
    
    print(f"Epoch {epoch+1}/{num_epochs} completed")

# Make predictions
test_data = np.random.rand(1, input_size).reshape(-1, 1)
h = np.zeros((hidden_size, 1))
c = np.zeros((hidden_size, 1))
prediction, _, _ = lstm.forward(test_data, h, c)
print("Prediction:", prediction)
