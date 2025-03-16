import sys
import os
sys.path.append(os.path.abspath('_given'))

import numpy as np
from matplotlib import pyplot as plt
from utils import part1CreateDataset, part1PlotBoundary


class MLP:
  def __init__(self, input_size, hidden_size, output_size):
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size

    # Initialize weights and biases
    self.weights_input_hidden = np.random.randn(self.input_size + 1, self.hidden_size) # 3x4 matrix
    #self.bias_hidden = np.zeros((1 , self.hidden_size)) # 1x4

    self.weights_hidden_output = np.random.randn(self.hidden_size + 1, self.output_size) # 5x1 matrix
    #self.bias_output = np.zeros((1 , self.output_size)) # 1x1


  def f_activation(self, x):
    return (np.exp(2*x)-1)/(np.exp(2*x)+1)  # Tanh
    #return 1 / (1 + np.exp(-x))             # Sigmoid
    #return np.maximum(0, x)                  # ReLU

  def f_activation_derivative(self, x):
    return (1 - np.square(x))                          # Tanh
    #return x * (1 - x)                      # Sigmoid
    #return (x>0).astype(float)               # ReLU

  def forward(self, inputs):
    # Forward pass through the network
    inputs = np.hstack((np.ones((inputs.shape[0], 1)), inputs)) # Kx3 matrix, first column is 1
    self.hidden_output =  self.f_activation(np.matmul( inputs , self.weights_input_hidden)) # Kx4

    self.hidden_output_w_bias = np.hstack((np.ones((self.hidden_output.shape[0], 1)), self.hidden_output)) # Kx5 matrix, first column is 1
    self.output = self.f_activation(np.matmul(self.hidden_output_w_bias , self.weights_hidden_output)) # Kx1
    return self.output


  def backward(self, inputs, targets, learning_rate):
    # Backward pass through the network

    # Compute output layer error and its gradient
    output_error = (self.output - targets) * 2.0 / targets.size  # Kx1
    output_delta = output_error * self.f_activation_derivative(self.output) * 2.0 # Kx1

    # Compute hidden layer error and its gradient
    hidden_error = np.matmul(output_error , self.weights_hidden_output[1:].T) # Kx4
    hidden_delta = hidden_error * self.f_activation_derivative(self.hidden_output) # Kx4

    inputs = np.hstack((np.ones((inputs.shape[0], 1)), inputs)) # Kx3 matrix, first column is 1

    # Update weights and biases
    self.weights_hidden_output -= learning_rate * np.matmul( self.hidden_output_w_bias.T , output_delta) # 5x1
    self.weights_input_hidden -= learning_rate * np.matmul( inputs.T , hidden_delta) # 3x4 matrix


# Generate the dataset
x_train, y_train, x_val, y_val = part1CreateDataset(train_samples=1000, val_samples=100, std=0.4)

# Define neural network parameters
input_size = 2
hidden_size = 4
output_size = 1
learning_rate = 0.001

# Create neural network
nn = MLP(input_size, hidden_size, output_size)

# Train the neural network
for epoch in range(50000):

  # Forward propagation
  output = nn.forward(x_train)

  # Backpropagation
  nn.backward(x_train, y_train, learning_rate)

  # Print the loss (MSE) every 1000 epochs
  if epoch % 1000 == 0:
    # Compute predictions for the entire validation set
    val_predictions = nn.forward(x_val)
    loss = 0.5 * np.mean(np.square(y_val - val_predictions))
    print(f'Epoch {epoch}: Loss = {loss}')


# Test the trained neural network
val_predictions = nn.forward(x_val)

y_predict = ((val_predictions > 0.5).astype(int)).reshape(-1,1)
accuracy = np.mean((y_predict == y_val).astype(int))
print(f'{accuracy*100}% of test examples classified correctly.')

part1PlotBoundary(x_val, y_val, nn)