import numpy as np

# 1. Activation function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# 2. XOR dataset (4 samples, 2 inputs each)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]) # Shape: (4, 2)

y = np.array([
    [0],
    [1],
    [1],
    [0]
]) # Shape: (4, 1)

# 3. Initialize weights and biases randomly
np.random.seed(42) # For reproducible random values

# Input layer (2) -> Hidden layer (4)
W1 = np.random.randn(2, 4)  # Shape: (2, 4)
b1 = np.zeros((1, 4))       # Shape: (1, 4)

# Hidden layer (4) -> Output layer (1)
W2 = np.random.randn(4, 1)  # Shape: (4, 1)
b2 = np.zeros((1, 1))       # Shape: (1, 1)

# ----------------------------------------------------
# 4. THE FEEDFORWARD PASS
# ----------------------------------------------------

# Step A: Linear combination for hidden layer
# (4, 2) @ (2, 4) + (1, 4)  -->  Shape: (4, 4)
z_hidden = np.dot(X, W1) + b1

# Step B: Squashing through activation function
# Shape: (4, 4)
a_hidden = sigmoid(z_hidden)

# Step C: Linear combination for output layer
# (4, 4) @ (4, 1) + (1, 1)  -->  Shape: (4, 1)
z_output = np.dot(a_hidden, W2) + b2

# Step D: Squashing through final activation
# Shape: (4, 1)
a_output = sigmoid(z_output)

# ----------------------------------------------------
# Check the shapes and initial predictions
# ----------------------------------------------------
print("a_output shape:", a_output.shape)
print("Initial predictions (untrained):\n", a_output)

# Derivative of sigmoid: sigma'(z) = a * (1 - a)
def sigmoid_derivative(a):
    return a * (1.0 - a)

# Set learning rate
learning_rate = 0.1
m = X.shape[0]  # Number of samples = 4

# ==========================================
# 1. OUTPUT LAYER BACKPROPAGATION
# ==========================================

# Step A: Loss gradient wrt final output (MSE: 0.5 * (a_output - y)^2)
# Shape: (4, 1)
dL_da_output = a_output - y

# Step B: Local error at output layer (delta_output = dL/dz_output)
# Element-wise multiplication (*) between (4, 1) and (4, 1)
delta_output = dL_da_output * sigmoid_derivative(a_output)  # Shape: (4, 1)

# Step C: Gradients for W2 and b2
# dW2 = (a_hidden)^T @ delta_output -> (4, 4)^T @ (4, 1) -> Shape: (4, 1)
dW2 = np.dot(a_hidden.T, delta_output) / m

# db2 = sum across all batch samples -> Shape: (1, 1)
db2 = np.sum(delta_output, axis=0, keepdims=True) / m


# ==========================================
# 2. PASS ERROR TO HIDDEN LAYER
# ==========================================

# Step A: Loss gradient wrt hidden activations (dL/da_hidden)
# delta_output @ (W2)^T -> (4, 1) @ (1, 4) -> Shape: (4, 4)
dL_da_hidden = np.dot(delta_output, W2.T)

# Step B: Local error at hidden layer (delta_hidden = dL/dz_hidden)
# Element-wise multiplication (*) between (4, 4) and (4, 4)
print(sigmoid_derivative(a_hidden).shape)
delta_hidden = dL_da_hidden * sigmoid_derivative(a_hidden)  # Shape: (4, 4)

# Step C: Gradients for W1 and b1
# dW1 = (X)^T @ delta_hidden -> (2, 4)^T @ (4, 4) -> (2, 4)
dW1 = np.dot(X.T, delta_hidden) / m

# db1 = sum across batch samples -> Shape: (1, 4)
db1 = np.sum(delta_hidden, axis=0, keepdims=True) / m


# ==========================================
# 3. UPDATE WEIGHTS AND BIASES (Gradient Descent)
# ==========================================
W2 = W2 - learning_rate * dW2
b2 = b2 - learning_rate * db2

W1 = W1 - learning_rate * dW1
b1 = b1 - learning_rate * db1