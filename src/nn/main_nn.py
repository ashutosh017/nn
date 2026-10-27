import numpy as np

# ---------------------------------------------------------------------
# 1. Dataset: XOR Logic Gate
# ---------------------------------------------------------------------
# Inputs (4 samples, 2 features)
a = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=float)  # shape: (4, 2)
# Targets (4 samples, 1 target output)
b = np.array([[0], [1], [0], [1]], dtype=float)              # shape: (4, 1)


# ---------------------------------------------------------------------
# 2. Activation and Cost Functions
# ---------------------------------------------------------------------
def sigmoid(x):
    """Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(z):
    """Derivative of sigmoid: s * (1 - s)."""
    s = sigmoid(z)
    return s * (1.0 - s)


def cost(output, expected_output):
    """Mean Squared Error (MSE) cost."""
    return np.round(np.mean(np.square(output - expected_output)), 6)


# ---------------------------------------------------------------------
# 3. Hyperparameters and Initialization
# ---------------------------------------------------------------------
INPUT_SIZE = 2          # Number of input features
HIDDEN_LAYER_SIZE = 4   # Number of hidden neurons
OUTPUT_LAYER_SIZE = 1   # Number of output neurons
LEARNING_RATE = 1.0     # Learning rate for gradient descent
EPOCHS = 10000          # Number of training epochs

# Set seed for reproducible initialization
np.random.seed(42)

# Weights & biases initialization
w_hidden = np.random.uniform(-1.0, 1.0, (INPUT_SIZE, HIDDEN_LAYER_SIZE))       # shape: (2, 4)
b_hidden = np.zeros((1, HIDDEN_LAYER_SIZE))                                   # shape: (1, 4)
w_output = np.random.uniform(-1.0, 1.0, (HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE)) # shape: (4, 1)
b_output = np.zeros((1, OUTPUT_LAYER_SIZE))                                   # shape: (1, 1)


# ---------------------------------------------------------------------
# 4. Forward Propagation
# ---------------------------------------------------------------------
def feedforward(inputs=a):
    """
    Computes activations across layers:
      z_hidden: (4, 2) @ (2, 4) + (1, 4) = (4, 4)
      a_hidden: sigmoid(z_hidden)       = (4, 4)
      z_output: (4, 4) @ (4, 1) + (1, 1) = (4, 1)
      a_output: sigmoid(z_output)       = (4, 1)
    """
    z_hidden = np.dot(inputs, w_hidden) + b_hidden
    a_hidden = sigmoid(z_hidden)

    z_output = np.dot(a_hidden, w_output) + b_output
    a_output = sigmoid(z_output)

    return z_hidden, a_hidden, z_output, a_output


# ---------------------------------------------------------------------
# 5. Backward Propagation (Backprop)
# ---------------------------------------------------------------------
def backprop(z_hidden, a_hidden, z_output, a_output):
    """
    Calculates gradients using the chain rule and updates weights and biases.
    """
    global w_output, b_output, w_hidden, b_hidden
    m = len(a)  # Number of training samples (4)

    # 1. Output Layer Gradients
    dc_a_output = 2.0 * (a_output - b) / m
    dc_z_output = dc_a_output * sigmoid_derivative(z_output)
    dc_w_output = np.dot(a_hidden.T, dc_z_output)
    dc_b_output = np.sum(dc_z_output, axis=0, keepdims=True)

    # 2. Hidden Layer Gradients
    dc_a_hidden = np.dot(dc_z_output, w_output.T)
    dc_z_hidden = dc_a_hidden * sigmoid_derivative(z_hidden)
    dc_w_hidden = np.dot(a.T, dc_z_hidden)
    dc_b_hidden = np.sum(dc_z_hidden, axis=0, keepdims=True)

    # 3. Update Weights and Biases (Gradient Descent)
    w_output -= LEARNING_RATE * dc_w_output
    b_output -= LEARNING_RATE * dc_b_output
    w_hidden -= LEARNING_RATE * dc_w_hidden
    b_hidden -= LEARNING_RATE * dc_b_hidden


# ---------------------------------------------------------------------
# 6. Training Loop & Entry Point
# ---------------------------------------------------------------------
def train(epochs=EPOCHS):
    """Runs the training loop and prints training progress and predictions."""
    # Compute initial prediction and cost
    _, _, _, a_output = feedforward()
    print("Cost before training:", cost(a_output, b))

    # Iterative Training Loop: Forward -> Backward -> Update
    for epoch in range(epochs):
        z_hidden, a_hidden, z_output, a_output = feedforward()
        backprop(z_hidden, a_hidden, z_output, a_output)

        if (epoch + 1) % 2000 == 0:
            current_cost = cost(a_output, b)
            print(f"Epoch {epoch + 1:5d}/{epochs} - Cost: {current_cost:.6f}")

    # Compute final prediction and cost
    _, _, _, a_output = feedforward()
    print("\nCost after training: ", cost(a_output, b))

    # Display final predictions
    print("\nPredictions on XOR:")
    for input_sample, target, pred in zip(a, b, a_output):
        print(f"Input: {input_sample.astype(int).tolist()} -> Target: {int(target[0])} -> Pred: {pred[0]:.4f} (Class: {int(pred[0] > 0.5)})")


def main():
    train()


if __name__ == "__main__":
    main()
