# Step 1. Decide input and output arrays.
import numpy as np
a = np.array([[0,0],[0,1], [1,0], [1,1]]) # 4,2
b = np.array([[0],[1], [1], [0]]) # 4,1

# Step 2. Decide number of hidden layers, sizes of hidden and output layers, learning rate.
INPUT_SIZE = 2      # 2 neurons in input layer
HIDDEN_SIZE = 4     # 4 neurons in hidden layer
OUTPUT_SIZE = 1     # 1 neuron in output layer
LEARNING_RATE = 0.2 # η (hyperparameter)

# Step 3. Weights and biases initialization.Step 3. Weights and biases initialization.
def randgen(size):
  return np.random.default_rng().random(size)

w_hidden = randgen((INPUT_SIZE, HIDDEN_SIZE))  # 2,4
b_hidden = randgen((1, HIDDEN_SIZE))           # 1,4
w_output = randgen((HIDDEN_SIZE, OUTPUT_SIZE)) # 4,1
b_output = randgen((1, OUTPUT_SIZE))           # 1,1

# Step 4. Define sigmoid, sigmoid_derivative, and loss function.
def sigmoid(x):
  return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
  s = sigmoid(x)
  return s*(1-s)

def loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Step 5. Define forward propagation function.
def feedforward():
  # hidden layer
  z_hidden = np.dot(a, w_hidden) + b_hidden         # (4x2).(2x4) + (1x4) = (4x4)
  a_hidden = sigmoid(z_hidden)                      # (4x4)

  # output layer
  z_output = np.dot(a_hidden, w_output ) + b_output # (4x4).(4x1) + (1x1) = (4x1)
  a_output = sigmoid(z_output)                      # (4x1)

  return z_hidden, a_hidden, z_output, a_output

# Step 6. Define backpropagation function.
def backprop(z_hidden, a_hidden, z_output, a_output):
  m = a.shape[0]                                       # (4)
  # output layer
  da_output = 2*(a_output - b)/m                       # (4x1)
  dz_output = da_output * sigmoid_derivative(z_output) # (4x1) * (4x1) = (4x1)
  dw_output = np.dot(a_hidden.T , dz_output)           # (4x4).(4x1) = (4x1)
  db_output = np.sum(dz_output, axis=0, keepdims=True) # (4x1) => (1x1)

  # hidden layer
  da_hidden = np.dot(dz_output, w_output.T)            # (4x1).(1x4) = (4x4)
  dz_hidden = da_hidden * sigmoid_derivative(z_hidden) # (4x1) * (4x1) = (4x1)
  dw_hidden = np.dot(a.T, dz_hidden)                   # (2x4).(4x4) = (2x4)
  db_hidden = np.sum(dz_hidden, axis=0, keepdims=True) # (4x4) => (1x4)

  return dw_output, db_output, dw_hidden, db_hidden
 

# Step 6. Define backpropagation function.
def gradient_descent(dc_w_output, dc_b_output, dc_w_hidden, dc_b_hidden):
    global w_output, b_output, w_hidden, b_hidden
    w_output-=(LEARNING_RATE*dc_w_output) 
    b_output-=(LEARNING_RATE*dc_b_output)
    w_hidden-=(LEARNING_RATE*dc_w_hidden)
    b_hidden-=(LEARNING_RATE*dc_b_hidden)

# Step 8. Define training function
def train(iterations, step):
   j = iterations/step
   print("training started...")
   for i in range(iterations+1):
      zh, ah, zo, ao = feedforward()
      dwo, dbo, dwh, dbh = backprop(zh, ah, zo, ao)
      gradient_descent(dwo, dbo, dwh, dbh)
      if i%j ==0 :
         l = loss(b, ao)
         print(f"loss at {i}th iteration: {l:.6f} ")



# Step 9. Define print results function and print the results
def print_results(iterations, step):
   _, _, _, ao = feedforward()
   print("results before training: ")
   for i in range(len(ao)):
    print(f"{a[i][0]} ^ {a[i][1]} = {round(ao[i][0])}")   
   train(iterations, step)
   _, _, _, ao = feedforward()
   print("results after training: ")
   for i in range(len(ao)):
    print(f"{a[i][0]} ^ {a[i][1]} = {round(ao[i][0])}")

print_results(10000, 5)