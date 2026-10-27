import numpy as np
a = np.array([[0,0],[0,1], [1,1], [1,0]]) # 4,2
b = np.array([[0],[1], [0], [1]]) # 4,1

def randgen(size):
  return np.random.default_rng().random(size)

INPUT_SIZE = 2 # two inputs A and B
HIDDEN_LAYER_SIZE = 4
OUTPUT_LAYER_SIZE = 1
LEARNING_RATE = 0.2

w_hidden = randgen((INPUT_SIZE, HIDDEN_LAYER_SIZE))  # 2,4
b_hidden = randgen((1,4)) # 1,4
w_output = randgen((HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE)) # 4,1
b_output = randgen((OUTPUT_LAYER_SIZE,OUTPUT_LAYER_SIZE)) # 1, 1 

def main():None

def sigmoid(x):
  return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
  s = sigmoid(x)
  return s*(1-s)



def cost(output, expected_output):
  m = expected_output.shape[0]
  return np.square(output-expected_output)/m

def seprate():
  print("-"*30)

def feedforward():
  z_hidden = np.dot(a,w_hidden)+b_hidden
  # print(z_hidden.shape)
  a_hidden = sigmoid(z_hidden)
  # print("shape a hidden: ", a_hidden.shape)

  z_output = np.dot(a_hidden,w_output )+ b_output
  # print("z output shape: ", z_output.shape)
  a_output = sigmoid(z_output)

  return z_hidden, a_hidden, z_output, a_output

def backprop():
  m = a[0].shape 
  dc_a_output = a_output - b # 4x1
  dc_z_output =  np.multiply(dc_a_output , sigmoid_derivative(z_output)) # 4x1

  # z_output = output weights * hidden layer output + output bias
  dc_w_output = np.dot(a_hidden.T, dc_z_output) # 4x1
  # print("shape dc output weighgts: ", dc_w_output.shape)

  dz_output_b_output = 1
  dc_b_output = np.sum(dc_z_output, axis=0, keepdims=True)/m # 1x1
  # print("shape of dc output bias: ", dc_b_output.shape) 


  dc_a_hidden = np.dot(dc_z_output,w_output.T) # 4x1 1x4 4x4

  dc_z_hidden = dc_a_hidden*sigmoid_derivative(z_hidden) # 4x4

  # z_hidden = input * hidden weights + hidden bias
  dc_w_hidden = np.dot(a.T, dc_z_hidden)/m # 2x4 4x4 2x4

  dc_b_hidden = np.sum(dc_z_hidden, axis=0, keepdims=True)/m # 1x4

  return  dc_w_hidden, dc_b_hidden, dc_w_output, dc_b_output

def gradient_descent(dc_w_hidden, dc_b_hidden, dc_w_output, dc_b_output):
    global w_output, b_output, w_hidden, b_hidden
    w_output-=(LEARNING_RATE*dc_w_output) 
    b_output-=(LEARNING_RATE*dc_b_output)
    w_hidden-=(LEARNING_RATE*dc_w_hidden)
    b_hidden-=(LEARNING_RATE*dc_b_hidden)


z_hidden, a_hidden, z_output, a_output = feedforward()



print("cost before training: \n", cost(a_output, b), '\n', np.round(a_output))
for i in range(10000):
  z_hidden, a_hidden, z_output, a_output = feedforward()  
  dwh, dbh, dwo, dbo = backprop()
  gradient_descent(dwh, dbh, dwo, dbo)
_, _, _, a_output = feedforward()
print("cost after training: \n", cost(a_output, b), '\n', np.round(a_output))
