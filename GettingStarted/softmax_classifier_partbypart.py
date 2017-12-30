#Basic Neural Networks Example
#Sourced from: http://cs231n.github.io/neural-networks-case-study/  Please read this resource long with this example so it makes sense!!!

from __future__ import absolute_import, division, print_function
import tflearn
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels

W = 0.01 * np.random.randn(D,K) # weight matrix
b = np.zeros((1,K)) # bias vector

# some hyperparameters (constants used in the code)
step_size = 1e-0
reg = 1e-3 # regularization strength  #regularization strength constant (lambda) - this is basically applying a penalty to increasing the magnitude of parameter values in order to reduce over-fitting.

for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
# lets visualize the data:
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show() #Plots graph in terminal, close to continue code execution

#Train a Softmax classifier
scores = np.dot(X, W) + b
pprint(scores) #Print out scores matrix [300x3] corresponding to each of the 3 classes, blue, red, and yellow

#Compute the loss with the Softmax classifier
num_examples = X.shape[0]
# get unnormalized probabilities
exp_scores = np.exp(scores)
# normalize them for each example
probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
pprint(probs) #Print out probabilities in matrix [300x3] 

#compute a 1D array of just probabilities assigned to the correct classes for each example.
corect_logprobs = -np.log(probs[range(num_examples),y])
pprint(corect_logprobs) #Print out probabilities in 1D Matrix

# compute the loss: average cross-entropy loss and regularization
data_loss = np.sum(corect_logprobs)/num_examples
reg_loss = 0.5*reg*np.sum(W*W)
loss = data_loss + reg_loss
pprint(loss) #Trying to make loss as close as possible to 0

#Computing the Analytic Gradient with Backpropagation
dscores = probs
dscores[range(num_examples),y] -= 1
dscores /= num_examples
pprint(dscores)

dW = np.dot(X.T, dscores)
db = np.sum(dscores, axis=0, keepdims=True)
dW += reg*W # don't forget the regularization gradient (this is W*reg)

W += -step_size * dW
b += -step_size * db

#Visualize the plot the resulting classifier
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show() #Plots graph in terminal, close to continue code execution