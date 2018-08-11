import numpy as np
from random import shuffle

def softmax(Z):
    return np.exp(Z)/np.sum(np.exp(Z))

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train  = X.shape[0]
  num_classes = W.shape[1]
  for i in range(num_train):
    Z = np.dot(X[i],W)
    A = softmax(Z)
    loss += -np.log(A[y[i]])
    for j in range(num_classes):
      if j == y[i]:
        dW[:, j] += (-1 + A[y[i]]) * X[i]
      else:
        dW[:, j] += A[y[i]] * X[i]
  
  loss = loss/num_train
  dW = dW/num_train

  loss = loss+reg*np.sum(W*W)
  dW += 2*reg*W
  
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  Z = np.dot(X,W)
  scores = np.exp(Z)/np.sum(np.exp(Z),axis=-1,keepdims=True)
  loss = np.sum(-np.log(scores[np.arange(num_train),y]))/num_train
  
  dTemp = scores
  dTemp[np.arange(num_train),y] = dTemp[np.arange(num_train),y] -1
  dW  = np.dot(X.T,dTemp)
  dW  = dW/num_train

  loss += reg*np.sum(W*W)
  dW += 2*reg*W


  return loss, dW

