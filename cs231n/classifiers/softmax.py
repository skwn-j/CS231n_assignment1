import numpy as np
from random import shuffle

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_dim = np.shape(W)[0]
  num_class = np.shape(W)[1]
  num_ex = np.shape(X)[0]
  scores = X.dot(W)
  for i in range(num_ex) :
    f = scores[i]
    f -= np.max(f)
    p = np.exp(f) / np.sum(np.exp(f))
    loss += -np.log(p[y[i]])
    for j in range(num_class) :
      if y[i] == j :
        dW[: , j] += -X[i].T*(1-p[j])
      else :
        dW[: , j] += X[i].T*p[j]

  loss /= num_ex
  dW /= num_ex

  loss += reg*np.sum(W * W)
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_ex = np.shape(X)[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  f = X.dot(W)
  max_val = np.amax(f, axis = 1).reshape((num_ex, 1))
  f -= max_val
  p = np.exp(f)/np.sum(np.exp(f), axis=1).reshape((num_ex, 1))
  loss = np.mean(-np.log(p[range(num_ex), y]))
  loss += reg * np.sum(W * W)

  p[range(num_ex), y] -= 1
  dW = X.T.dot(p)

  dW /= num_ex
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

