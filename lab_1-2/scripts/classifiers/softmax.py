from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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
    # Implementation: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    for i in range(X.shape[0]):
        scores = X[i] @ W
        scores -= scores.max()
        log_probs = scores - np.log(np.exp(scores).sum())
        loss -= log_probs[y[i]]
        probs = np.exp(log_probs)
        for j in range(W.shape[1]):
            dW[:, j] += (probs[j] - (j == y[i])) * X[i]
    loss = loss / X.shape[0] + reg * np.sum(W * W)
    dW = dW / X.shape[0] + 2 * reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # Implementation: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    scores = X @ W
    scores -= scores.max(axis=1, keepdims=True)
    log_probs = scores - np.log(np.exp(scores).sum(axis=1, keepdims=True))
    loss = -log_probs[np.arange(len(X)), y].mean() + reg * np.sum(W * W)
    probs = np.exp(log_probs)
    probs[np.arange(len(X)), y] -= 1
    dW = X.T @ probs / len(X) + 2 * reg * W

    return loss, dW
