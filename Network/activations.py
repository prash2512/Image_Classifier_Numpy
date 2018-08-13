from builtins import range
import numpy as np

def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = np.maximum(0,x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    mask = np.ones(cache.shape)
    mask[cache<0] = 0
    dx, x = mask*dout, cache
    
    return dx

def sigmoid_forward(x):
    """
    Computes the forward pass for a layer of sigmoid.

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    cache = x
    out = 1/(1+np.exp(-x))
    return out,cache

def sigmoid_backward(dout,cache):
    """
    Computes the backward pass for a layer of sigmoid

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dout,x = dout, cache
    dx = sigmoid_forward(x)*(1-sigmoid_forward(x))*dout
    return dx 

def tanh_forward(x):
    """
    Computes the forward pass for a layer of sigmoid.

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    cache = x
    out = np.tanh(x)
    return out,cache

def tanh_backward(dout,cache):
    """
    Computes the backward pass for a layer of tanh

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dout,x = dout, cache
    dx = (1-tanh_forward(x)**2)*dout
    return dx 
