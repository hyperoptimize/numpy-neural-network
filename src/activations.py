"""
Implementation of common activation functions used for neural networks

@author: Utkarsh Simha
@email: usimha@ucsd.edu
"""

import numpy as np

def relu( x, leaky=False ):
    """Computes the ReLU activation function elementwise for a given input
    
    The Rectified Linear Unit (Nair & Hinton, 2010) function is defined as
            
            f(x) = max( 0, x )

    The function also provides the option to compute the leaky ReLU which
    has a small negative slope when x < 0. This is used to fix the "dying
    ReLU problem". The range of values of the output lie between [0,inf) in
    the case of the vanilla ReLU, and (-inf,inf) for the leaky ReLU.

    Args:
	x (numpy.ndarray): Numpy array of dtype `float32`, `float64`,
        `int32`, `int64` with shape (M,N)
        leaky (bool, optional): Flag to use the "leaky ReLU" variant
        Default: False

    Returns:
        numpy.ndarray of shape (M,N) where each element corresponds
        to elementwise application of the ReLU (or leaky ReLU function)

    Raises:
        Some error for range, type?

    """

    neg_slope = 0.01 #Small negative slope
    if leaky:
        z = neg_slope * x
    else:
        z = np.zeros( x.shape )

    #Compute elementwise maximum
    return np.maximum( z, x )

def sigmoid( x ):
    """Computes the sigmoid activation function elementwise for a given input
    
    The Sigmoid activation function is given by 
            
            f(x) = 1 / ( 1 + exp(-x) )

    The range of the output of the sigmoid function lies in [0,1]

    Args:
	x (numpy.ndarray): Numpy array of dtype `float32`, `float64`,
        `int32`, `int64` with shape (M,N)

    Returns:
        numpy.ndarray of shape (M,N) where each element corresponds
        to elementwise application of the sigmoid activation function

    Raises:
        Some error for range, type?

    """
    
    return 1. / ( 1. + np.exp( - x ) )

def softmax( x, axis=-1 ):
    """Computes the softmax activation function across a given axis
    
    The Softmax activation function is given by 

        f(x_i) = exp( x_i ) / \sum_{j} exp( x_j ), for each i

    The Softmax function results in a probability distribution over
    the given axis, which sums to 1. Each element represents a
    probability that lies in [0,1].

    Args:
	x (numpy.ndarray): Numpy array of dtype `float32`, `float64`,
        `int32`, `int64` with shape (M,N)
        axis (int): Axis over which the softmax normalization must
        be performed. Default: -1, which corresponds to last axis

    Returns:
        numpy.ndarray of shape (M,N) where each element corresponds
        to elementwise application of the softmax activation function

    Raises:
        Some error for range, type?

    """
    x = x - x.max()
    exps = np.exp( x )
    exps[ exps == 0 ] += np.random.random()
    norm = np.sum( exps, axis=axis, keepdims=True )
    return exps / norm
    
