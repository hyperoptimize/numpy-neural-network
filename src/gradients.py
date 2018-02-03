import numpy as np
from activations import sigmoid, softmax
import pdb


def relu_grad( x ):
    """
    Computes the gradient of the ReLU activation function. This is 
    given by
        g'(x) = 1 if x > 0 else 0

    Although the gradient is not defined at x=0, which is the inflection
    point of the curve, we can approximate the gradient to be zero

    Args:
        x (np.ndarray): The input vector to the relu during forward
        propogation of shape (M,N)

    Returns:
        A vector of derivatives of shape (M,N)
    """
    x[ x <= 0 ] = 0
    x[ x > 0 ] = 1
    return x

def sigmoid_grad( x ):
    """
    Computes the derivative of the sigmoid function given by
        \sigma( x ) \odot \sigma( 1 - x )

    Here \odot represents the haddamard product which is an elemen-wise
    multiplication of two vectors

    Args:
        x (np.ndarray): The input vector to the sigmoid during forward
        propogation of shape (M,N)

    Returns:
        A vector of derivatives of shape (M,N)
    """
    sig_x = sigmoid( x )
    return sig_x * ( 1 - sig_x )

def softmax_grad( x ):
    """
    Computes the gradient of the softmax function. The softmax takes
    a vector of inputs and computes the jacobian for the partial
    derivatives of the function with respect to the input
        J_{i,j} = \partial f_i( x ) / \partial x_j

    Args:
        x (np.ndarray): The input vector to the softmax during forward
        propogation of shape (M,)

    Returns:
        A Jacobian of shape (M,M) which contains the partial derivatives
    """

    f_x = np.squeeze( softmax( x ) )
    n = len( f_x )
    mask = np.eye( n, dtype=bool )

    jac = np.zeros( ( n, n ) )

    diag_idx, _ = np.where( mask == True )
    jac[ mask ] = f_x[ diag_idx ] * ( 1 - f_x[ diag_idx ] )

    i_idx, k_idx = np.where( mask == False )
    jac[ ~mask ] = - f_x[ i_idx ] * f_x[ k_idx ]

    return jac
