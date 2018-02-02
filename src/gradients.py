import numpy as np
from activations import sigmoid, softmax
import pdb


def relu_grad( x ):
    x_prime = x.copy()
    x_prime[ x_prime <= 0 ] = 0
    x_prime[ x_prime > 0 ] = 1
    return x_prime

def sigmoid_grad( x ):
    #TODO This doesn't work for x as a matrix
    return sigmoid( x ) * ( 1 - sigmoid( x ) )

def softmax_grad( x ):
    #x is a vector
    #returns a jacobian matrix
    f_x = np.squeeze( softmax( x ) )
    n = len( f_x )
    mask = np.eye( n, dtype=bool )

    jac = np.zeros( ( n, n ) )

    diag_idx, _ = np.where( mask == True )
    jac[ mask ] = f_x[ diag_idx ] * ( 1 - f_x[ diag_idx ] )

    i_idx, k_idx = np.where( mask == False )
    jac[ ~mask ] = - f_x[ i_idx ] * f_x[ k_idx ]

    return jac

def cross_entropy_grad( logits=None, labels=None ):
    assert ( logits is not None and labels is not None ),\
            "Usage: cross_entropy( logits=.., labels=.. )"
    return labels - logits
    
