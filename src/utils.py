import numpy as np
import string
np.random.seed( 1729 )

def to_categorical( x, num_classes=None ):
    """Convert a label vector into categorical labels
    If the num of classes is not given, it is inferred
    from the data"""
    if num_classes is None:
        num_classes = len( list( set( x.tolist() ) ) )
    return ( np.arange( num_classes ) == x[ :, None ] ).astype( np.float32 )

def init_weights( shape, dtype=np.float32 ):
    """Initialize weights of a neural network randomly"""
    #TODO allow xavier init
    return np.random.random( shape, dtype )

def init_bias( shape, dtype=np.float32 ):
    """Initialize biases of a neural network randomly"""
    #TODO allow xavier init
    return np.zeros( shape, dtype )

def cross_entropy( logits=None, labels=None ):
    """Compute the cross entropy error between the
    two output distributions"""
    #TODO ensure only kwargs are passed
    assert ( logits is not None and labels is not None ),\
            "Usage: cross_entropy( logits=.., labels=.. )"
    #To avoid divide by zero error, we can add a small value
    if np.any( logits == 0 ):
        logits += np.random.random()
    return -1. * np.mean( np.sum( labels * np.log( logits ), axis=1 ) )

def rand_str( N=4 ):
    """Generate a random string of N characters"""
    chars = list( string.ascii_uppercase + string.digits ) 
    return ''.join( np.random.choice( chars, size=N ) )
