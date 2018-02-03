import numpy as np
import string

def to_categorical( x, num_classes=None ):
    if num_classes is None:
        num_classes = len( list( set( x.tolist() ) ) )
    return ( np.arange( num_classes ) == x[ :, None ] ).astype( np.float32 )

def init_weights( shape, dtype=np.float32 ):
    #TODO allow xavier init
    return np.random.random( shape, dtype )

def init_bias( shape, dtype=np.float32 ):
    return np.zeros( shape, dtype )

def cross_entropy( logits=None, labels=None ):
    assert ( logits is not None and labels is not None ),\
            "Usage: cross_entropy( logits=.., labels=.. )"
    if np.any( logits == 0 ):
        logits += np.random.random()
    return -1. * np.mean( np.sum( labels * np.log( logits ), axis=1 ) )

def rand_str( N=4 ):
    chars = list( string.ascii_uppercase + string.digits ) 
    return ''.join( np.random.choice( chars, size=N ) )

