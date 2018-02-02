import numpy as np

def to_categorical( x, num_classes=None ):
    if num_classes is None:
        num_classes = list( set( x.tolist() ) )
    return ( np.arange( num_classes ) == x[ :, None ] ).astype( np.float32 )

def init_weights( shape, dtype=np.float32 ):
    #TODO allow xavier init
    return np.random.random( shape, dtype )

def init_bias( shape, dtype=np.float32 ):
    return np.zeros( shape, dtype )

def cross_entropy( logits=None, labels=None ):
    assert ( logits is not None and labels is not None ),\
            "Usage: cross_entropy( logits=.., labels=.. )"
    return -1. * np.mean( np.sum( labels * np.log( logits ), axis=1 ) )

def kl_divergence( ):
    #TODO Implement KL Divergence
    pass
