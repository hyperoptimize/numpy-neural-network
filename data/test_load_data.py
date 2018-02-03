from load_data import Dataset
import numpy as np
import os
np.random.seed( 1729 )


if __name__ == '__main__':
    data = Dataset( os.getcwd() )
    
    """MNIST params"""
    num_examples = 60000
    inp_shape = 784
    num_classes = 10
    assert( data.num_examples() == num_examples )
    assert( data.inp_shape() == inp_shape )
    assert( data.num_classes() == num_classes )

    for _ in range( 10000 ):
        batch_size = np.random.randint( low=1, high=num_examples )
        batch_x, batch_y = data.next_batch( batch_size )
        assert( batch_x.shape[0] == batch_size )
        assert( batch_y.shape[0] == batch_size )

        assert( batch_x.shape[1] == inp_shape )
