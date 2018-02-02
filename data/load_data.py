'''Adapted from TensorFlow's mnist.py''' 
import numpy as np
import os.path

class Dataset:
    def __init__( self, train_file, test_file ):
        #If numpy file doesn't exist, then only load
        if( os.path.isfile( "mnist.npz" ) ):
            mnist_data = np.load( "mnist.npz" )
            self.train = ( mnist_data["train_x"], mnist_data["train_y"] )
            self.test = ( mnist_data["test_x"], mnist_data["test_y"] )
        else:
            self.train = self._read_csv( train_file )
            self.test = self._read_csv( test_file )
            self.train = ( self.train[:,1:], self.train[:,0] )
            self.test = ( self.test[:,1:], self.test[:,0] )
            np.savez_compressed( "mnist.npz",
                    train_x=self.train[0],
                    train_y=self.train[1],
                    test_x=self.test[0],
                    test_y=self.test[1] )

        self.x, self.y = self.train

        self.num_examples = self.x.shape[0]
        self.index = 0
        self.epochs = 0

    def _read_csv( self, csv_file ):
        ''' Read CSV file and return numpy array'''
        with open( csv_file, "rb" ) as f_handle:
            arr = []
            for line in f_handle:
                l = line.strip().split(",")
                arr.append( np.asarray( l, dtype=np.int32 ) )
        return np.asarray( arr )

    def num_examples( self ):
        return self.num_examples

    def num_iters( self, batch_size ):
        return np.ceil( self.num_examples / batch_size ).astype( np.int32 )

    def input_shape( self ):
        if len( self.x.shape ) == 2:
            return self.x.shape[0]
        return 1

    def num_classes( self ):
        return len( np.unique( self.y ) )

    def next_batch( self, batch_size ):
        #Support shuffle
        start_idx = self.index
        if( start_idx + batch_size > self.num_examples ):
            self.epochs += 1

            #Get rest of examples from current epoch
            rest_ex = self.num_examples - start_idx
            x_rest = self.x[start_idx:self.num_examples]
            y_rest = self.y[start_idx:self.num_examples]

            #Get remaining examples from next epoch
            start_idx = 0
            end_idx = batch_size - rest_ex
            x_remain = self.x[start_idx:end_idx]
            y_remain = self.y[start_idx:end_idx]

            self.index = end_idx
            return np.concatenate( ( x_rest, x_remain ), axis=0 ), np.concatenate( ( y_rest, y_remain ), axis=0 )
        else:
            end_idx = start_idx + batch_size
            x = self.x[start_idx:end_idx]
            y = self.y[start_idx:end_idx]
            return ( x, y )


def load_data():
    dataset = Dataset( "./mnist_train.csv", "./mnist_test.csv" )
    return dataset

test = False
if( test ):
    mnist = load_data()
    batch_size = 1000
    print mnist.num_iters( batch_size )
    print mnist.input_shape()
    print mnist.num_classes()
    for _ in range( 1000 ):
        batch_x, batch_y = mnist.next_batch( batch_size )
        assert( batch_x.shape[0] == batch_size )
        assert( batch_y.shape[0] == batch_size )
