'''Adapted from TensorFlow's mnist.py''' 
import numpy as np
import os
np.random.seed( 1729 )

class Dataset:
    def __init__( self, data_dir, shuffle=True ):
        #TODO write a generator which does a lazy read

        """If numpy file doesn't exist, load only then"""
        data_file = os.path.join( data_dir, "mnist.npz" )
        if( os.path.isfile( data_file ) ):
            mnist_data = np.load( data_file )
            self.train = ( mnist_data["train_x"], mnist_data["train_y"] )
            self.test = ( mnist_data["test_x"], mnist_data["test_y"] )
        else:
            train_file = os.path.join( data_dir, "mnist_train.csv" )
            train_file = os.path.join( data_dir, "mnist_test.csv" )
            self.train = self._read_csv( train_file )
            self.test = self._read_csv( test_file )
            self.train = ( self.train[:,1:], self.train[:,0] )
            self.test = ( self.test[:,1:], self.test[:,0] )
            np.savez_compressed( data_file,
                    train_x=self.train[0],
                    train_y=self.train[1],
                    test_x=self.test[0],
                    test_y=self.test[1] )

        self.x, self.y = self.train

        self.n_examples = self.x.shape[0]
        self.index = 0
        self.epochs = 0

        """Shuffle the input"""
        if shuffle:
            shuffle_idxs = np.arange( self.n_examples )
            np.random.shuffle( shuffle_idxs )
            np.random.shuffle( shuffle_idxs )
            self.x = self.x[shuffle_idxs]
            self.y = self.y[shuffle_idxs]

    def _read_csv( self, csv_file ):
        """Read CSV file and return numpy array"""
        with open( csv_file, "rb" ) as f_handle:
            arr = []
            for line in f_handle:
                l = line.strip().split(",")
                arr.append( np.asarray( l, dtype=np.int32 ) )
        return np.asarray( arr )

    def num_examples( self ):
        return self.n_examples

    def num_iters( self, batch_size ):
        """Iterations required to iterate over the dataset with the given batch size"""
        return np.ceil( self.n_examples / batch_size ).astype( np.int32 )

    def inp_shape( self ):
        if len( self.x.shape ) == 2:
            return self.x.shape[1]
        return 1

    def num_classes( self ):
        return len( np.unique( self.y ) )

    def next_batch( self, batch_size ):
        """Gets the next batch of data, given the batch size"""
        start_idx = self.index
        if( start_idx + batch_size > self.n_examples ):
            self.epochs += 1

            #Get rest of examples from current epoch
            rest_ex = self.n_examples - start_idx
            x_rest = self.x[start_idx:self.n_examples]
            y_rest = self.y[start_idx:self.n_examples]

            #Get remaining examples from next epoch
            start_idx = 0
            end_idx = batch_size - rest_ex
            x_remain = self.x[start_idx:end_idx]
            y_remain = self.y[start_idx:end_idx]

            self.index = end_idx
            x = np.concatenate( ( x_rest, x_remain ), axis=0 )
            y = np.concatenate( ( y_rest, y_remain ), axis=0 )
            return x, y
        else:
            end_idx = start_idx + batch_size
            x = self.x[start_idx:end_idx]
            y = self.y[start_idx:end_idx]
            return ( x, y )
