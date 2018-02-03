import numpy as np
from utils import rand_str

class Parameter( object ):
    def __init__( self, data=None, is_placeholder=False, requires_grad=False, name=None ):
        self.data = data
        assert( type(data) in [ np.ndarray, type(None), np.float32, np.float64, np.int32, np.int64 ] )
        self.requires_grad = requires_grad
        self.is_placeholder = is_placeholder
        self.grad = None
        assert ( data is not None and is_placeholder is False ) or \
            ( data is None and is_placeholder is True ), \
            "Only placholders don't have data"

        self.param_name = rand_str() if not name else name

    def update( self, lr ):

        if self.requires_grad is True:
            if self.data is not None:
                #TODO Fix this hack
                if self.param_name[0] is "B":
                    self.grad = np.sum( self.grad, axis=0 )
                assert( self.data.shape == self.grad.shape ),\
                        "Gradient and data shape don't match: {} and {}".format( self.data.shape, self.grad.shape )
                assert( np.any( self.data != self.grad ) ),\
                        "Parameter has not updated value"
            self.data = self.data - lr * np.sum( self.grad, axis=0 )
        else:
            self.data = None

    def get_grad( self ):
        return self.grad

    def get_data( self ):
        return self.data

    def get_shape( self ):
        if self.data is not None:
            return self.data.shape
        return None

    def set_grad_( self, grad_val ):
        self.grad = grad_val

    def set_data_( self, data_val ):
        self.is_placeholder = False
        self.data = data_val

    def __str__( self ):
        return self.param_name
