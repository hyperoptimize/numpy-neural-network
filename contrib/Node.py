from Parameter import Parameter
import NeuralNetwork
from gradients import cross_entropy_grad, relu_grad, softmax_grad
from activations import relu, softmax
from utils import cross_entropy, rand_str
import numpy as np

class Node():
    
    def __init__( self ):
        self.is_term = False

    def forward( self ):
        pass

    def backward( self ):
        pass

    def get_output( self ):
        pass

class Add(Node):
    
    def __init__( self, A, B ):

        self.node_name = "Add{}".format( rand_str() )
        
        if type( A ) is not Parameter:
            A = A.get_output()
        self.A = A

        if type( B ) is not Parameter:
           B = B.get_output()
        self.B = B

        NeuralNetwork.NeuralNetwork.add_node( self )
        NeuralNetwork.NeuralNetwork.add_param( self.A )
        NeuralNetwork.NeuralNetwork.add_param( self.B )

        self.out = Parameter( is_placeholder=True, name=self.node_name+"_out" )

    def forward( self ):
        res = self.A.get_data() + self.B.get_data()
        self.out.set_data_( res )
        return self.out
    
    def get_output( self ):
        return self.out
    
    def backward( self ):
        inc_grad = self.out.get_grad()
        self.A.set_grad_( inc_grad )
        self.B.set_grad_( inc_grad )
        return 

    def __str__( self ):
        return self.node_name
    
class Mul(Node):
    
    def __init__( self, A, B ):

        self.node_name = "Mul{}".format( rand_str() )
        
        if type( A ) is not Parameter:
            A = A.get_output()
        self.A = A

        if type( B ) is not Parameter:
           B = B.get_output()
        self.B = B

        NeuralNetwork.NeuralNetwork.add_node( self )
        NeuralNetwork.NeuralNetwork.add_param( self.A )
        NeuralNetwork.NeuralNetwork.add_param( self.B )

        self.out = Parameter( is_placeholder=True, name=self.node_name+"_out" )


    def forward( self ):
        res = self.A.get_data().dot(self.B.get_data())
        self.out.set_data_( res )
        return self.out
    
    def get_output( self ):
        return self.out
    
    def backward( self ):
        inc_grad = self.out.get_grad()
        self.A.set_grad_( inc_grad.dot(self.B.get_data().T ))
        self.B.set_grad_( self.A.get_data().T.dot(inc_grad ))
        return 

    def __str__( self ):
        return self.node_name

class ReLU(Node):
    
    def __init__( self, A ):

        self.node_name = "ReLU{}".format( rand_str() )
        
        if type( A ) is not Parameter:
            A = A.get_output()
        self.A = A

        NeuralNetwork.NeuralNetwork.add_node( self )
        NeuralNetwork.NeuralNetwork.add_param( self.A )

        self.out = Parameter( is_placeholder=True, name=self.node_name+"_out" )

    def forward( self ):
        res = relu( self.A.get_data() )
        self.out.set_data_( res )
        return self.out
    
    def get_output( self ):
        return self.out
    
    def backward( self ):
        inc_grad = self.out.get_grad()
        local_grad = relu_grad(self.A.get_data())
        self.A.set_grad_(inc_grad * local_grad)
        return 

    def __str__( self ):
        return self.node_name

class SoftmaxCrossEnt(Node):
    
    def __init__( self, logits=None, labels=None ):

        assert ( logits is not None and labels is not None ),\
                "Usage: cross_entropy( logits=.., labels=.. )"

        self.node_name = "Loss{}".format( rand_str() )
        
        if type(logits) is not Parameter:
            logits = logits.get_output()
        self.logits = logits

        if type(labels) is not Parameter:
            labels = labels.get_output()
        self.labels = labels
        
        NeuralNetwork.NeuralNetwork.add_node( self )
        NeuralNetwork.NeuralNetwork.add_param( self.logits )
        NeuralNetwork.NeuralNetwork.add_param( self.labels )

        self.out = Parameter( is_placeholder=True, name=self.node_name+"_out" )
        self.pred = Parameter( is_placeholder=True, name=self.node_name+"_pred" )

    def forward( self ):
        logits = self.logits.get_data()
        labels = self.labels.get_data()
        pred = softmax( logits )
        res = cross_entropy( logits=pred, labels=labels )
        self.out.set_data_( res )
        self.pred.set_data_( pred )
        return self.out
    
    def get_output( self ):
        return self.out
    
    def backward( self ):
        logits = self.logits.get_data()
        labels = self.labels.get_data()
        self.logits.set_grad_( labels - logits )
        return 

    def __str__( self ):
        return self.node_name

    def get_pred( self ):
        """Only for Softmax cross entropy"""
        return self.pred
