import numpy as np
from Parameter import Parameter
import Node

class NeuralNetwork( object ):

    """Do not modify these. Use the static functions associated"""
    node_list = []
    param_list = []

    def __init__( self, inp_shape, num_classes, hidden_layers, lr=0.01 ):
        self.inp_shape = inp_shape
        self.num_classes = num_classes
        self.hidden_layers = hidden_layers
        self.hidden_layers.append( num_classes )
        self.lr = lr

        NeuralNetwork.clear_param_list()
        self._build_network()
        self.node_list = NeuralNetwork.get_node_list()
        self.param_list = NeuralNetwork.get_param_list()
        assert self.node_list, "Node list is empty"
        assert self.param_list, "Parameter list is empty"


    def _build_network( self ):
        self.X = Parameter( is_placeholder = True, name="X" )
        self.Y = Parameter( is_placeholder = True, name="Y" )

        prev_inp = self.X
        prev_shape = self.inp_shape
        for idx, num_units in enumerate( self.hidden_layers ):
            W = Parameter( np.random.random( ( prev_shape, num_units ) ), requires_grad=True, name="W"+str(idx+1) )
            B = Parameter( np.zeros( ( num_units, ) ), requires_grad=True, name="B"+str(idx+1) )
            if idx == len( self.hidden_layers )-1:
                out = Node.Add( Node.Mul( prev_inp, W ), B )
                out = Node.SoftmaxCrossEnt( out, self.Y )
            else:
                out = Node.ReLU( Node.Add( Node.Mul( prev_inp, W ), B ) )
            prev_inp = out
            prev_shape = num_units
        self.loss = out

    def _forward( self ):
        for node in self.node_list:
            node.forward()
        for param in self.param_list:
            print "Param: {}\n{}\n\n".format( param, param.get_data() )

    def _backward( self ):
        for node in self.node_list[::-1]:
            node.backward()

    def _update_params( self, lr ):
        for param in self.param_list:
            param.update( lr )

    def _optimize( self ):
        self._backward()
        self._update_params( self.lr )

    def fit( self, x, y ):
        self.X.set_data_( x )
        self.Y.set_data_( y )
        self._forward()
        self._optimize()
        loss, pred = self.loss.get_output(), self.loss.get_pred()
        return loss.get_data(), pred.get_data()

    def predict( self, x ):
        self.X.set_data_( x )
        pred = self._forward()
        return pred.get_data()

    @staticmethod
    def get_node_list():
        return NeuralNetwork.node_list[:]

    @staticmethod
    def add_node( node ):
        NeuralNetwork.node_list.append( node )

    @staticmethod
    def clear_node_list():
        NeuralNetwork.node_list = []

    @staticmethod
    def get_param_list():
        return NeuralNetwork.param_list[:]

    @staticmethod
    def add_param( node ):
        NeuralNetwork.param_list.append( node )

    @staticmethod
    def clear_param_list():
        NeuralNetwork.param_list = []
