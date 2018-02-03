import numpy as np
from utils import cross_entropy
from activations import relu, softmax, sigmoid
from gradients import relu_grad, sigmoid_grad

np.random.seed( 1729 )

class NeuralNetwork( object ):
    def __init__( self, inp_shape, num_classes, hidden_units, lr=0.01, reg_lam=0.001 ):
        """Class for building and training a simple 2 layer feed-forward Neural Network

        The Neural Network consists of 2 hidden layers with ReLU and uses a Softmax 
        Cross-Entropy to compute the loss. The training is done using Stochastic
        Gradient Descent.
        
        Args:
            inp_shape (int): number of input features
            num_classes (int): num of classes in the output layer
            hidden_units (int): number of hidden units in the hidden layers
            lr (float, optional): learning rate, defaults to 0.01
            reg_lam (float, optional): regularization parameter, defaults to 0.001

        """
        self.inp_shape = inp_shape
        self.num_classes = num_classes
        self.hidden_units = hidden_units
        self.lr = lr
        self.reg_lam = reg_lam
        self._build_network()

    def _build_network( self ):
        """Build a 2 hidden layer neural network with ReLU activations
        and a softmax output layer"""

        self.W1 = np.random.random( ( self.inp_shape, self.hidden_units ) )
        self.b1 = np.zeros( ( self.hidden_units, ) )
        self.hid1 = lambda x: relu( np.dot( x, self.W1 ) + self.b1 )

        self.W2 = np.random.random( ( self.hidden_units, self.hidden_units ) )
        self.b2 = np.zeros( ( self.hidden_units, ) )
        self.hid2 = lambda x: relu( np.dot( x, self.W2 ) + self.b2 )

        self.W3 = np.random.random( ( self.hidden_units, self.num_classes ) )
        self.b3 = np.zeros( ( self.num_classes, ) )
        self.hid3 = lambda x: softmax( np.dot( x, self.W3 ) + self.b3 )

    def _forward( self ):
        """Perform the forward pass"""
        self.a1 = self.hid1( self.X )
        self.a2 = self.hid2( self.a1 )
        self.pred = self.hid3( self.a2 )
        return self.pred

    def _compute_loss( self ):
        """Compute the loss function"""
        loss = cross_entropy( logits=self.pred, labels=self.Y )
        loss += self.reg_lam / 2 * np.linalg.norm( self.W1 )**2 
        loss += self.reg_lam / 2 * np.linalg.norm( self.W2 )**2
        loss += self.reg_lam / 2 * np.linalg.norm( self.W3 )**2
        return loss

    def _backward( self ):
        """Perform the backward pass"""
        delta3 = - ( self.Y - self.pred )
        self.deltaW3 = np.dot( self.a2.T, delta3 )
        self.deltaW3 += self.reg_lam * self.W3
        self.deltab3 = np.dot( delta3.T, np.ones( ( self.num_examples, ) ) )

        """"delta2 = delta3 . W3.T * g'(z2)"""
        z2 = np.dot( self.a1, self.W2 ) + self.b2
        delta2 = delta3.dot( self.W3.T ) * relu_grad( z2 )
        self.deltaW2 = np.dot( self.a2.T, delta2 )
        self.deltaW2 += self.reg_lam * self.W2
        self.deltab2 = np.sum( delta2, axis=0 )

        """"delta1 = delta2 . W2.T * g'(z1)"""
        z1 = np.dot( self.X, self.W1 ) + self.b1
        delta1 = delta2.dot( self.W2.T ) * relu_grad( z1 )
        self.deltaW1 = np.dot( self.X.T, delta1 )
        self.deltaW1 += self.reg_lam * self.W1
        self.deltab1 = np.sum( delta1, axis=0 )

    def _update_params( self ):
        """Update the parameters of the network"""
        updateW1 = - ( self.lr / self.num_examples ) * self.deltaW1
        assert( np.any( self.W1 != updateW1 ) )
        self.W1 += updateW1

        updateW2 = - ( self.lr / self.num_examples ) * self.deltaW2
        assert( np.any( self.W2 != updateW2 ) )
        self.W2 += updateW2

        updateW3 = - ( self.lr / self.num_examples ) * self.deltaW3
        assert( np.any( self.W3 != updateW3 ) )
        self.W3 += updateW3

        updateb1 = - ( self.lr / self.num_examples )* self.deltab1
        assert( np.any( self.b1 != updateb1 ) )
        self.b1 += updateb1

        updateb2 = - ( self.lr / self.num_examples ) * self.deltab2
        assert( np.any( self.b2 != updateb2 ) )
        self.b2 += updateb2

        updateb3 = - ( self.lr / self.num_examples ) * self.deltab3
        assert( np.any( self.b3 != updateb3 ) )
        self.b3 += updateb3

    def _clip_grads( self, threshold=60. ):
        """Clip gradients which have a very high norm"""
        norm = np.linalg.norm
        cutoff = lambda grad: threshold / norm( grad ) if norm( grad ) > threshold else 1
        self.deltaW1 *= cutoff( self.deltaW1 )
        self.deltaW2 *= cutoff( self.deltaW2 )
        self.deltaW3 *= cutoff( self.deltaW3 )

        self.deltab1 *= cutoff( self.deltab1 )
        self.deltab2 *= cutoff( self.deltab2 )
        self.deltab3 *= cutoff( self.deltab3 )

    def _optimize( self ):
        """Perform gradient descent using backpropagation"""
        self._backward()
        self._clip_grads() #ReLU seems to cause exploding grads
        self._update_params()

    def _preprocess( self, whiten=False ):
        """Preprocess input either using whitening or
        using normalization"""
        if whiten:
            mu = np.mean( self.X, axis=0, keepdims=True )
            sigma = np.std( self.X, axis=0, keepdims=True )
            sigma += np.random.random()
            self.X = ( self.X - mu ) / sigma
        else:
            self.X = self.X / np.linalg.norm( self.X )

    def fit( self, x, y ):
        self.X = x
        self.num_examples = self.X.shape[0]
        self._preprocess()
        self.Y = y
        pred = self._forward()
        loss = self._compute_loss()
        self._optimize()
        return loss

    def predict( self, x ):
        self.X = x
        self._preprocess()
        pdb.set_trace()
        pred = self._forward()
        return np.argmax( pred, axis=1 )

    def accuracy( self ):
        pred = np.argmax( self.pred, axis=1 )
        target = np.argmax( self.Y, axis=1 )
        n = pred.shape[0]
        acc = 1. * np.sum( ( pred == target ).astype( np.int32 ) ) / n
        return acc
