from data.load_data import Dataset
from src.NeuralNetwork import NeuralNetwork
from src.utils import to_categorical
import numpy as np

DATA_DIR="./data"

def train( data, model, n_epochs=200, batch_size=128 ):
    num_iters = data.num_iters( batch_size )
    loss = []
    avg_loss = 0.
    avg_acc = 0.
    for epoch in range( 1, n_epochs+1 ):
        for _ in range( num_iters ):
            batch_x, batch_y = data.next_batch( batch_size )
            batch_y = to_categorical( batch_y )

            avg_loss += model.fit( batch_x, batch_y )
            avg_acc += model.accuracy()
        if epoch % 10 == 0:
            n = 10. * num_iters
            print "Epoch: {}; Loss: {}".format( epoch, avg_loss / n )
            print "Acc: {}".format( avg_acc / n )
            loss.append( avg_loss )
            avg_loss = 0.
            avg_acc = 0.

    test_x, test_y = data.test
    pred = model.predict( test_x )
    test_acc = np.mean( test_y == pred )
    print "Test accuracy: {}".format( test_acc )

if __name__ == '__main__':
    data = Dataset( DATA_DIR )
    inp_shape, num_classes = data.inp_shape(), data.num_classes()
    model = NeuralNetwork( inp_shape, num_classes, hidden_units=64 )
    train( data, model )

