import numpy as np

import theano
import theano.tensor as T
from hiddenLayer import HiddenLayer

np.random.seed(42)
this_rng = np.random.RandomState(1234)

class HiddenLayerStack(object):
    def __init__(self,  input, n_in, layer_sizes,rng=this_rng):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        
        self.hidden_layers = getHiddenLayers(input,n_in, layer_sizes)       
        self.output_size = layer_sizes[-1]
        
        self.output = self.hidden_layers[-1].output
        
def getHiddenLayers(input,n_in, layer_sizes):
    input_size = n_in
    index = 0
    next_layer_input = input
    layer_list = []
    while index < len(layer_sizes):
        next_layer = HiddenLayer(rng=this_rng, input=next_layer_input, \
                n_in=input_size, n_out=layer_sizes[index])
        layer_list.append(next_layer)
        next_layer_input=next_layer.output
        input_size = layer_sizes[index]
        index+=1
    return layer_list
    