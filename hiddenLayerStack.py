import numpy as np

from hiddenLayer import HiddenLayer, _dropout_from_layer, DropoutHiddenLayer
from logistic_sgd import LogisticRegression


class HiddenLayerStack(object):
    def __init__(self,  input, n_in, layer_sizes,rng):
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
        
        self.params =   [layer.W for layer in self.hidden_layers] + \
                        [layer.b for layer in self.hidden_layers]
        
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
    
class DropoutHiddenLayerStack(object):
    """A multilayer perceptron with all the trappings required to do dropout
    training.

    """
    def __init__(self,
            rng,
            input,
            layer_sizes,
            dropout_rates,
            activations,
            use_bias=True):

        #rectified_linear_activation = lambda x: T.maximum(0.0, x)

        # Set up all the hidden layers
        weight_matrix_sizes = zip(layer_sizes, layer_sizes[1:])
        self.layers = []
        self.dropout_layers = []
        next_layer_input = input
        #first_layer = True
        # dropout the input
        next_dropout_layer_input = _dropout_from_layer(rng, input, p=dropout_rates[0])
        layer_counter = 0        
        for n_in, n_out in weight_matrix_sizes[:-1]:
            next_dropout_layer = DropoutHiddenLayer(rng=rng,
                    input=next_dropout_layer_input,
                    activation=activations[layer_counter],
                    n_in=n_in, n_out=n_out, use_bias=use_bias,
                    dropout_rate=dropout_rates[layer_counter + 1])
            self.dropout_layers.append(next_dropout_layer)
            next_dropout_layer_input = next_dropout_layer.output

            # Reuse the paramters from the dropout layer here, in a different
            # path through the graph.
            next_layer = HiddenLayer(rng=rng,
                    input=next_layer_input,
                    activation=activations[layer_counter],
                    # scale the weight matrix W with (1-p)
                    W=next_dropout_layer.W * (1 - dropout_rates[layer_counter]),
                    b=next_dropout_layer.b,
                    n_in=n_in, n_out=n_out,
                    use_bias=use_bias)
            self.layers.append(next_layer)
            next_layer_input = next_layer.output
            #first_layer = False
            layer_counter += 1
        
        # Set up the output layer
        n_in, n_out = weight_matrix_sizes[-1]
        dropout_output_layer = LogisticRegression(
                input=next_dropout_layer_input,
                n_in=n_in, n_out=n_out)
        self.dropout_layers.append(dropout_output_layer)

        # Again, reuse paramters in the dropout output.
        output_layer = LogisticRegression(
            input=next_layer_input,
            # scale the weight matrix W with (1-p)
            W=dropout_output_layer.W * (1 - dropout_rates[-1]),
            b=dropout_output_layer.b,
            n_in=n_in, n_out=n_out)
        self.layers.append(output_layer)

        # Use the negative log likelihood of the logistic regression layer as
        # the objective.
        self.dropout_negative_log_likelihood = self.dropout_layers[-1].negative_log_likelihood
        self.dropout_errors = self.dropout_layers[-1].errors

        self.negative_log_likelihood = self.layers[-1].negative_log_likelihood
        self.errors = self.layers[-1].errors

        # Grab all the parameters together.
        self.params = [ param for layer in self.dropout_layers for param in layer.params ]
        
        self.output = self.layers[-1].output
        
        self.predict_class = self.layers[-1].y_pred        
        self.predict_proba = self.layers[-1].p_y_given_x