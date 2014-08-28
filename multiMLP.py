__docformat__ = 'restructedtext en'

import numpy as np



from logistic_sgd import LogisticRegression
from hiddenLayerStack import HiddenLayerStack

np.random.seed(42)
rng = np.random.RandomState(1234)


class MultiMLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``hiddenLayerStack`` class)  while the
    top layer is a softamx layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, input, n_in, layer_sizes, n_out):


        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a hiddenLayerStack with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayerStack = HiddenLayerStack(input, n_in, layer_sizes)

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayerStack.output,
            n_in= self.hiddenLayerStack.output_size,
            n_out=n_out)

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = sum([abs(layer.W).sum() for layer in self.hiddenLayerStack.hidden_layers]) \
                + abs(self.logRegressionLayer.W).sum()

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = sum([(layer.W ** 2).sum() for layer in self.hiddenLayerStack.hidden_layers]) \
                    + (self.logRegressionLayer.W ** 2).sum()

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayerStack.params + self.logRegressionLayer.params
        
        self.predict_class = self.logRegressionLayer.y_pred
        
        self.predict_proba = self.logRegressionLayer.p_y_given_x
