
import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T
from mlp import MLP
from logistic_sgd import  load_data


np.random.seed(42)
rng = np.random.RandomState(1234)



class MLPClassifier(object) :
    
    def __init__(self,input_size,output_size,n_hidden=500,learning_rate=0.01, 
            L1_reg=0.00, L2_reg=0.0001, 
            n_epochs=1000,batch_size=20):
        self.learning_rate = learning_rate
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg
        self.n_epochs = n_epochs
        self.batch_size=batch_size
        self.n_hidden = n_hidden
        self.x = T.matrix('x')      
        self.mlp =  MLP(input = self.x, n_in = input_size, \
                     n_hidden = n_hidden, n_out = output_size)
        
    def fit(self,X,y):
         X_train, X_valid, y_train, y_valid = self.splitData(X,y)
         train_model(self.mlp,self.x, X_train,X_valid,y_train,
                     y_valid, self.L1_reg, 
                     self.L2_reg, self.learning_rate, 
                     self.n_epochs, self.batch_size)
#        
    def predict(self,X, y = None):        
        fit_model = theano.function(
            inputs=[],
            outputs=self.mlp.predict_class,
            givens={self.x : X}
            )
        output = fit_model()
        if(y != None):
            validate_model = theano.function(inputs=[],
            outputs=self.mlp.errors(y),
            givens={
                self.x: X,
                y: y})
            print((' validation error %f %%') % (validate_model() * 100.))
        return fit_model()
        
    def predict_proba(self,X, y = None):        
        fit_model = theano.function(
            inputs=[],
            outputs=self.mlp.predict_proba,
            givens={self.x : X}
            )
        output = fit_model()
        if(y != None):
            validate_model = theano.function(inputs=[],
            outputs=self.mlp.errors(y),
            givens={
                self.x: X,
                y: y})
            print((' validation error %f %%') % (validate_model() * 100.))
        return fit_model()
        
    def getSharedInstance(self,array):
            return theano.shared(np.asarray(array, dtype=theano.config.floatX))
        
    def splitData(self,X,y):    
        r = np.random.rand(X.shape[0])    
        
        X_train = self.getSharedInstance(X[r<0.9])
        X_valid = self.getSharedInstance(X[r>=0.9])
        
        y_train = T.as_tensor_variable(y[r<0.9])
        y_valid = T.as_tensor_variable(y[r>=0.9])
        
        # First 90% train, Lirst 10% validation
        return  X_train, \
                X_valid, \
                T.cast(y_train,'int32'), \
                T.cast(y_valid,'int32')
    
  
def train_model(classifier,x, X_train,X_valid,y_train,y_valid, L1_reg, L2_reg, 
                learning_rate, n_epochs, batch_size) :
    
    index = T.lscalar()
    y = T.ivector('y')     
    
    n_train_batches = X_train.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = X_valid.get_value(borrow=True).shape[0] / batch_size
    
    cost = classifier.negative_log_likelihood(y) \
         + L1_reg * classifier.L1 \
         + L2_reg * classifier.L2_sqr       
    
    validate_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: X_valid[index * batch_size:(index + 1) * batch_size],
                y: y_valid[index * batch_size:(index + 1) * batch_size]})

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = []
    for param in classifier.params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    updates = []
    for param, gparam in zip(classifier.params, gparams):
        updates.append((param, param - learning_rate * gparam))

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(inputs=[index], outputs=cost,
            updates=updates,
            givens={
                x: X_train[index * batch_size:(index + 1) * batch_size],
                y: y_train[index * batch_size:(index + 1) * batch_size]})    
    
    print '... training'

    # early-stopping parameters
    patience = 1000000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    #best_params = None
    best_validation_loss = np.inf
    best_iter = 0
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                     (epoch, minibatch_index + 1, n_train_batches,
                      this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    #test_losses = [test_model(i) for i
                    #               in xrange(n_test_batches)]
                    #test_score = numpy.mean(test_losses)

                    #print(('     epoch %i, minibatch %i/%i, test error of '
                    #       'best model %f %%') %
                    #      (epoch, minibatch_index + 1, n_train_batches,
                    #       test_score * 100.))

            if patience <= iter:
                    done_looping = True
                    break

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i %%') %
          (best_validation_loss * 100., best_iter + 1))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
                          

    

def run():
    
    datasets = load_data('mnist.pkl.gz')
    
    X, y = datasets[0]
    
    clf = MLPClassifier(28 * 28, 10, n_epochs = 10)
    
    clf.train(X.get_value(),y.eval())
    
    #print(X_train.get_value().shape[0])
    #X_train = shared(X_train.get_value())
    #print(X_train.get_value().shape[0])
    X_test, y_test = datasets[2]
    
    clf.fit(X_test,y_test)
    
    #r = np.random.rand(X_train.get_value().shape[0])    
        
    #X_train = shared(X_train.get_value()[r<0.9])
    #X_valid = shared(X_valid.get_value()[r<0.9])
        
    #y_train = T.as_tensor_variable(y_train.eval()[r<0.9])
    #y_valid = T.as_tensor_variable(y_valid.eval()[r<0.9],dtype='int32')
    
                 
