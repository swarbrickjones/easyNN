import os
import sys
import time
import numpy as np
from collections import OrderedDict
import theano
import theano.tensor as T
from theano.ifelse import ifelse
from hiddenLayerStack import DropoutHiddenLayerStack


initial_learning_rate = 1.0
learning_rate_decay = 0.998
squared_filter_length_limit = 15.0

batch_size = 100
n_epochs = 100

#### the params for momentum
mom_start = 0.5
mom_end = 0.99
# for epoch in [0, mom_epoch_interval], the momentum increases linearly
# from mom_start to mom_end. After mom_epoch_interval, it stay at mom_end
mom_epoch_interval = 500
mom_params = {"start": mom_start,
                  "end": mom_end,
                  "interval": mom_epoch_interval}

class MultiMLPClassifier(object) :
    
    def __init__(self,
                layer_sizes, 
                activations,
                dropout_rates,
                initial_learning_rate = initial_learning_rate,
                learning_rate_decay = learning_rate_decay,
                squared_filter_length_limit = squared_filter_length_limit,
                n_epochs=n_epochs,
                batch_size=batch_size,
                mom_params = mom_params,                
                dropout = True,
                use_bias = True,
                random_seed=1234):
            
        assert len(layer_sizes) - 1 == len(dropout_rates)
            
        self.initial_learning_rate = initial_learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.squared_filter_length_limit = squared_filter_length_limit
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.mom_params = mom_params
        self.activations = activations
        self.dropout = dropout
        self.dropout_rates = dropout_rates
        self.layer_sizes = layer_sizes
        self.use_bias = use_bias
        self.rng = np.random.RandomState(random_seed)
        self.initial_learning_rate = initial_learning_rate
        
        self.n_epochs = n_epochs
        self.batch_size=batch_size
        self.x = T.matrix('x')      
        self.hidden_layers = DropoutHiddenLayerStack(rng=self.rng, input=self.x,
                     layer_sizes=layer_sizes,
                     dropout_rates=dropout_rates,
                     activations=activations,
                     use_bias=use_bias)
                     
        self.mom_start = mom_params["start"]
        self.mom_end = mom_params["end"]
        self.mom_epoch_interval = mom_params["interval"]
       
        
        
    def fit(self,X,y):
         X_train, X_valid, y_train, y_valid = self.splitData(X,y)
         train_model(self.hidden_layers,self.x, X_train,X_valid,y_train,y_valid, 
                initial_learning_rate=self.initial_learning_rate,
                learning_rate_decay=self.learning_rate_decay,
                squared_filter_length_limit=self.squared_filter_length_limit,
                mom_start=self.mom_start,
                mom_end =self.mom_end ,
                mom_epoch_interval=self.mom_epoch_interval,
                dropout=self.dropout,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size)
#        
    def predict(self,X, y = None):        
        fit_model = theano.function(
            inputs=[],
            outputs=self.hidden_layers.predict_class,
            givens={self.x : X}
            )
        if(y != None):
            validate_model = theano.function(inputs=[],
            outputs=self.hidden_layers.errors(y),
            givens={
                self.x: X,
                y: y})
            print((' validation error %f %%') % (validate_model() * 100.))
        return fit_model()
        
    def predict_proba(self,X, y = None):        
        fit_model = theano.function(
            inputs=[],
            outputs=self.hidden_layers.predict_proba,
            givens={self.x : X}
            )
        if(y != None):
            validate_model = theano.function(inputs=[],
            outputs=self.hidden_layers.errors(y),
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
    
  
def train_model(classifier,x, X_train,X_valid,y_train,y_valid, 
                initial_learning_rate,
                learning_rate_decay,
                squared_filter_length_limit,
                mom_start,
                mom_end,    
                mom_epoch_interval,  ## momentum params
                dropout,
                n_epochs,
                batch_size) :
    
    index = T.lscalar()
    epoch = T.scalar()
    y = T.ivector('y')   
    
    learning_rate = theano.shared(np.asarray(initial_learning_rate,
        dtype=theano.config.floatX))
    
    n_train_batches = X_train.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = X_valid.get_value(borrow=True).shape[0] / batch_size
    
    cost = classifier.negative_log_likelihood(y)
    dropout_cost = classifier.dropout_negative_log_likelihood(y)    
    
    validate_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: X_valid[index * batch_size:(index + 1) * batch_size],
                y: y_valid[index * batch_size:(index + 1) * batch_size]})

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = []
    for param in classifier.params:
        # Use the right cost function here to train with or without dropout.
        gparam = T.grad(dropout_cost if dropout else cost, param)
        gparams.append(gparam)
        
    gparams_mom = []
    for param in classifier.params:
        gparam_mom = theano.shared(np.zeros(param.get_value(borrow=True).shape,
            dtype=theano.config.floatX))
        gparams_mom.append(gparam_mom)
        
    mom = ifelse(epoch < mom_epoch_interval,
            mom_start*(1.0 - epoch/mom_epoch_interval) + mom_end*(epoch/mom_epoch_interval),
            mom_end)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    updates = OrderedDict()
    for gparam_mom, gparam in zip(gparams_mom, gparams):
        # Misha Denil's original version
        #updates[gparam_mom] = mom * gparam_mom + (1. - mom) * gparam
      
        # change the update rule to match Hinton's dropout paper
        updates[gparam_mom] = mom * gparam_mom - (1. - mom) * learning_rate * gparam

    # ... and take a step along that direction
    for param, gparam_mom in zip(classifier.params, gparams_mom):        
        stepped_param = param + updates[gparam_mom]

        # This is a silly hack to constrain the norms of the rows of the weight
        # matrices. 
        if param.get_value(borrow=True).ndim == 2:
            #squared_norms = T.sum(stepped_param**2, axis=1).reshape((stepped_param.shape[0],1))
            #scale = T.clip(T.sqrt(squared_filter_length_limit / squared_norms), 0., 1.)
            #updates[param] = stepped_param * scale
            
            # constrain the norms of the COLUMNs of the weight, according to
            # https://github.com/BVLC/caffe/issues/109
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(squared_filter_length_limit))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param
            
    # Compile theano function for training.  This returns the training cost and
    # updates the model parameters.
    output = dropout_cost if dropout else cost
    train_model = theano.function(inputs=[epoch, index], outputs=output,
            updates=updates,
            givens={
                x: X_train[index * batch_size:(index + 1) * batch_size],
                y: y_train[index * batch_size:(index + 1) * batch_size]})
    #theano.printing.pydotprint(train_model, outfile="train_file.png",
    #        var_with_name_simple=True)

    # Theano function to decay the learning rate, this is separate from the
    # training function because we only want to do this once each epoch instead
    # of after each minibatch.
    decay_learning_rate = theano.function(inputs=[], outputs=learning_rate,
            updates={learning_rate: learning_rate * learning_rate_decay})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    best_validation_errors = np.inf
    best_iter = 0
    test_score = 0.
    epoch_counter = 0
    start_time = time.clock()

    while epoch_counter < n_epochs:
        # Train this epoch
        epoch_counter = epoch_counter + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(epoch_counter, minibatch_index)

        # Compute loss on validation set
        validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
        this_validation_errors = np.sum(validation_losses)

        # Report and save progress.
        print "epoch {}, test error {}, learning_rate={}{}".format(
                epoch_counter, this_validation_errors,
                learning_rate.get_value(borrow=True),
                " **" if this_validation_errors < best_validation_errors else "")

        best_validation_errors = min(best_validation_errors,
                this_validation_errors)
                
        new_learning_rate = decay_learning_rate()

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_errors * 100., best_iter, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
                          
