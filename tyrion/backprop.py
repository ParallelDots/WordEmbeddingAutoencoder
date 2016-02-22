'''
Re-implementation of Word Embedding in theano
'''
import numpy as np
import theano
from theano import tensor as T


rng = np.random

class TrainModel(object):
    def __init__(self, maxnum, reduced_dims):
        self.threshold = 1e-2
        # Input variable (equivalent to dummyword in original implementation)
        self.inputs = theano.shared(np.zeros((maxnum,1)))
        self.W1 = theano.shared((rng.randn(reduced_dims, maxnum)*0.1).astype(theano.config.floatX), name='W1')
        self.W2 = theano.shared((rng.randn(maxnum, reduced_dims)*0.1).astype(theano.config.floatX), name='W2')
        self.output = T.dot(self.W1, self.inputs)
        self.recons = T.dot(self.W2, self.output)
        self.totloss = T.sum((self.inputs- self.recons)**2)

    def trainonone(self, wordvec, learnrate=0.4):
	self.inputs.set_value(wordvec)
        W1_grad = T.grad(self.totloss,self.W1)
	temp = T.set_subtensor(W1_grad[(W1_grad > self.threshold).nonzero()], self.threshold)
        W1_upd = T.set_subtensor(temp[(temp < -1*self.threshold).nonzero()], -1*self.threshold)
	W2_grad = T.grad(self.totloss, self.W2)
	temp = T.set_subtensor(W2_grad[(W2_grad > self.threshold).nonzero()], self.threshold)
        W2_upd = T.set_subtensor(temp[(temp < -1*self.threshold).nonzero()], -1*self.threshold)
        updates = [(self.W1, self.W1 - learnrate*W1_upd), (self.W2, self.W2- learnrate*W2_upd)]
        train = theano.function([], self.totloss, updates=updates, allow_input_downcast=True)
        self.loss = train()
        print "Loss incurred : ",self.loss

    def getoutput(self, wordvec):
	# Returns the embedding given a word (calculate output W1*input
	self.inputs.set_value(wordvec)
	genembedding = self.output.eval()
	return genembedding

