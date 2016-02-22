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
        self.W1 = theano.shared(rng.randn(reduced_dims, maxnum)*0.1, name='W1')
        self.W2 = theano.shared(rng.randn(maxnum, reduced_dims)*0.1, name='W2')
        self.output = T.dot(self.W1, self.inputs)
        self.recons = T.dot(self.W2, self.output)
        # L2 Loss calculation
        self.totloss = T.sum((self.output-self.recons)**2)

    def trainonone(self, wordvec, learningrate=0.4):
        # Set input value to input wordvector
        self.inputs.set_value(wordvec)
        W1_grad = T.grad(self.totloss,self.W1)
        # Gradient clipping to avoid gradient exploding
        temp = T.set_subtensor(W1_grad[(W1_grad > self.threshold).nonzero()], self.threshold)
        W1_upd = T.set_subtensor(temp[(temp < -1*self.threshold).nonzero()], -1*self.threshold)

        W2_grad = T.grad(self.totloss, self.W2)
        temp = T.set_subtensor(W2_grad[(W2_grad > self.threshold).nonzero()], self.threshold)
        W2_upd = T.set_subtensor(temp[temp < -1*self.threshold], -1*self.threshold)

        updates = [(self.W1, self.W1- learningrate*W1_upd), (self.W2, self.W2- learningrate*W2_upd)]

        train = theano.function([], self.totloss, updates=updates, allow_input_downcast=True)
        self.loss = train()
        print "Loss incurred : ",self.loss

    def getoutput(self,wordvec):
        # Function to generate embedding given a word (calculate ouput vector W1*input)
        self.inputs.set_value(wordvec)
        genembedding = self.output.eval()
        return genembedding
