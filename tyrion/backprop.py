'''
Re-implementation of Word Embedding in theano
'''
import numpy as np
import theano
from theano import tensor as T
from compatibility import range, pickle


rng = np.random


class TrainModel(object):
    def __init__(self, maxnum, reduced_dims, learnrate=0.4):
        self.threshold = 1e-2
        # Input variable (equivalent to dummyword in original implementation)
        self.inputs = theano.shared(np.zeros((maxnum, 1), dtype=np.float32))
        self.W1 = theano.shared((rng.randn(reduced_dims, maxnum)*0.1)
                                .astype(theano.config.floatX), name='W1')
        self.W2 = theano.shared((rng.randn(maxnum, reduced_dims)*0.1).astype
                                (theano.config.floatX), name='W2')
        self.output = T.dot(self.W1, self.inputs)
        self.recons = T.dot(self.W2, self.output)
        self.totloss = T.sum((self.inputs - self.recons)**2)
        self.W1_grad = T.clip(T.grad(self.totloss, self.W1),
                         -1*self.threshold, self.threshold)
        self.W2_grad = T.clip(T.grad(self.totloss, self.W2),
                         -1*self.threshold, self.threshold)
        self.updates = [(self.W1, self.W1 - learnrate * self.W1_grad),
                   (self.W2, self.W2 - learnrate * self.W2_grad)]
        self.train = theano.function([], self.totloss, updates=self.updates,
                                allow_input_downcast=True)

    def trainonone(self, wordvec):
        wordvec = np.array(wordvec, dtype=np.float32)
        self.inputs.set_value(wordvec)
        # Gradients w.r.t paramters with values clipped in range (-1*threshold,
        # threshold)
        self.loss = self.train()
        print ("Loss incurred : ", self.loss)

    def getoutput(self, wordvec):
        # Returns the embedding given a word (calculate output W1*input
        self.inputs.set_value(wordvec)
        genembedding = self.output.eval()
        return genembedding
