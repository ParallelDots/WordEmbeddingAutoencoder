'''
Re-implementation of Word Embedding in theano
'''
import numpy as np
import theano
from theano import tensor as T


rng = np.random


class TrainModel(object):

    def __init__(self, maxnum, reduced_dims, learnrate=0.001):
        self.threshold = 1e-2
        # Input variable (equivalent to dummyword in original implementation)
        self.inputs = theano.shared(np.zeros((maxnum,1), dtype=np.float32))
        self.W1 = theano.shared(np.asarray(rng.randn(reduced_dims, maxnum) * 0.1,
                                           dtype=theano.config.floatX), name='W1', borrow=True)
        self.W2 = theano.shared(np.asarray(rng.rand(maxnum, reduced_dims) * 0.1,
                                           dtype=theano.config.floatX), name='W2', borrow=True)
        self.params = [self.W1, self.W2]
        self.gW = theano.shared(value=np.ones((reduced_dims, maxnum), dtype=theano.config.floatX), borrow=True)
        self.gW1 = theano.shared(value=np.ones((maxnum, reduced_dims), dtype=theano.config.floatX), borrow=True)
        self.output = T.dot(self.W1, self.inputs)
        self.recons = T.dot(self.W2, self.output)
        self.hist = [self.gW, self.gW1]
        # l2 regularized loss (to avoid exploding of paramters)
        self.totloss = T.sum((self.inputs - self.recons)**2) + 0.0001 * T.sum(self.W1 ** 2) + 0.0001 * T.sum(self.W2 ** 2)
        self.grad = [T.grad(self.totloss, param) for param in self.params]
        self.grad = [T.clip(grad, -self.threshold, self.threshold) for grad in self.grad]
        # AdaGrad update
        self.update_hist = [(hist, T.sqrt(hist + g**2)) for (hist,g) in zip(self.hist, self.grad)]
        self.updates = [(param, param - (learnrate / hist) * grad) for param,hist,grad in zip(self.params, self.hist, self.grad)]
        self.updates += self.update_hist
        self.train = theano.function([], self.totloss, updates=self.updates,
                                     allow_input_downcast=True)

    def trainonone(self, pa, ntimes):
        for k in range(ntimes):
            for wordvec in pa.getallwordvecs():
                wordvec = np.array(wordvec[1], dtype=np.float32)
                self.inputs.set_value(wordvec)
                self.loss = self.train()
                print("Loss incurred at epoch %d is %f :"%(k, self.loss))

    def getoutput(self, wordvec):
        # Returns the embedding given a word (calculate output W1*input
        self.inputs.set_value(wordvec)
        genembedding = self.output.eval()
        return genembedding
