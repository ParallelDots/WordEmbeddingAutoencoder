'''
Re-implementation of Word Embedding in tensorflow
'''
import numpy as np
import theano
import tensorflow as tf
from compatibility import range, pickle


rng = np.random


class TrainModel(object):
    def __init__(self, maxnum, reduced_dims, learnrate=0.0001, reg1=0.0001, reg2=0.0001):
        self.threshold = 1e-2
        # Input variable (equivalent to dummyword in original implementation)
        self.inputs = tf.placeholder("float", [maxnum, 1])
        self.W1 = tf.Variable((rng.randn(reduced_dims, maxnum)*0.1).astype(np.float32)
                                , name='W1')
        self.W2 = tf.Variable((rng.randn(maxnum, reduced_dims)*0.1).astype(np.float32)
                                , name='W2')
        self.output = tf.matmul(self.W1, self.inputs)
        self.recons = tf.matmul(self.W2, self.output)
        self.loss = tf.reduce_sum((self.inputs - self.recons)**2) + reg1 * tf.reduce_sum(self.W1 ** 2) + reg2 * tf.reduce_sum(self.W2 ** 2)
        self.train = tf.train.RMSPropOptimizer(learnrate, 0.9).minimize(self.loss)
        self.session = tf.Session()
        tf.initialize_all_variables().run(session=self.session)    

       
    def trainonone(self, pa, ntimes):    
        for k in range(ntimes):
            for wordvec in pa.getallwordvecs():
                _, loss = self.session.run([self.train, self.loss], feed_dict={self.inputs: wordvec[1]})
                print("Loss incurred : ", loss)
        self.saver = tf.train.Saver()
        self.saver.save(self.session, 'embedding.chk')

    def getoutput(self, wordvec, sess_path):
        # Returns the embedding given a word (calculate output W1*input 
        self.saver.restore(self.session, sess_path)
        embedding = self.session.run(self.output, feed_dict={self.inputs:wordvec})
        return embedding
