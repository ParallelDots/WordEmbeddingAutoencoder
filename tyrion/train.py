'''
Code to train Word embeddings on given corpus.
'''

import sys
from backprop import TrainModel
from probarray import ProbArray
import cPickle as pickle
import re
import json
import os
import numpy as np
import argparse


def train(folder,contextSize=5,min_count=100, newdims=100, ntimes=2,
          maxnum=10000, lr=0.4):
    '''
    Function to train autoencoder.
    '''
    lr_decay = 0.95
    pa = ProbArray()
    # Frequency to filter out low freq words
    freq = {}
    filepaths = map(lambda x: folder + "/" + x,os.listdir(folder))
    rgx = re.compile("([\w][\w']*\w)")
    # Another iterator to count frequency of words
    print "Pre-processing (clearning garbage words)"
    for filepath in filepaths:
        text = open(filepath).read().lower()
        tokens = re.findall(rgx, text)
        N = len(tokens)
        for i in xrange(0,N):
            if tokens[i] in freq:
                freq[tokens[i]] += 1
            else:
                freq[tokens[i]] = 1

    # Sort the frequencies storing in (freq, token) pairs and prune words with freq < min_count
    tokenFreq = sorted(freq.items(), key = lambda x: x[1])
    garbageWords = []
    for item in tokenFreq:
        if item[1] < min_count:
            garbageWords.append(item[0])

    print "Generating co-occurence matrix"
    doc_text = ""
    for filepath in filepaths:
        text = open(filepath).read().lower()
        words = re.findall(rgx, text)
        N = len(words)
        temp = [' '] * (N +  contextSize)
        temp[contextSize : (contextSize + N)] = words
        words = temp
        for i in xrange(contextSize, (contextSize + N)):
            # Filter out garbage words"
            #if words[i] not in garbageWords:
            # Include context size specified by user
            for j in xrange(i-contextSize, i):
                if words[i] != ' ' and words[j] != ' ':
                        pa.addcontext(words[j], words[i])
                        pa.addcontext(words[i], words[j])

    print "Co-occurence matrix generated"
    print "Starting training"
    tm = TrainModel(maxnum, newdims)
    pa.freeze()
    for k in xrange(ntimes):
        for numwordvec in pa.getallwordvecs():
            tm.trainonone(numwordvec[1])
        lr /=float(1+k*lr_decay)

    wordembeddings = {}
    for numwordvec in pa.getallwordvecs():
        (num, wordvec) = numwordvec
        word = pa.wordnumrelation.getWord(num)
        embedding = tm.getoutput(wordvec)
        wordembeddings[word] = embedding

    print "Training proces done, dumping embedding into persistant storage!"

    outfile = open("./embeddings.pickle", "w")
    pickle.dump(wordembeddings, outfile)
    outfile.close()
    print "Training completed! Embedding done."

if __name__ == "__main__":
    train()
