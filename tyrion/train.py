'''
Code to train Word embeddings on given corpus.
'''

import sys
from backprop import TrainModel
from probarray import ProbArray
import pickle
import re
import json
import os
import numpy as np
import argparse
import time
import pdb

def train(folder,contextSize=5,min_count=100, newdims=100, ntimes=2,
          maxnum=10000, lr=0.4):
    '''
    Function to train autoencoder.
    '''
    t = time.time()
    lr_decay = 0.95
    pa = ProbArray()
    # Frequency to filter out low freq words
    freq = {}
    filepaths = list(map(lambda x: folder + "/" + x,os.listdir(folder)))
    rgx = re.compile("([\w][\w']*\w)")
    # Another iterator to count frequency of words
    print ("Pre-processing (clearning garbage words)")
    for filepath in filepaths:
        text = open(filepath).read().lower()
        tokens = re.findall(rgx, text)

        N = len(tokens)
        for i in range(0,N):
            if tokens[i] in freq:
                freq[tokens[i]] += 1
            else:
                freq[tokens[i]] = 1
    pdb.set_trace()

    # Sort the frequencies storing in (freq, token) pairs and prune words with freq < min_count
    tokenFreq = sorted(freq.items(), key = lambda x: x[1])
    garbageWords = []
    for item in tokenFreq:
        if item[1] < min_count:
            garbageWords.append(item[0])

    print ("Generating co-occurence matrix")
    doc_text = ""
    for filepath in filepaths:
        text = open(filepath).read().lower()
        words = re.findall(rgx, text)
        N = len(words)
        temp = [' '] * (N +  contextSize)
        temp[contextSize : (contextSize + N)] = words
        words = temp
        for i in range(contextSize, (contextSize + N)):
            # Filter out garbage words"
            #if words[i] not in garbageWords:
            # Include context size specified by user
            for j in range(i-contextSize, i):
                if words[i] != ' ' and words[j] != ' ':
                        pa.addcontext(words[j], words[i])
                        pa.addcontext(words[i], words[j])

    print ("Co-occurence matrix generated")
    print ("Starting training")
    tm = TrainModel(maxnum, newdims)
    pa.freeze()
    for k in range(ntimes):
        for numwordvec in pa.getallwordvecs():
            tm.trainonone(numwordvec[1])
        lr /=float(1+k*lr_decay)

    wordembeddings = {}
    for numwordvec in pa.getallwordvecs():
        (num, wordvec) = numwordvec
        word = pa.wordnumrelation.getWord(num)
        embedding = tm.getoutput(wordvec)
        wordembeddings[word] = embedding

    print ("Training proces done, dumping embedding into persistant storage!")

    open(r'.\embeddings.pickle', "wb")
    pickle.dump(wordembeddings, outfile)
    outfile.close()
    print ("Training completed! Embedding done.")
    print ("time is %f" % (time.time()-t))

if __name__ == "__main__":
    train()
