'''
ETL code for extracting wiki text from wiki xml dump.
'''

import sys
from backprop import TrainModel
from probarray import ProbArray
import cPickle as pickle
import re
import os
import argparse


# Initialize hyperparamters of the model
contextSize = 0
min_count = 0
folder = ""
maxnum = 10000
newdims = 100
ntimes = 2
lr = 0


def arg_parser():
    global contextSize, min_count, folder, maxnum, newdims, ntimes, lr
    parser = argparse.ArgumentParser(description='Generates word embeddings from a corpus specified by user based on paper by Lebert 2015')
    parser.add_argument('path', help='Path to folder containing text files to be parsed')
    parser.add_argument('--context_size', help='Context window size to be considered', type=int)
    parser.add_argument('--maxnum', help='Maximum context words to be included based on context freq ', type=int)
    parser.add_argument('--dim', help='Dimension of word vector', type=int)
    parser.add_argument('--ntimes', help='number of epochs for training', type=int)
    parser.add_argument('--learning_rate', help='learning rate for driving optimization', type=float)
    args = parser.parse_args()
    folder = args.path
    if args.context_size:
        contextSize = args.context_size
    else:
        contextSize = 5
    if args.maxnum:
        maxnum = args.maxnum
    else:
        maxnum = 10000
    if args.dim:
        newdims = args.dim
    else:
        newdims = 100
    if args.ntimes:
        ntimes = args.ntimes
    else:
        ntimes = 2
    if args.learning_rate:
        lr = args.learning_rate
    else:
        lr = 0.4


def train():
    global folder, min_count, newdims, ntimes, maxnum, lr
    lr_decay = 0.95
    arg_parser()
    pa = ProbArray()
    # Frequency to filter out low freq words
    freq = {}
    filepaths = map(lambda x: folder + "/" + x, os.listdir(folder))
    rgx = re.compile("([\w][\w']*\w)")
    print "Pre-processing (clearning garbage words)"
    for filepath in filepaths:
        text = open(filepath).read().lower()
        tokens = re.findall(rgx, text)
        N = len(tokens)
        for i in xrange(0, N):
            if tokens[i] in freq:
                freq[tokens[i]] += 1
            else:
                freq[tokens[i]] = 1

    # Sort the frequencies storing in (freq, token) pairs and prune words with freq < min_count
    tokenFreq = sorted(freq.items(), key=lambda x: x[1])
    garbageWords = []
    for item in tokenFreq:
        if item[1] < min_count:
            garbageWords.append(item[0])

    print "Generating co-occurence matrix"
    for filepath in filepaths:
        text = open(filepath).read().lower()
        words = re.findall(rgx, text)
        N = len(words)
        temp = [' '] * (N + 2 * contextSize)
        temp[contextSize: (contextSize + N)] = words
        words = temp
        for i in xrange(contextSize, (contextSize + N)):
            if words[i] not in garbageWords:
                for j in xrange(i-contextSize, i + contextSize+1):
                    if i != j and words[i] != ' ' and words[j] != ' ':
                            pa.addcontext(words[j], words[i])
                            pa.addcontext(words[i], words[j])
    print "Co-occurence matrix generated"
    print "Starting training"
    tm = TrainModel(maxnum, newdims)
    pa.freeze()
    for k in xrange(ntimes):
        for numwordvec in pa.getallwordvecs():
            tm.trainonone(numwordvec[1], lr)
        lr /= float(1+k*lr_decay)

    wordembeddings = {}
    for numwordvec in pa.getallwordvecs():
        (num, wordvec) = numwordvec
        word = pa.wordnumrelation.getWord(num)
        embedding = tm.getoutput(wordvec)
        wordembeddings[word] = embedding

    print "Training proces done, dumping embedding into persistant storage!"

    outfile = open("../embeddings.pickle", "w")
    pickle.dump(wordembeddings, outfile)
    outfile.close()

    print "Training completed! Embedding done."

if __name__ == "__main__":
    train()
