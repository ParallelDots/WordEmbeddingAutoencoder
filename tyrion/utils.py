import heapq,random
from compatibility import range, pickle
import numpy as np
from scipy.spatial.distance import cosine

def gen_embedding(word):
    '''
    Generates embedding of the word from the model trained.
    '''
    try:
        f = open('./embeddings.pickle')
        embeddings = pickle.load(f)
        return embeddings[word]
    except Exception as e:
        print ("Exception: Model file not found, please train the model first by runing train")

def closest_words(word,topn=10):
    '''
    Returns top 10 closest words to word provided by user.
    '''
    try:
        embeddings = pickle.load(open("./embeddings.pickle"))
        words = embeddings.keys()
        closest = []
        # Embedding of word provided by user
        vec = embeddings[word]
        for word in words:
            heapq.heappush(closest, (1 - cosine(vec, embeddings[word]), word))
        closest_words = heapq.nlargest(topn, closest)
        return closest_words
    except Exception as e:
        print ("Exception: Model file not found, please train the model by running train function")

