import heapq,random
from compatibility import range, pickle
import numpy as np
from scipy.spatial.distance import cosine

def gen_embedding(word):
    '''
    Generates embedding of the word from the model trained.
    '''
    try:
        with open('./embeddings.pickle', 'rb') as f:
            embeddings = pickle.load(f)
        return embeddings[word]
    except Exception as e:
        print ("Exception: Model file not found, please train the model first by runing train")

def closest_words(word,topn=10):
    '''
    Returns top 10 closest words to word provided by user.
    '''
    try:
        with open("./embeddings.pickle", "rb") as f:
            embeddings = pickle.load(f)
        words = embeddings.keys()
        closest = []
        # Embedding of word provided by user
        vec = embeddings[word]
        for word in words:
            heapq.heappush(closest, (1 - cosine(vec, embeddings[word]), word))
        closest_words = heapq.nlargest(topn, closest)
        return closest_words
    except Exception as e:
        print ("Exception: Check if model file is present, if not them train the model first, if present, then vocabulary issue, queried word not present in vocab")

