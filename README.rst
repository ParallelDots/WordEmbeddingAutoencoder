# Tyrion - Word Embedding Autoencoder

An autoencoder to calculate word embeddings as mentioned in Lebret/Collobert paper 2015

Runs the model mentined in http://arxiv.org/abs/1412.4930 on set of text files stored in directory.

Requirements:

1. Numpy http://www.numpy.org/

2. Theano http://deeplearning.net/software/theano/install.html

3. Scipy 

Usage:

First run train.py with arguments as specified, it will generate the embedding file.

To generate word embeddings and to find closest words to a word, use utils module in tyrion. Ex.

>> from tyrion import utils
>> embedding = utils.gen_embedding('<word>')
>> close_words = utils.closest_words('<word>',n=10)


Todo: (Prioritywise)

1. Implement a closely related paper on phrase embeddings (http://arxiv.org/abs/1506.05703) (ICML 2015)
2. Try implementing AdaGrad ( for optimization )
