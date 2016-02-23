# Tyrion - Word Embedding Autoencoder

An autoencoder to calculate word embeddings as mentioned in Lebret/Collobert paper 2015

R:uns the model mentined in http://arxiv.org/abs/1412.4930 on set of text files stored in directory.

Requirements:

1. Numpy http://www.numpy.org/

2. Theano http://deeplearning.net/software/theano/install.html

3. Scipy http://www.scipy.org/install.html

# Installation 

To install the package run setup.py file, it will install and include necessary files in system python directory

$ python setup.py install 

# Training

To train the model on bunch of text files run following commnands in python console:

\>> from tyrion import train

\>> train.train('/path/to/text/corpus')

# Setting hyperparameters

Different hyperparameters are set by default in training module, to set hyperparamters manually use following command while training

\>> from tyrion import train

>> train.train('/path/to/text/corpus', contextSize=size, min_count=count, newdims=dims, ntimes=ntimes, lr=learningrate)

Arguments explained:

1. ContextSize is the contex window size used for constructing coocurence matrix

2. min_count is minimum threshold frequency for words, removes garbage words below this frequency

3. newdims is the dimension of embedding desired.

4. ntimes is number of epochs for training model

5. lr is the learning rate to drive optimization routine

# Utility functions

To generate word embeddings and to find closest words to a word, use utils module in tyrion. Ex.

>> from tyrion import utils

>> embedding = utils.gen_embedding('word')

>> close_words = utils.closest_words('word',n=10)


# Future work

1. Implement a closely related paper on phrase embeddings (http://arxiv.org/abs/1506.05703) (ICML 2015)
2. Try implementing AdaGrad ( for optimization )
