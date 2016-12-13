# Tyrion - Word Embedding Autoencoder

An autoencoder to calculate word embeddings as mentioned in Lebret/Collobert paper 2015

Runs the model mentined in http://arxiv.org/abs/1412.4930 on set of text files stored in directory on GPU

Supports both theano and tensorflow implementation

With GPU support code is 5-8x faster than it's CPU version. 

Requirements:

1. Numpy 

2. Tensorflow (or) Theano 

3. Scipy 

# Installation 

To install the package follow the instructions mentioned below 

# Install from source 

To install the package from source run setup.py file, it will install and include necessary files in system python directory

$ python setup.py install 

# Installation from pip (currently not supported !! )

Tyrion can also be install from pip. Run following command from console

$ pip install tyrion

# Training

You have an option to choose the backend to train the model on (theano/tensorflow). To select a backend, adjust the backend flag in config.json file supplied along with the code. 

To set theano backend, change "backend" to 'th'.

To set tensorflow backend, change "backend" to 'tf'.

To set parameters of the model, like learning rate, context window size etc. change the appropriate fields in the config file.

To train the model on bunch of text files run following commnands in python console:

There is a sample data folder in project's folder, that can be used for sanity checking

> python train.py CONFIG_PATH DATADIR_PATH

# Setting hyperparameters

Different hyperparameters are set by default in training module, to set hyperparamters manually follow the instruction in the previous section. Some arguments explained are:


1. ContextSize is the contex window size used for constructing coocurence matrix

2. min_count is minimum threshold frequency for words, removes garbage words below this frequency

3. newdims is the dimension of embedding desired.

4. ntimes is number of epochs for training model

5. lr is the learning rate to drive optimization routine

# Utility functions

To generate word embeddings and to find closest words to a word, use utils module in tyrion. Ex.

\>> from tyrion import utils

\>> embedding = utils.gen_embedding('word')

\>> close_words = utils.closest_words('word',n=10)
