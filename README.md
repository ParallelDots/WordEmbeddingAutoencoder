# WordEmbeddingAutoencoder
An autoencoder to calculate word embeddings as mentioned in Lebret/Collobert paper 2015

Runs the model mentined in http://arxiv.org/abs/1412.4930 on set of text files in a directory, which can be specified in ETL folder.

Uses:

Numpy http://www.numpy.org/

Kayak https://github.com/HIPS/Kayak

Tested on around 200k blog articles dataset. Works well for proper nouns, needs improvements elsewhere

Todo: (Priorotywise)

1. Widen the context window (right now just 2 words)
2. Remove grabage words (words below certain frequency)
3. Implement a colsely related pape on phrase (http://arxiv.org/abs/1506.05703) (ICML 2015)
4. Try implementing on Theano as well ( for GPU )
