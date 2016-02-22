from setuptools import setup

setup(name='tyrion',
      version='0.1',
      description='Word Embedding based on paper by Lebret, Collobert, 2015',
      url= 'https://github.com/shashankg7/WordEmbeddingAutoencoder',
      packages=['tyrion'],
      install_requires=['numpy','theano'],
      author=('Muktabh Mayank, Shashank gupta'),
      author_email=('muktabh@paralleldots.com,  h20104198@pilani.bits-pilani.ac.in'),
      license='MIT',
      classifier=['Development Status :: 3 - Alpha',
                  'License :: OSI Approved :: MIT',
                  'Topic :: Scientific/Engineering :: Artificial Intelligence']
      )
