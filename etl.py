from probarray import ProbArray
from backprop import TrainModel
import os,re
import cPickle as pickle 

pa = ProbArray()
folder = "ie_data"
filepaths = map(lambda x: folder+"/"+x,os.listdir(folder))
rgx = re.compile("([\w][\w']*\w)")
for filepath in filepaths:
	alltext = open(filepath).read().lower()
	words = filter(lambda x: len(x)<15 ,re.findall(rgx,alltext))
	for i in xrange(1,len(words)):
		pa.addcontext(words[i-1],words[i])
		pa.addcontext(words[i],words[i-1])

maxnum = 20000 #pa.wordnumrelation.maxnum
newdims = 100 #suppose 100 dimensions afterall
tm = TrainModel(maxnum,newdims)
ntimes = 2
pa.freeze()
for k in xrange(ntimes):
	for numwordvec in pa.getallwordvecs():
		tm.trainonone(numwordvec[1])
wordembeddings = {}
for numwordvec in pa.getallwordvecs():
	(num,wordvec) = numwordvec
	word = pa.wordnumrelation.getWord(num)
	embedding = tm.getoutput(wordvec)
	wordembeddings[word] = embedding
outfile = open("embeddings.pickle","w")
pickle.dump(wordembeddings,outfile)
outfile.close()



