from collections import defaultdict,Counter
import numpy as np
from compatibility import range, pickle
class WordNumRelation(object):
	"""docstring for WordNumRelation
	Holds 2 dictionary to return a number corresponding to word and a word related to a number
	"""
	def __init__(self):
		self.maxnum = 0
		self.W2N = {}
		self.N2W = {}

	def getNum(self,word,training=False):
		if word in self.W2N:
			return self.W2N[word]
		elif training:
			self.W2N[word] = self.maxnum
			self.N2W[self.maxnum] = word
			self.maxnum+=1
			return self.maxnum-1
		else:
			## Used during non training times, the new words are all given an output None
			return None

	def getWord(self,num):
		if num in self.N2W:
			return self.N2W[num]
		else:
			return None

def takezero():
	return 0.

class ProbArray(object):
	"""docstring for ProbArray"""
	def __init__(self,topncontextwords = 10000):
		self.wordcooccurance = defaultdict(dict)
		self.wordcount = defaultdict(takezero)
		self.wordnumrelation = WordNumRelation()
		self.topN = topncontextwords
		self.frozen = False
		self.topwordnums = None

	def addcontext(self,preword,postword):
		""" Send this function every co-occurance pair"""
		""" Stores values so as to calculate P(w,t)/P(w,t-1) later """
		prenum = self.wordnumrelation.getNum(preword,True)
		postnum = self.wordnumrelation.getNum(postword,True)
		if postnum in self.wordcooccurance[prenum]:
			self.wordcooccurance[prenum][postnum]+=1
		else:
			self.wordcooccurance[prenum][postnum]=1
		self.wordcount[prenum]+=1

	def freeze(self):
		c = Counter(self.wordcount)
		mcw = c.most_common(self.topN)
		self.topwordnums = []
		for (wordnum,occur) in mcw:
			self.topwordnums.append((wordnum,occur))
		self.frozen = True

	def makevectorfornum(self,num,topn=True):
		if not topn:
			count = float(self.wordcount[num])
			probarray = np.zeros((self.wordnumrelation.maxnum,1), dtype=float32)
			for (num2,times) in self.wordcooccurance[num].items():
				probarray[num2] = times/count
			return np.sqrt(probarray)
		else:
			if not self.frozen:
				raise Exception("Freeze top words first, run obj.freeze()")
			count = float(self.wordcount[num])
			probarray = np.zeros((self.topN,1), dtype=np.float32)
			for (numinarr,(num1,count)) in enumerate(self.topwordnums):
				if num1 in self.wordcooccurance[num]:
					times = self.wordcooccurance[num][num1]
				else:
					times = 0
				probarray[numinarr] = times/count
			return np.sqrt(probarray)

	def makevector(self,word):
		num = self.wordnumrelation.getNum(word)
		if num is None:
			return None
		return self.makevectorfornum(num)

	def getallwordvecs(self):
		for num in self.wordcooccurance.keys():
			yield (num,self.makevectorfornum(num))
