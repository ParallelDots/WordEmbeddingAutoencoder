import heapq,csv,random
import cPickle as pickle
import numpy as np
from scipy.spatial.distance import cosine
if __name__=='__main__':
	embeddings = pickle.load(open("embeddings.pickle"))
	print type(embeddings)
	n = 45
	words = embeddings.keys() 
	#print len(words),X.shape
	L = range(1,100)
	closest_words = {}
	for num in words:
		#print words[num],X[num,:]
		if random.choice(L)!=50:
			# Taking 1% words at random
			continue
		print "trying: ",num
		closest = []
		for num2 in words:
			try:
				if np.linalg.norm(embeddings[num2])==0:
					continue
				heapq.heappush(closest,(1 - cosine(embeddings[num],embeddings[num2]),num2))
			except KeyError:
				print "KeyError seen"
		actual_closest = heapq.nlargest(n,closest)
		closest_words[num] = actual_closest
	opfile = open("outputs.csv","w")
	CSVout = csv.writer(opfile)
	for word,relateds in closest_words.items():
		csvl = [word]
		csvl.extend(relateds)
		CSVout.writerow(csvl)
	opfile.close()
