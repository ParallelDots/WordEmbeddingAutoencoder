import kayak as ky
import numpy as np 

class TrainModel(object):
	def __init__(self,maxnum,reduced_dims):
		self.threshold = 1e-2
		dummyword = np.zeros((maxnum,1))
		W1 = np.random.randn(reduced_dims,maxnum) * 0.1
		W2 = np.random.randn(maxnum,reduced_dims) * 0.1
		self.input = ky.Parameter(dummyword)
		self.W1 = ky.Parameter(W1)
		self.W2 = ky.Parameter(W2)
		self.output = ky.MatMult(self.W1,self.input)
		self.recons = ky.MatMult(self.W2,self.output)
		self.loss = ky.MatSum(ky.L2Loss(self.recons,self.input))
		#self.totloss = ky.MatAdd(self.loss,ky.L2Norm(self.W2,weight=1e-2),ky.L2Norm(self.W1,weight = 1e-2))
		self.totloss = self.loss

	def trainonone(self,wordvec,learnrate=0.4):
		self.input.value = wordvec
		#print wordvec,#" Weights:  ",self.W1.value,self.W2.value
		W1_grad = self.totloss.grad(self.W1)
		print "Weight1: ",W1_grad,"output: ",self.output.value
		W1_grad[W1_grad > self.threshold]= self.threshold
		W1_grad[W1_grad < -1*self.threshold] = -1*self.threshold
		W2_grad = self.totloss.grad(self.W2)
                W2_grad[W2_grad > self.threshold]= self.threshold
                W2_grad[W2_grad < -1*self.threshold] = -1*self.threshold

		#print "Grads: ",W1_grad,W2_grad
		self.W1.value -= learnrate*W1_grad
		self.W2.value -= learnrate*W2_grad
		print "loss: ",self.loss.value

	def getoutput(self,wordvec):
		self.input.value = wordvec
		return self.output.value



