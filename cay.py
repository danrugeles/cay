import numpy as np
from scipy.special import gammaln
from aux import addrandomtopic,dictionate

import time

data=np.array([[0,0],
							[0,1],
							[1,0],
							[1,1],
							[0,1],
							[2,2],
							[2,1],
							[1,2],
							[0,1],
							[1,1],
							[0,1],
							[2,3],
							[3,3],
							[3,2],
							[3,4],
							[4,3],
							[4,4],
							[3,3],
							[3,3],
							[3,2],
							[1,1],
							[1,0],
							[4,4],
							[4,3],
							[1,1],
							[1,0],
							[1,2],
							[2,1],
							[0,1],
							[0,1],
							[2,2],
							[4,3],
							[3,5],
							[4,3],
							[3,2],
							[2,4],
							[4,3],
							[3,3],
							[4,3],
							[4,3],
							[4,3],
							[1,4]])
						

class Model:

	def __init__(self,data,alpha,beta):
	
		#** Preprocess the data
		self.data,idx2vals,vals2idx,self.counts=dictionate(data)	#self.data is dictionated data
		self.V=len(idx2vals[0]) # Total number of observed variables in V
		self.W=len(idx2vals[1]) # Total number of observed variables in W
		
		self.alpha=alpha
		self.beta=beta
	
		# Global parameters	
		self.currV=0 	# Current number of observed variables in V
		self.currW=0 	# Current number of observed variables in W
		self.Vs=set()	# Set of Vs
		self.Ws=set()	# Set of Ws
		self.K=0			# Current number of existing K
		self.nvk_=np.zeros((self.V,self.K)) 
		self.n_kw=np.zeros((self.W,self.K))
		self.n_k_=np.zeros(self.K)
		self.sum_N=0
		self.P_new=self.alpha
	

	# Remove empty columns from structure with the exception of the first column
	def removeEmptyCols(self,idx):
	
	 	assert(np.sum(self.n_kw[:][:,idx]) == 0 and
	 				 np.sum(self.nvk_[:][:,idx]) == 0 and
	 				 self.n_k_[idx] == 0  or 
	 				 (np.sum(self.n_kw[:][:,idx]) != 0 and
	 				 	np.sum(self.nvk_[:][:,idx]) != 0 and
	 				 	self.n_k_[idx] != 0))
	 	
		if np.sum(self.n_kw[:][:,idx]) == 0:
			self.n_kw=np.delete(self.n_kw,(idx),axis=1)
			self.nvk_=np.delete(self.nvk_,(idx),axis=1)
			self.n_k_=np.delete(self.n_k_,(idx))
			self.sum_N=np.delete(self.sum_N,(idx))
			self.data.T[-1][self.data.T[-1]>idx]-=1
			self.K-=1
	

	def update_topic(self,rowid,it):
	
		x,y,currk=self.data[rowid]
		
		#**1. Leave from Current Topic
		self.n_kw[y][currk]-=1
		self.nvk_[x][currk]-=1
		self.n_k_[currk]-=1 
	
		
		# While observing the data construct Set of W and V
		if it==0:
			self.Ws.add(y)
			self.Vs.add(x)
			self.P_new=self.alpha/(len(self.Ws)*len(self.Vs))**2
			self.sum_N=2*self.n_k_+len(self.Ws)*len(self.Vs)*self.beta
		else:
			self.sum_N[currk]-=2
		
		W_=len(self.Ws)*1.0
		V_=len(self.Vs)*1.0
		
		if currk>0:
			#currk needs to be updated as well
			self.removeEmptyCols(currk)
			

		
		Nxy=self.nvk_[x]/W_+self.n_kw[y]/V_+self.beta
		log_Nvu=np.log(self.nvk_/W_+self.n_kw[y]/V_+self.beta+1)
		log_Nxw=np.log(self.nvk_[x]/W_+self.n_kw/V_+self.beta+1)
	
		#* Compute the terms used for calculating the posterior
		A=gammaln(self.sum_N)-gammaln(self.sum_N+W_+V_)
		B=gammaln(Nxy+2)-gammaln(Nxy)
		C=np.sum(log_Nvu,0)+np.sum(log_Nxw,0) 
		
		log_p_z=A+B+C
		
		p_z = np.exp(log_p_z-log_p_z.max()) # it may be optimized if p_z[0] is screwing up
		p_z = np.multiply(self.n_k_,p_z)
		p_z[0] = self.P_new
		p_z = p_z / p_z.sum()
		newk=np.random.multinomial(1, p_z / p_z.sum()).argmax()
		
		
		if newk==0:
	
			self.K+=1
	
			self.n_kw=np.hstack((self.n_kw,np.zeros((self.W,1))))
			self.nvk_=np.hstack((self.nvk_,np.zeros((self.V,1))))
			self.n_k_=np.hstack((self.n_k_,0))
			self.sum_N=np.hstack((self.sum_N,0))
			
			#* Sits at Last Table
			self.n_kw[y][-1]+=1
			self.nvk_[x][-1]+=1
			self.n_k_[-1]+=1
			self.sum_N[-1]=2+len(self.Ws)*len(self.Vs)*self.beta
			self.data[rowid][-1]=self.K
			
		else:
			#* Sits at New Table
			self.n_kw[y][newk]+=1
			self.nvk_[x][newk]+=1
			self.n_k_[newk]+=1
			self.data[rowid][-1]=newk
			
			if it>0:
				self.sum_N[newk]+=2

	
	def inference(self,iterations_max):
		
		#** Initialize the topics
		self.data=np.hstack((self.data,np.zeros((np.shape(self.data)[0],1))))
		self.data=np.asarray(np.asarray(self.data,dtype=np.float),dtype=np.int)
		
		#** Initialize the book-keeping
		self.nvk_=np.array([self.counts[0]]).T
		self.n_kw=np.array([self.counts[1]]).T
		self.n_k_=np.array([np.shape(self.data)[0]])	
	
		#** MAIN LOOP
		for it in range(iterations_max):
			for rowid in range(len(self.data)):	
				self.update_topic(rowid,it)

			print "Iteration",it,"Number of topics",len(self.n_k_)-1
			
		self.printTopics()	
		
		print "\nTopic Allocations"
		print self.data
			
	def loglikelihood(self):
		return 0
	

	def printTopics(self):
		ntopics=len(self.n_k_)-1
		topics=[]
		for i in range(ntopics):
			topics.append(np.zeros((self.V,self.W)))
		
		for row in self.data:
			x,y,t=row
			topics[t-1][x][y]+=1 # given the fact that 0 is not a topic
		
		for i,topic in enumerate(topics):
			np.save("topic"+str(i),topic)
			print "\nTopic "+str(i)+"------------------------ \n",topic	
			print "Row Topic :   ",np.around(np.sum(topic,axis=0),decimals=1)
			print "Column Topic: ",np.around(np.sum(topic,axis=1),decimals=1)
			
	

if __name__=="__main__":

	alpha=0.01 #>0.00001- NIPS or > 0.01 - small toy
	beta=1.0 #150.0 - NIPS or  ~1.2- small toy
	iterations=30
	
	m= Model(data,alpha,beta)
	m.inference(iterations)
	
	print "Likelihood",m.loglikelihood()
	
		
