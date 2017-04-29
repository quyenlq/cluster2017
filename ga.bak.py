import numpy as np
import matplotlib.pyplot as plt
import pdb


class Solution:
	def __init__(self, c,p,data,M):
		self.M_ = M
		self.centroids_ = c
		self.partitions_ = p
		self.data_ = data
		self.distortion_=0
		self.calculate_distortion()


	def get_partition(self,label):
		return self.data_[self.partitions_==label]

	def calculate_distortion(self):
		C=self.centroids_
		P=self.partitions_
		sse=0
		for i in range(self.M_):
			c = C[i]
			p = self.get_partition(i)
			sse+=sum([np.dot(x-c,x-c) for x in p])
		self._distortion = sse
		return sse

	def pnn(self,size_ok):
		# Find closest neighbors Q of Cnew
		Q = np.array([])
		QCost = np.array([])
		for labela,ca in enumerate(self.centroids_):
			# Array of costs from Ca to another point
			min_cost= -1
			q=-1
			for labelb,cb in enumerate(self.centroids_):
				if not labela==labelb:
					cost=self.cost(labela,labelb)
					if (cost<min_cost or min_cost ==-1): 
						q = labelb
						min_cost=cost
			Q=np.append(Q,q)
			QCost = np.append(QCost,min_cost)
		while(self.M_>size_ok):
			# Find minimum distance
			centroid_merge=np.argmin(QCost)
			centroid_to_merge=Q[centroid_merge]
			self.merge(centroid_merge,centroid_to_merge)


	def merge(self,cen1,cen2):




	def cost(self,label1,label2):
		na = self.data_[self.partitions_==label1].shape[0]
		nb = self.data_[self.partitions_==label2].shape[0]
		ca= self.centroids_[label1]
		cb= self.centroids_[label2]
		cost = (na*nb)/(na+nb)*np.dot(ca-cb,ca-cb)
		return cost



def centroids_index(cens_A, cens_B, n_clus):
	return n_clus - np.unique(np.array([np.argmin([np.dot(cenA-cenB,cenA-cenB) for cenA in cens_A]) for cenB in cens_B])).size

def partition(C,X):
	return np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in C]) for x_i in X])

def ga(data,M,S=45,T=100):
	solutions = np.array([])
	# Generate initial solutions
	for i in range(S):
		c = data[np.random.choice(np.arange(len(data)),M)]
		p = partition(c,data)
		s = Solution(c,p,data,M)
		solutions=np.append(solutions,s)
	for i in range(T):
		solutions = gen_new_solutions(solutions,data,M,S)
		# solutions = solutions[np.argsort([S.get_dist(data) for S in solutions])]
		# np.append(final_solutions,solutions[0])
	# Return solution with smallest distortion 
	return solutions[np.argsort([S.get_dist(data) for S in solutions])][0]

def gen_new_solutions(solutions,data,M,S):
	for i in range(S):
		# pdb.set_trace();
		np.random.shuffle(solutions)
		pair= solutions[:2]
		cross_solution(pair[0], pair[1],data,M)

def cross_solution(s1,s2,data,M):
	# Get centroids of solutions
	c1 = s1.centroids_
	c2 = s2.centroids_
	# Get partition of solutions
	p1 = s1.partitions_
	# Simple trick to change labels of partition2
	# x belongs to partition 0 of p2 => partition M of pnew
	p2 = s2.partitions_+M
	# Cnew = C1 union C2 , size 2M
	cnew = np.append(c1,c2,axis=0)
	# Combine partition
	pnew = combine_partition(cnew,p1,p2,data)
	# Recalculate centroids
	cnew = update_centroids(cnew,pnew,2*M,data)
	# Forming new solutions, size of cluster is doubled:
	new_solution = Solution(cnew, pnew, data, 2*M)
	new_solution.pnn(M)
	# remove empty cluster
	# for i in range(2*M):
	# 	if pnew[pnew==i].size==0:
	# 		cnew=np.delete(cnew,cnew[i])
	
	# pnn(cnew,pnew,data,M)

def combine_partition(cnew,p1,p2,data):
	pnew = np.array([])
	# For each x data point
	for idx,x in enumerate(data):
		# Select centroids of partition which x belongs
		c1x= cnew[p1[idx]]
		c2x= cnew[p2[idx]]
		# Compare which centroid is closer to x, then assign x to that partition
		if (np.dot(x-c1x,x-c1x) <= np.dot(x-c2x,x-c2x)):
			pnew=np.append(pnew,p1[idx])
		else: 
			pnew=np.append(pnew,p2[idx])
	# pnew now has labels form 0 to M-1
	return pnew

def update_centroids(cnew,pnew,M,data):
	return np.array([data[pnew == l].mean(axis = 0) for l in range(M)])


def run_ga(file,M):
	f = open(file+'.txt')
	fgt = open(file+'-gt.txt')
	data = np.loadtxt(f)
	gt = np.loadtxt(fgt)

	# solution=ga(data,M)
	solution=ga(data,M,S=2,T=1)
	c=solution.centroids_
	p=solution.partition_
	ci=max(centroids_index(centroids,gt,k),centroids_index(gt,centroids,k))
	print 'CI: %.2f'  % ci




Sc=9
S=(Sc*(Sc+1))/2
T=100
np.random.seed(0)
k = [15,15,15,15,20,35,50,8,100,100,16];
files = ['s1','s2','s3','s4', 'a1', 'a2', 'a3','unbalance','birch1', 'birch2','dim32']
T = 1
def main():
	# np.random.seed(1)
	for t in range(T):
		for f,c in zip(files,k):
			print 'Running data set %s' %f
			run_ga(f,c)

main()
