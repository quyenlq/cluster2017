import numpy as np
import matplotlib.pyplot as plt
import pdb


class Centroid:
	def __init__(self, label,x):
		self.label_=label;
		self.x_ = x;


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
			cx = c.x_
			p = self.get_partition(i)
			sse+=sum([np.dot(x-cx,x-cx) for x in p])
		self.distortion_ = sse
		return sse

	def pnn(self,size_ok):
		# Find closest neighbors Q of Cnew
		Q,QCost= self.calculate_cost() 
		# print QCost
		while(self.M_>size_ok):
			# Find minimum distance
			centroid_merge=self.centroids_[np.argmin(QCost)]
			centroid_to_merge=Q[np.argmin(QCost)]
			self.merge(centroid_merge,centroid_to_merge)
			Q,Qcost=self.calculate_cost()
			self.M_-=1

	def calculate_cost(self):
		Q = np.array([])
		QCost = np.array([])
		for ca in self.centroids_:
			# Array of costs from Ca to another point
			min_cost= -1
			q=-1
			for cb in self.centroids_:
				if not ca.label_==cb.label_:
					cost=self.cost(ca,cb)
					if (cost<min_cost or min_cost ==-1): 
						q = cb
						min_cost=cost
			Q=np.append(Q,q)
			QCost = np.append(QCost,min_cost)
		return Q,QCost

	def merge(self,cen1,cen2):
		# p1 = self.get_partition(cen1.label)
		p2 = self.get_partition(cen2.label_)
		p2=cen1.label_
		# remove cluster
		for c in self.centroids_:
			if c.label_==cen2.label_:
				self.centroids_=np.delete(self.centroids_,np.where(self.centroids_==c))
				break


	def cost(self,ca,cb):
		na = self.get_partition(ca.label_).shape[0]
		nb = self.get_partition(cb.label_).shape[0]
		cost = (na*nb)/(na+nb)*np.dot(ca.x_-cb.x_,ca.x_-cb.x_)
		return cost



# def centroids_index(cens_A, gt, n_clus):
	# # for ca in cens_A:
	# # 	print ca.x_ - gt[0] 
	# for cenB in gt:
	# 	for cenA in cens_A:

	# print np.unique(np.array([np.argmin([np.dot(cenA.x_-cenB,cenA.x_-cenB) for cenA in cens_A]) for cenB in gt])).size
	# # print "hahaha"
	# # return n_clus - np.unique(np.array([np.argmin([np.dot(cenA.x_-cenB,cenA.x_-cenB) for cenA in cens_A]) for cenB in gt])).size
	# return 0

def partition(C,X):
	return np.array([np.argmin([np.dot(x_i-y_k.x_, x_i-y_k.x_) for y_k in C]) for x_i in X])

def ga(data,M,S=45,T=100):
	solutions = np.array([])
	# Generate initial solutions
	for i in range(S):
		xs = data[np.random.choice(np.arange(len(data)),M)]
		c = np.array([Centroid(lbl,x) for lbl,x in enumerate(xs)])
		p = partition(c,data)
		s = Solution(c,p,data,M)
		print s.distortion_
		solutions=np.append(solutions,s)
	for i in range(T):
		solution = gen_new_solutions(solutions,data,M,S)
		np.append(solutions, solution)
		# solutions = solutions[np.argsort([S.get_dist(data) for S in solutions])]
		# np.append(final_solutions,solutions[0])
	# Return solution with smallest distortion 
	return solutions[np.argsort([S.calculate_distortion() for S in solutions])][0]

def gen_new_solutions(solutions,data,M,S):
	new_solutions= np.array([])
	for i in range(S):
		# pdb.set_trace();
		np.random.shuffle(solutions)
		pair= solutions[:2]
		new_solution =cross_solution(pair[0], pair[1],data,M)
		np.append(new_solutions,new_solution)
	return new_solutions

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
	return new_solution

def combine_partition(cnew,p1,p2,data):
	pnew = np.array([])
	# For each x data point
	for idx,x in enumerate(data):
		# Select centroids of partition which x belongs
		c1x= cnew[p1[idx]]
		c2x= cnew[p2[idx]]
		# Compare which centroid is closer to x, then assign x to that partition
		if (np.dot(x-c1x.x_,x-c1x.x_) <= np.dot(x-c2x.x_,x-c2x.x_)):
			pnew=np.append(pnew,p1[idx])
		else: 
			pnew=np.append(pnew,p2[idx])
	# pnew now has labels form 0 to M-1
	return pnew

def update_centroids(cnew,pnew,M,data):
	cnew = np.array([])
	for i in range(M):
		x = data[pnew == i].mean(axis = 0)
		c = Centroid(i,x)
		cnew=np.append(cnew,c)
	return cnew


def run_ga(file,M):
	f = open(file+'.txt')
	fgt = open(file+'-gt.txt')
	data = np.loadtxt(f)
	gt = np.loadtxt(fgt)
	# solution=ga(data,M)
	solution=ga(data,M,S=2,T=10)
	cx = np.array([c.x_ for c in solution.centroids_])
	p=solution.partitions_
	print solution.calculate_distortion()
	# ci=max(centroids_index(c,gt,k),centroids_index(gt,c,k))
	# print 'CI: %.2f'  % ci
	plt.figure()
	for i in range(0,M):
		mask = (p == i)
		c = data[mask]
		plt.scatter(c[:, 0], c[:, 1], s=2, marker='.', c='b')
	plt.scatter(cx[:,0], cx[:,1], marker='.', c='r')
	plt.scatter(gt[:,0], gt[:,1],marker='.', c='m', )
	plt.show()




Sc=9
S=(Sc*(Sc+1))/2
T=100
np.random.seed(0)
k = [15,15,15,15,20,35,50,8,100,100,16];
files = ['s1','s2','s3','s4', 'a1', 'a2', 'a3','unbalance','birch1', 'birch2','dim32']
T = 1
def main():
	np.random.seed(1)
	for t in range(T):
		for f,c in zip(files,k):
			print 'Running data set %s' %f
			run_ga(f,c)
			break;

main()
