import numpy as np
import copy
import matplotlib.pyplot as plt
import pdb
from solution import Solution, Point


# M: number of clusters
# return a new solution with M random centroids
def gen_initial_solution(data, K):
	random_centroids = data[np.random.choice(np.arange(len(data)),K)]
	centroid_points = np.array([])
	i = 0
	for c in random_centroids:
		centroid_point = Point(c, label=i)
		i+=1
		centroid_points = np.append(centroid_points,centroid_point)
	# Point class wrapper
	data_points = np.array([Point(x) for x in data])
	return Solution(centroid_points,data_points)

# f = data, c = number of cluster, k: number of cluster
def ga(data,K):
	population = np.array([])
	for s in range(S):
		# Gen S seed data
		new_solution = gen_initial_solution(data,K)
		new_solution.partition()
		new_solution.calculate_MSE()
		population=np.append(population,new_solution)
	# top_solutions = np.array([])
	for t in range(T):
		print 'GA Iter #%d'%t
		#print 'Start generate %d new solutions from population set'%(S*(S-1)/2)
		ancestors=gen_new_generation(population,K)
		#print 'Generated %d ancestors' %ancestors.size
		population = np.append(population,ancestors)
		#print 'Sort population'
		# pdb.set_trace()
		population.sort()
		# sorted(population, key=lambda x: tuple(x.MSE_))
		#print 'Remove weak ancestor'
		population = population[:S]
		# print 'Gen completed, picked best, remove the rest'
		# best_solution = get_best_solutions(new_solutions)
		# top_solutions=np.append(top_solutions,best_solution)s
	return population[0]

# Gen new solutions from initial solutions
# repeat S times to generate S solutions
def gen_new_generation(solutions,K):
	ancestors = np.array([])
	for i in range(S):
		for j in range(S):
			if j>i:
				parent = get_pair(solutions,i,j)
				ancestor = cross_over(parent, K)
				ancestors = np.append(ancestors, ancestor)
	return ancestors

# Crossover 2 solution to generate one solution
def cross_over(pair,K):
	# print 'Start crossover, choose from sol pair'
	sol1 = copy.deepcopy(pair[0])
	sol2 = copy.deepcopy(pair[1])
	combined_centroids = combine_centroids(sol1.centroids_,sol2.centroids_,K)
	combined_partitions = combine_partitions(sol1,sol2,combined_centroids,K)
	sol = Solution(combined_centroids,combined_partitions)
	sol.update_centroid()
	sol.remove_empty_clusters()
	# print 'Remove empty cluster, now sol has %d clusters' %(sol.centroids_.size)
	sol.relabel()
	# print 'Relabel cluster'
	sol=pnn(sol, K)
	if GLA:
		sol.GLA(GLA_STEPS)
	return sol


def pnn(solution, K):
	solution.find_nearest_neighbor()
	while solution.centroids_.size > K:
		solution.mergePNN()
		# print("merge 2 cluster, cluster left: %d"%solution.centroids_.size)
	solution.relabel()
	solution.calculate_MSE()
	return solution


def get_pair(solutions,i,j):
	return np.array([solutions[i], solutions[j]])

def combine_centroids(c1,c2,K):
	c_new = copy.deepcopy(c2)
	for c in c_new:
		c.label_+=K
	c_com = np.append(c1,c_new)
	return c_com

def combine_partitions(sol1,sol2,c_com,K):
	N = sol1.points_.size
	combine_p = np.array([])
	for i in range(N):
		# x1,x2 are the same points
		# x: new point
		x = sol1.points_[i]
		d1 = sol1.distance_to_centroid(x,c_com[x.label_])
		d2 = sol2.distance_to_centroid(x,c_com[x.label_+K])
		if d1 < d2:
			x_new = Point(x.xy_, x.label_)
		else:
			x_new = Point(x.xy_, x.label_+ K)
		combine_p=np.append(combine_p,x_new)
	return combine_p

def plot_solution(sol,gt):
	plt.figure()
	cx = np.array([c.xy_ for c in sol.centroids_])
	for c in sol.centroids_:
		partition = sol.get_partition(c.label_)
		pxy = np.array([[p.getX(), p.getY()] for p in partition])
		# pdb.set_trace()
		plt.scatter(pxy[:, 0], pxy[:, 1], s=2, marker='.', c='b')
	plt.scatter(cx[:, 0], cx[:, 1], marker='.', c='r')
	plt.scatter(gt[:, 0], gt[:, 1], marker='.', c='m', )
	plt.show()

def ci(sol,gt,k):
	cens = np.array([c.xy_ for c in sol.centroids_])
	return max(one_way_ci(cens,gt,k),one_way_ci(gt,cens,k))

def one_way_ci(cens_A, cens_B, n_clus):
	orphans = np.ones(n_clus);
	for cenA in cens_A:
		dist = []
		for cenB in cens_B:
			d = np.dot(cenA-cenB,cenA-cenB)
			dist.append(d);
		mapTo = dist.index(min(dist))
		orphans[mapTo]=0;
	return sum(orphans)

################################################################################################
# MAIN
################################################################################################


k_full = [15,15,15,15,20,35,50,8,100,100,16];
files_full = ['s1','s2','s3','s4', 'a1', 'a2', 'a3','unbalance','birch1', 'birch2','dim32']
k_lightweight = [15,15,15,15,20,35,50,8,16];
files_lightweight = ['s1','s2','s3','s4', 'a1', 'a2', 'a3','unbalance','dim32']
k_test = [15]
files_test = ['s1']


S = 9
T = 10
GLA = False 
GLA_STEPS = 3
ITER = 5
def main(arg):
	if arg=='full':
		print("Use full dataset, might be slow")
		files=files_full
		k=k_full
	elif arg=='test':
		print("Testing algorith, only use S1")
		files=files_test
		k=k_test
	else:
		print("Lightweight mode, skip Birch1 and Birch2")
		files = files_lightweight
		k = k_lightweight
	for i in range(ITER):
		print("ITERATION #%d"%i)
		for f,c in zip(files,k):
			print 'Running data set %s' %f
			fi = open(f+'.txt')
			fi_gt = open(f+'-gt.txt')
			data = np.loadtxt(fi)
			gt = np.loadtxt(fi_gt)
			sol=ga(data,c)
			CI = ci(sol,gt,c)
			# plot_solution(sol,gt)
			print("FINISH DATASET %s CI:=%d, MSE=%.2f"%(f,CI,sol.MSE_))

#np.random.seed(2991)
from sys import argv
if len(argv)==2 and (argv[1] == 'full' or argv[1] == 'test'):
	main(argv[1])
else:
	main('lightweight')



