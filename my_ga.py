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
	solutions = np.array([])
	for s in range(S):
		# Gen S seed data
		new_solution = gen_initial_solution(data,K)
		new_solution.partition()
		solutions=np.append(solutions,new_solution)
	top_solutions = np.array([])
	for t in range(T):
		print 'ITERATION #%d'%t
		print 'Start generate %d new solutions from population set'%S
		new_solutions=gen_new_solutions(solutions,K)
		print 'Gen completed, picked best, remove the rest'
		best_solution = get_best_solutions(new_solutions)
		top_solutions=np.append(top_solutions,best_solution)
	print 'Finish, return best of bests'
	return get_best_solutions(top_solutions)

def get_best_solutions(sols):
	# print [s.MSE() for s in sols]
	return sols[np.argmin([s.MSE() for s in sols])]

# Gen new solutions from initial solutions
# repeat S times to generate S solutions
def gen_new_solutions(solutions,K):
	new_solutions = np.array([])
	for i in range(S):
		# print 'Pick pair of solutions %d' %i
		solution_pair = get_pair(solutions)
		new_solution = cross_over(solution_pair,K)
		new_solutions=np.append(new_solutions,new_solution)
	return new_solutions

# Crossover 2 solution to generate one solution
def cross_over(pair,K):
	# print 'Start crossover, choose from sol pair'
	sol1 = copy.deepcopy(pair[0])
	sol2 = copy.deepcopy(pair[1])
	combined_centroids = combine_centroids(sol1.centroids_,sol2.centroids_)
	combined_partitions = combine_partitions(sol1,sol2,combined_centroids)
	sol = Solution(combined_centroids,combined_partitions)
	sol.update_centroid()
	sol.remove_empty_clusters()
	# print 'Remove empty cluster, now sol has %d clusters' %(sol.centroids_.size)
	sol.relabel()
	# print 'Relabel cluster'
	return pnn(sol,K)


def pnn(solution, K):
	solution.find_nearest_neighbor()
	while solution.centroids_.size > K:
		solution.mergePNN()
		# print("merge 2 cluster, cluster left: %d"%solution.centroids_.size)
	solution.relabel()
	return solution


def get_pair(solutions):
	pair = np.random.choice(solutions, 2)
	while pair[0]==pair[1]:
		pair = np.random.choice(solutions, 2)
	return pair

def combine_centroids(c1,c2):
	add_lbl = c1.size
	c_new = np.copy(c2)
	for c in c_new:
		c.label_+=add_lbl
	c_com = np.append(c1,c_new)
	return c_com

def combine_partitions(sol1,sol2,c_com):
	k = c_com.size/2
	N = sol1.points_.size
	combine_p = np.array([])
	for i in range(N):
		# x1,x2 are the same points
		# x: new point
		x = sol1.points_[i]
		d1 = sol1.distance_to_centroid(x,c_com[x.label_])
		d2 = sol2.distance_to_centroid(x,c_com[x.label_+k])
		if d1 < d2:
			x_new = Point(x.xy_, x.label_)
		else:
			x_new = Point(x.xy_, x.label_+ k)
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
			d = np.linalg.norm(cenA-cenB)
			dist.append(d);
		mapTo = dist.index(min(dist))
		orphans[mapTo]=0;
	return sum(orphans)

################################################################################################
# MAIN
################################################################################################


k = [15,15,15,15,20,35,50,8,100,100,16];
files = ['s1','s2','s3','s4', 'a1', 'a2', 'a3','unbalance','birch1', 'birch2','dim32']
S = 45
T = 10
def main():
	np.random.seed(1)
	for t in range(T):
		for f,c in zip(files,k):
			print 'Running data set %s' %f
			fi = open(f+'.txt')
			fi_gt = open(f+'-gt.txt')
			data = np.loadtxt(fi)
			gt = np.loadtxt(fi_gt)
			sol=ga(data,c)
			CI = ci(sol,gt,c)
			# plot_solution(sol,gt)
			print("FINISH CI:=%d, MSE=%.2f"%(CI,sol.MSE()))
			# break

main()

