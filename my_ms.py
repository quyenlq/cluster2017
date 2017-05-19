# -*- coding: utf-8 -*-
import numpy as np
import math
import  pdb
from sklearn.cluster import estimate_bandwidth
import matplotlib.pyplot as plt


def dist(x1, x2):
	return math.sqrt(np.dot(x1-x2,x1-x2));

def nearest_points(X, p, distance):
	eligible_X = []
	for x in X:
		distance_between = dist(x, p)
		if distance_between <= distance:
			eligible_X.append(x)
	return eligible_X

def gaussian_kernel(distance, bandwidth):
	val = (1/(bandwidth*math.sqrt(2*math.pi))) * np.exp(-0.5*((distance / bandwidth))**2)
	return val



def ms(data,look_distance,gt):
	X = np.copy(data)
	n_iterations = 10
	kernel_bandwidth = estimate_bandwidth(X)
	print("Kernel bandwidth: %.3f" % kernel_bandwidth)
	for it in range(n_iterations):
		for i, x in enumerate(X):
			### Step 1. For each datapoint x ∈ X, find the neighbouring points N(x) of x.
			neighbours = nearest_points(X, x, look_distance/2)
			### Step 2. For each datapoint x ∈ X, calculate the mean shift m(x).
			numerator = 0
			denominator = 0
			for neighbour in neighbours:
				distance = dist(neighbour, x)
				weight = gaussian_kernel(distance, kernel_bandwidth)
				numerator += (weight * neighbour)
				denominator += weight

			new_x = numerator / denominator

			### Step 3. For each datapoint x ∈ X, update x ← m(x).
			X[i] = new_x
		if it>5:
			plot_solution(X,gt)
	return X


def plot_solution(X,gt):
	plt.figure()
	# cx = np.array([c.xy_ for c in sol.centroids_])
	# for c in sol.centroids_:
	# 	partition = sol.get_partition(c.label_)
	# 	pxy = np.array([[p.getX(), p.getY()] for p in partition])
	# 	# pdb.set_trace()
	# 	plt.scatter(pxy[:, 0], pxy[:, 1], s=2, marker='.', c='b')
	plt.scatter(X[:, 0], X[:, 1], marker='.', c='r')
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
look_distance_full =[50000,50000,50000,50000,5000,5000,5000,20000,5000,5000,20]
k_lightweight = [15,15,15,15,20,35,50,8,16];
files_lightweight = ['s1','s2','s3','s4', 'a1', 'a2', 'a3','unbalance','dim32']
look_distance_lightweight =[50000,50000,50000,50000,5000,5000,5000,20000,20]
k_test = [15]
files_test = ['s1']
look_distance_test =[50000]
# look_distance = 6  # How far to look for neighbours.
# kernel_bandwidth = 4  # Kernel parameter.

def main(arg):
	if arg=='full':
		print("Use full dataset, might be slow")
		files=files_full
		k=k_full
		look_distance = look_distance_full
	elif arg=='test':
		print("Testing algorith, only use S1")
		files=files_test
		k=k_test
		look_distance = look_distance_test
	else:
		print("Lightweight mode, skip Birch1 and Birch2")
		files = files_lightweight
		k = k_lightweight
		look_distance = look_distance_lightweight
	for f,c,d in zip(files,k,look_distance):
		print 'Running data set %s' %f
		fi = open(f+'.txt')
		fi_gt = open(f+'-gt.txt')
		data = np.loadtxt(fi)
		gt = np.loadtxt(fi_gt)
		X=ms(data, d,gt)
		plot_solution(X,gt)
		# CI = ci(sol,gt,c)
		# print("FINISH DATASET %s CI:=%d, MSE=%.2f" % (f, CI, sol.MSE_))
		# plot_solution(sol,gt)



np.random.seed(2991)
from sys import argv
if len(argv)==2 and (argv[1] == 'full' or argv[1] == 'test'):
	main(argv[1])
else:
	main('lightweight')



