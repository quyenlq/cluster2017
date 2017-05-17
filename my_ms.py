import numpy as np
import numpy.math as math
import sklearn
import copy
import matplotlib.pyplot as plt




def dist(x1, x2):
    return np.dot(x1-x2,x1-x2);

def nearest_points(X, p, distance = 5):
    eligible_X = []
    for x in X:
        distance_between = dist(x, p)
        if distance_between <= distance:
            eligible_X.append(x)
    return eligible_X

def gaussian_kernel(distance, bandwidth):
    val = (1/(bandwidth*math.sqrt(2*math.pi))) * np.exp(-0.5*((distance / bandwidth))**2)
    return val



def ms(data,look_distance):
	X = np.copy(data)
	past_X = []
	n_iterations = 5
	kernel_bandwidth = sklearn.cluster.estimate_bandwidth(data)
	for it in range(n_iterations):
		for i, x in enumerate(X):
			### Step 1. For each datapoint x ∈ X, find the neighbouring points N(x) of x.
			neighbours = nearest_points(X, x, look_distance)

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

		past_X.append(np.copy(X))


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
look_distance_full =[50000,50000,50000,50000,5000,5000,5000,20000,5000,5000,20]
k_lightweight = [15,15,15,15,20,35,50,8,16];
files_lightweight = ['s1','s2','s3','s4', 'a1', 'a2', 'a3','unbalance','dim32']
look_distance_lightweight =[50000,50000,50000,50000,5000,5000,5000,20000,20]
k_test = [15]
files_test = ['s1']
look_distance_test =[50000]
# look_distance = 6  # How far to look for neighbours.
# kernel_bandwidth = 4  # Kernel parameter.

S = 5
T = 5
GLA = True
GLA_STEPS = 3
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
	for f,c in zip(files,k):
		print 'Running data set %s' %f
		fi = open(f+'.txt')
		fi_gt = open(f+'-gt.txt')
		data = np.loadtxt(fi)
		gt = np.loadtxt(fi_gt)
		sol=ms(data,c)
		CI = ci(sol,gt,c)
		print("FINISH DATASET %s CI:=%d, MSE=%.2f" % (f, CI, sol.MSE_))
		plot_solution(sol,gt)



np.random.seed(2991)
from sys import argv
if len(argv)==2 and (argv[1] == 'full' or argv[1] == 'test'):
	main(argv[1])
else:
	main('lightweight')



