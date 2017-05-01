import numpy as np
import pdb
class Solution:
	# nn_: Point list of nearest neighbor
	# dists_: Float list of distances to nearest neighbor
	nn_ = None
	dists_ = None

	def __init__(self, c,p):
		self.centroids_ = c
		self.points_ = p

	# Partition
	def partition(self):
		for p in self.points_:
			dists = np.array([])
			for c in self.centroids_:
				dist = np.dot(p.xy_-c.xy_,p.xy_-c.xy_)
				dists = np.append(dists, dist)
			centroid = self.centroids_[np.argmin(dists)]
			p.label_ = centroid.label_
		self.update_centroid()

	# Update centroids
	def update_centroid(self):
		for c in self.centroids_:
			lbl = c.label_
			points = self.get_partition(lbl)
			new_c_xy = np.mean([p.xy_ for p in points])
			c.xy_ = new_c_xy

	# Calculate mean square error
	def MSE(self):
		TSE = 0
		for c in self.centroids_:
			p = self.get_partition(c.label_)
			for x in p:
				e = np.dot(x.xy_-c.xy_,x.xy_-c.xy_)
				TSE+=e

		return TSE/self.points_.size


	# Distance from a point to its centroid
	@staticmethod
	def distance_to_centroid(x, c):
		# c = self.get_centroid(x.label_)
		return np.dot(x.xy_-c.xy_,x.xy_-c.xy_)

	# Get partition from label
	def get_partition(self,label):
		return np.array([x for x in self.points_ if x.label_==label])

	# Get centroid point from label
	def get_centroid(self,label):
		return np.array([c for c in self.centroids_ if c.label_==label])

	# Remove cluster with size == 0
	def remove_empty_clusters(self):
		# Get cluster centroid with partition size = 0
		empty_c = np.array([c for c in self.centroids_ if self.get_partition(c.label_).size==0])
		# Get index of these partitions
		empty_c_idx = np.array([c.label_ for c in empty_c])
		# If exists nearest neighbor array, remove that of deleted cluster
		if self.nn_ is not None:
			# empty_nn = self.nn_[empty_c_idx]
			self.nn_ = np.delete(self.nn_,empty_c_idx)
		# Similar to dists array
		if self.dists_ is not None:
			# empty_dists = self.dists_[empty_c_idx]
			self.dists_ = np.delete(self.dists_, empty_c_idx)
		self.centroids_ = np.delete(self.centroids_, empty_c_idx)

	# Relabel from zero to size of centroids
	def relabel(self):
		i = 0
		for c in self.centroids_:
			print 'Rename label %d to label %d'
			p = self.get_partition(c.label_)
			for x in p:
				x.label_=i
			c.label_=i
			i+=1

	# Find nearest neighbor, save list of nearest neighbor and corresponding
	# distance to object attributes nn_ & dists_
	def find_nearest_neighbor(self):
		self.nn_ = np.array([])
		self.dists_ = np.array([])
		for a in self.centroids_:
			na = self.get_partition(a.label_).size
			nn_c = None
			min_dist = -1
			for b in self.centroids_:
				if b!=a:
					nb = self.get_partition(b.label_).size
					dist = (np.dot(a.xy_-b.xy_,a.xy_-b.xy_)*na*nb)/(na+nb)
					if dist<min_dist or min_dist == -1:
						nn_c = b
						min_dist=dist
			self.nn_ = np.append(self.nn_,nn_c)
			self.dists_ = np.append(self.dists_, min_dist)

	# Update nearest neighbor after merge two cluster
	def updateNN(self, lbl1, lbl2):
		for idx,a in enumerate(self.centroids_):
			# For each centroid of solution
			# find a new neighbor only if this has been
			# changed because of merging
			nearest_c = self.nn_[idx]
			if nearest_c.label_==lbl1 or nearest_c.label_==lbl2:
				na = self.get_partition(a.label_).size
				nn_c = None
				min_dist = -1
				for b in self.centroids_:
					if b != a and b!=nearest_c:
						nb = self.get_partition(b.label_).size
						dist = (np.dot(a.xy_ - b.xy_, a.xy_ - b.xy_) * na * nb) / (na + nb)
						if dist < min_dist or min_dist == -1:
							nn_c = b
							min_dist = dist
				self.nn_[idx] = nn_c
				self.dists_[idx] = min_dist

	# Merge two clusters which have smallest merging cost function
	def mergePNN(self):
		# Get current nearest pair
		# lbl1: index of smallest value in dists
		# equivalent to first label of nearest pair
		# for example: dists = [ 3 2 (1) 4 ], nn_ = [c2,c4,(c3),c1]
		# then lbl1 = 2, c2 = nn_[lbl1]= c3 -> lbl2 = c3.label_
		lbl1 = np.argmin(self.dists_)
		c1 = self.centroids_[lbl1]
		c2 = self.nn_[lbl1]
		lbl2 = c2.label_

		p1 = self.get_partition(lbl1)
		n1 = p1.size
		p2 = self.get_partition(lbl2)
		n2 = p2.size
		# Move centroid c1
		cmerge_xy = ((n1*c1.xy_) + (n2*c2.xy_))/(n1+n2)
		c1.xy_ = cmerge_xy
		# Change label of partition
		# no need to change points in p1
		for p in p2:
			p.label_ = c1.label_
		# Update nearest neighbors
		self.updateNN(lbl1,lbl2)
		self.remove_empty_clusters()

	def display(self):
		print([p.label_ for p in self.points_])

class Point:
	def __init__(self,xy,label=-1):
		self.xy_ = xy
		self.label_ = label

	def getX(self):
		return self.xy[0]

	def getY(self):
		return self.xy[1]

	def __str__(self):
		return  "%d: %.2f %.2f" %(self.label_,self.xy_[0], self.xy_[1])
