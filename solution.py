import numpy as np

class Solution:
	def __init__(self, c,p):
		self.centroids_ = c
		self.points = p

	def MSE(self):
		return 0

	def get_partition(label):
		return [for x in points if x.label_==label]

	def get_centroid(label):
		return [for c in centroids if c.label_==label]

class Point:
	def __init__(self,xy,label=-1):
		self.xy_ = xy
		self.label_ = label

	def getX(self):
		return self.xy[0]

	def getY(self):
		return self.xy[1]
