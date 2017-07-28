'''
the scrip is the program of
fuzzzy k-means for gray image segemntation
2017.7.26
'''

from __future__ import division
import numpy as np
import random
import math

######################################

def initialization(X,n_cluster,init):
	'''select n_cluster samples as the cluster centers,and set the 
		membership matrix as zero matrix

		Parameter :
		----------
		X : matrix
			the data to be clusterd
		n_cluster,init
		returns :
		---------
		cluster_center,member_degree
	'''
	image_matrix=X
	shape=image_matrix.shape
	cluster_center=[]
	member_degree=np.zeros((shape[0],shape[1],n_cluster))
	if init=='random':
		cluster_row=random.sample(range(shape[0]),n_cluster)
		cluster_col=random.sample(range(shape[1]),n_cluster)
		for k in range(n_cluster):
			cluster_center.append(image_matrix[cluster_row[k],cluster_col[k]])

		return cluster_center,member_degree
	else:
		for i in range(n_cluster):
			cluster_center.append(random.randint(0,256))

		return cluster_center,member_degree



def update(X,cluster_center,member_degree,n_cluster,m_fuzzy):
	'''update the cluster center matrix and membership matrix for each iteration
		Parameter:
		----------
		X,cluster_center,member_degree,n_cluster,m_fuzzy

		returns:
		--------
		cluster_center,member_degree
	'''

	#Calculate membership degree matrix
	image_matrix=X
	for i in range(image_matrix.shape[0]):
		for j in range(image_matrix.shape[1]):
			for k in range(n_cluster):
				up=abs(image_matrix[i,j]-cluster_center[k])
				s=0
				for c in range(n_cluster):
					down=abs(image_matrix[i,j]-cluster_center[c])
					s=s+math.pow(up/down,2/(m_fuzzy-1))
				member_degree[i,j,k]=1/s
				
	#Recalculate the cluster center
	for k in range(n_cluster):
		up=np.multiply(np.power(member_degree[:,:,k],m_fuzzy),image_matrix)
		down=np.power(member_degree[:,:,k],m_fuzzy)
		cluster_center[k]=up.sum()/down.sum()      
	
	return cluster_center,member_degree

def fuzzy_kmeans(X,n_iter,n_cluster,m_fuzzy,init):
	'''iteration and start clustering
		parameter :
		-----------
		n_iter,n_cluster,m_fuzzy

		returns :
		----------
		cluster_center,member_degree,labels
	'''
	cluster_center,member_degree=initialization(X,n_cluster,init)
	i=1
	while i<=n_iter:
		cluster_center,member_degree=update(X,cluster_center,member_degree,n_cluster,m_fuzzy)
		print('iteration times:%d'%i)
		i=i+1
	labels=np.argmax(member_degree,axis=2)

	return cluster_center,member_degree,labels

class Fuzzy_Kmeans(object):
	''' 2D fuzzy kmeans cluster

	Parameter:
	---------
	n_cluster : int,default:2

	n_iter : int,default:10

	m_fuzzy : int,dafault:2
		control the fuzzy degress of cluster

	init :['random','array']
		the initilization method

	Attribute:
	---------
	cluster_center_ : array,[1_clusters...n_cluster]
		value of pixel

	member_degree_ : 3D matrix,[image_shape[0],image_shape[1],n_cluster]
		degree of menbership for each point of im

	labels_ :matrix
		the value is the class index of each point in image matrix
	
	examples:
	-------
	'''
	def __init__(self,init,n_cluster=2,n_iter=100,m_fuzzy=2):
		self.n_cluster=n_cluster
		self.n_iter=n_iter
		self.m_fuzzy=m_fuzzy
		self.init=init

	def fit(self,X):
		'''computer the cluster
		Parameter:
		--------
		X:matrix
			the data to be clustered
		'''
		self.cluster_center_,self.member_degree_,self.labels_= \
			fuzzy_kmeans(X,self.n_iter,self.n_cluster,self.m_fuzzy,self.init)

		return self

	def predict(self):
		'''get the cluster labels
			return : labels_
		'''
		return self.labels_


	
