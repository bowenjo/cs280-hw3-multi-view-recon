import numpy as np 
import cv2

import matplotlib
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

def plot_3d(points, cam_centers):
	'''
	Plots 3D point cloud of actual coordinates of correspondences between two cameras.

	Also plots center for two cameras.

	INPUT:
	- points: Nx3 array, 3D points to be plotted
	- cam_centers: 4x3 array, 3D points of camera centers
	'''

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	print('num points: {}'.format(points.shape))

	ax.scatter(points[:,0], points[:,1], points[:,2], c='b', marker='.', alpha=.3)
	ax.plot(cam_centers[0:2,0], cam_centers[0:2,1], cam_centers[0:2,2], c='r')
	ax.plot(cam_centers[2:,0], cam_centers[2:,1], cam_centers[2:,2], c='g')

	plt.show()
	
