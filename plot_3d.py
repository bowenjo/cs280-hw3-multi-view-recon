import numpy as np 
import cv2

import matplotlib
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

def plot_3d(points, R, t):
	'''
	Plots 3D point cloud of actual coordinates of correspondences between two cameras.

	Also plots center for two cameras.

	INPUT:
	- points: Nx3 array, 3D points to be plotted
	- cam_centers: 4x3 array, 3D points of camera centers
	'''
	cam1_back = np.array([0, 0, 0])
	cam1_front = np.array([0, 0, .1])

	cam2_back = cam1_back @ R.T + t
	cam2_front = cam1_front @ R.T + t

	cam_centers = np.vstack((cam1_back, cam1_front, cam2_back, cam2_front))

	# plot the points and the camera positions
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	ax.scatter(points[:,0], points[:,1], points[:,2], c='b', marker='.', alpha=.3)
	ax.plot(cam_centers[0:2,0], cam_centers[0:2,1], cam_centers[0:2,2], c='r', lw=10)
	ax.plot(cam_centers[2:,0], cam_centers[2:,1], cam_centers[2:,2], c='g', lw=10)

	ax.set_xlim([-5,5])
	ax.set_ylim([-5,5])
	ax.set_zlim([-1,6])

	plt.show()
	
