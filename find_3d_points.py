import numpy as np 
import scipy

def find_3d_points(P1,P2,matches):
	"""
	performs triangfulation given two camera matrices and collection of corresponding points

	INPUTS:
	--------------------------
	P1: 3x4 numpy array
		camera matrix for image 1
	P2: 3x4 numpy array
		camera matrix for image 2
	matches: Nx4 numpy array
		corresponding points array for two images 
		where matches[i,:2] is a point (w,h) in the first image
		and matches[i,2:] is the corresponding point in the second image

	OUTPUTS:
	-------------------------
	points_3d.T: Nx3 numpy array
		3D point locations
	rec_err: float
		mean reconstruction error (mean distance between the 2D points and the projected 3D points in the two images.)
	"""

	N = matches.shape[0]

	# initialize 3D point array
	points_3d = np.zeros((3,N))
	# initialize reconstruction error array
	rec_err = np.zeros(N)

	# Loop over each corresponding point pair and triangulate the 3d point
	for i in range(N):
		x1 = matches[i,:2, None]
		x2 = matches[i,2:, None]

		# build the linear system 
		A = np.array([x1[0]*P1[2,:] - P1[0,:],
					  x1[1]*P1[2,:] - P1[1,:],
					  x2[0]*P2[2,:] - P2[0,:],
					  x2[1]*P2[2,:] - P2[1,:]])

		# take SVD of A
		U,S,V = scipy.linalg.svd(A)
		X = V[3,:,None] # X is the right singular vector associated with the smallest singular value of A

		# store the points
		points_3d[:,i, None] = X[0:3]/X[3] 

		# calculate reconstruction error
		pr1 = P1@X # projection in first image
		pr1 = pr1[0:2] / pr1[2] # convert back to non-homogenous
		pr2 = P2@X # projection in second image
		pr2 = pr2[0:2] / pr2[2] # convert back to non-homogenous

		rec_err[i] =  (np.linalg.norm(x1 - pr1) + np.linalg.norm(x2 - pr2)) / 2 # mean distance between projection of 3D point and 2D point in each image

	# take the average reconstruction error across all points 
	rec_err = np.mean(rec_err)

	return(points_3d.T, rec_err)







