import numpy as np 
import cv2


def fundamental_matrix(matches):
	"""
	Computes the fundamental matrix and residual error given a collection of corresponding points

	INPUT:
	-----------
	matches: Nx4 numpy array
		corresponding points array for two images 
		where matches[i,:2] is a point (w,h) in the first image
		and matches[i,2:] is the corresponding point in the second image

	OUTPUT:
	-----------
	F: 3x3 numpy array
		fundamental matrix
	res_err: float
		mean squared distance between points in the two images and their corresponding epipolar lines 
	"""

	## Normalize
   	## -------------------------------------------------------------
	N = matches.shape[0]

	# convert 2d points to homogeneous coordinates
	pts_1_orig = np.hstack((matches[:, :2], np.ones((N,1)))).T
	pts_2_orig = np.hstack((matches[:, 2:], np.ones((N,1)))).T

	# just to compare residuals against ---> delete in final draft
	F_opencv, mask = cv2.findFundamentalMat(pts_1_orig.T,pts_2_orig.T,cv2.FM_8POINT)

	# find normalization matrices
	T_1 = normalization_matrix(pts_1_orig)
	T_2 = normalization_matrix(pts_2_orig)

	# normalize points
	pts_1 = T_1 @ pts_1_orig
	pts_2 = T_2 @ pts_2_orig

	## Optimize
	## ------------------------------------------------------------
	A = np.array([pts_1[0,:]*pts_2[0,:], 
				  pts_1[1,:]*pts_2[0,:], 
				  pts_2[0,:],
				  pts_1[0,:]*pts_2[1,:],
				  pts_1[1,:]*pts_2[1,:],
				  pts_2[1,:],
				  pts_1[0,:],
				  pts_1[1,:],
				  np.ones(N)]).T

	U,S,V = np.linalg.svd(A)
	F = V[8,:].reshape(3,3) # min F is row in V corresponding to smallest singular value

	# constrain F to having rank 2
	U,S,V = np.linalg.svd(F)
	F = U@np.diag([S[0], S[1], 0])@V

	## Denormalize
	## ------------------------------------------------------------
	F = T_2.T @ F @ T_1

	## Residual Error
	## ------------------------------------------------------------
	d_1_2 = np.diag(pts_2_orig.T @ F @ pts_1_orig) / np.linalg.norm(F @ pts_1_orig, axis=0)
	d_2_1 = np.diag(pts_1_orig.T @ F.T @ pts_2_orig) / np.linalg.norm(F.T @ pts_2_orig, axis=0)

	
	# confusion on piazza over correct formulation of residual. Below is the handout's version. 
	# d_1_2 = np.abs(np.diag(pts_1_orig.T @ F @ pts_2_orig)) / np.linalg.norm(F @ pts_2_orig, axis=0)
	# d_2_1 = np.abs(np.diag(pts_2_orig.T @ F @ pts_1_orig)) / np.linalg.norm(F @ pts_1_orig, axis=0)

	res_err = np.sum(d_1_2**2 + d_2_1**2) / 2*N

	return(F, res_err)


def normalization_matrix(pts):
	# find shift to origin
	mu = np.mean(pts, axis=1).reshape(3,1)

	# scale the mean distance to  sqrt(2)
	mean_dist = np.mean(np.linalg.norm(pts - mu, axis=0))
	sigma = np.sqrt(2)/mean_dist

	# consturct the normalization matrix
	T = np.array([[sigma, 0, -sigma*mu[0]],
				  [0, sigma, -sigma*mu[1]],
				  [0 , 0,  1]])

	return(T)
