import numpy as np 


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
	F = U @ np.diag([S[0], S[1], 0]) @ V

	## Denormalize
	## ------------------------------------------------------------
	F = T_2.T @ F @ T_1

	## Residual Error
	## ------------------------------------------------------------
	res_err = 0
	for i in range(N):
		p1 = pts_1_orig[:,i,None]
		p2 = pts_2_orig[:,i,None]

		d_1_2 = (p1.T @ F @ p2) / np.linalg.norm(F @ p2) # distance between x1 and epipolar line Fx2
		d_2_1 = (p2.T @ F @ p1) / np.linalg.norm(F @ p1) # distance between x2 and epipolar line Fx1	

		# confusion on piazza over correct formulation of residual. Above is what was given in the handout. 
		# d_1_2 = (p2.T@F@p1) / np.linalg.norm(F@p1)
		# d_2_1 = (p1.T@F.T@p2) / np.linalg.norm(F.T@p2)

		res_err += (d_1_2**2 + d_2_1**2) 

	res_err = res_err/(2*N)

	return(F, res_err)


def normalization_matrix(pts):
	# find shift to origin
	mu = np.mean(pts, axis=1).reshape(3,1)

	# scale the mean distance to  sqrt(2)
	mean_dist = np.mean(np.linalg.norm((pts - mu)[:2,:], axis=0))
	sigma = np.sqrt(2)/mean_dist

	# consturct the normalization matrix
	T = np.array([[sigma, 0, -sigma*mu[0]],
				  [0, sigma, -sigma*mu[1]],
				  [0 , 0,  1]])

	return(T)
