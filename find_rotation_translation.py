import numpy as np 
import cv2

import matplotlib
import matplotlib.pyplot as plt 

def find_rotation_translation(E_mat):
	"""
	Computes the essential matrix from the fundamental matrix and the calibration matrices.
	Uses the essential matrix to find and return the rotational matrix and the translational vector.
	
	INPUT:
	__________
	- E_mat: 3x3 array, essential matrix

	OUTPUT:
	__________
	- R_mat: 3x3 array, rotational matrix
	- t_vec: 3x1 array, translational vector
	"""
	
	# rotation matrix for 90 deg angle


	R_mat_90deg = np.array([[1, -1, 0],
							[1, 0 , 0], 
							[0, 0, 1]])

	# take svd of essential matrix
	U, s, V = np.linalg.svd(E_mat)
	V = V.T

	# initialize R and t cell arrays
	R_mats = []
	t_vecs = []

	for sign in [1, -1]: # loop over each sign (the sign in ambigous)

		t_vec = sign*U[:,2] # translation vector is third left singular value of the Essential matrix
		t_vecs.append(t_vec)

		for R_mat_90 in [R_mat_90deg, R_mat_90deg.T]: # loop over each roation direction
			R_mat = sign * U @ R_mat_90 @ V.T

			det_val = np.linalg.det(R_mat)
			if int(np.round(det_val)) == 1: # viable rotation matrices have det = 1

				R_mats.append(R_mat)		
			
	return R_mats, t_vecs