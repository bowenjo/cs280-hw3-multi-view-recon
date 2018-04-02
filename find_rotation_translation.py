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


	U, s, V = np.linalg.svd(E_mat)
	V = V.T

	R_mats = []
	t_vecs = []

	for sign in [1, -1]:
		t_vec2 = sign*U[:,2]
		t_vecs.append(t_vec2)
		for R_mat_90 in [R_mat_90deg, R_mat_90deg.T]:
			R_mat = sign * U @ R_mat_90 @ V.T

			det_val = np.linalg.det(R_mat)
			
			s = np.array([1, 1, 0])

			'''
			t_x_mat = sign * U @ np.diag(s) @ R_mat_90deg @ U.T
			
			t_vec = np.ones((R_mat.shape[0], 1))
			
			t_vec[0, 0] = t_x_mat[2, 1]
			t_vec[1, 0] = t_x_mat[0, 2]
			t_vec[2, 0] = t_x_mat[1, 0]
			'''
			
			#t_vec3 = np.cross(U[:,0], U[:,1])

			if int(np.round(det_val)) == 1:
				R_mats.append(R_mat)
	
	#print('\nt mat: {} \nt vec: {} \nt vec2: {} \nt vec3: {}'.format(t_x_mat, t_vec, t_vec2, t_vec3))
			
			
	return R_mats, t_vecs
