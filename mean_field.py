import numpy as np 
from numpy import linalg as LA 
import numpy.matlib
import math 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from multiprocessing import Pool

np.seterr(all='raise')

def fermi_function(energy,  beta, mu=0):
	energy = np.real(energy)
	try:
		if ((beta * (energy - mu)) < -110):
			return 1
		elif((beta * (energy - mu)) > 120):
			return 0
		else:
			return 1 / (1 + np.exp(beta * (energy - mu)))
	except:
		print("Exception has occured", energy)

def moment_number_integral(hamiltonian, params):
	print("Return the correct value for mu_f")

def self_consistent(kx, ky, params):
	print("working on it")
	for j in range(10):
		j = -1 * j

		''' 
		For some J, we will find xi
		for a range of k
		start loop:
			guess a xi 
			calculate the order parameter and compare with xi
			calculate the number of local moment and tune mu_f
			calculate the order parameter and compare with xi
			update xi_guess
		
		'''

def generate_hamiltonian(kx,ky,mu_f, mu_c):
	dims=(4,4)
	hamiltonian = np.zeros(dims, dtype=complex)
	A = ky + 1j*kx
	B = ky - 1j*kx


	hamiltonian[0][1] = B
	hamiltonian[1][0] = A
	hamiltonian[0][0] = mu_c
	hamiltonian[1][1] = mu_c 
	hamiltonian[2][2] = mu_f
	hamiltonian[3][3] = mu_f
	return hamiltonian
def integral_helper(U,U_dagger,eigen_vals,params):
	U_11 = U[0][0]
	C_13 = U_dagger[0][2]
	U_12 = U[0][1]
	C_23 = U_dagger[1][2]
	U_13 = U[0][2]
	C_33 = U_dagger[2][2]
	U_14 = U[0][3]
	C_43 = U_dagger[3][2]
	beta = params['beta']
	nf_0 = fermi_function(eigen_vals[0],beta)
	nf_1 = fermi_function(eigen_vals[1],beta)
	nf_2 = fermi_function(eigen_vals[2],beta)
	nf_3 = fermi_function(eigen_vals[3],beta)
	return (U_11*C_13*nf_0 + U_12*C_23*nf_1 + U_13*C_33*nf_2 + U_14*C_43*nf_3)
def get_Xi(Xi_guess, params):
	Xi_act = 0
	for kx in range(-30,30):
		for ky in range(-30,30):
			params['mu_c'] = 0
			params['mu_f'] = 0
			# params['beta'] = 100
			H = generate_hamiltonian(kx/20,ky/20, 0,0)
			H[0][2] = Xi_guess
			H[1][3] = Xi_guess
			H[2][0] = np.conj(Xi_guess)
			H[3][1] = np.conj(Xi_guess)
			eig_vals,U = LA.eig(H)
			D = np.diag(eig_vals)
			U_dagger = LA.inv(U)
			Xi_act +=  np.real(get_Xi_helper(U, U_dagger,eig_vals,params))
	return (3 * params['antifm_const'] / 2) * Xi_act



def get_Xi_helper(U,U_dagger,eigen_vals,params):
	U_13 = U[0][2]
	U_23 = U[1][2]
	U_33 = U[2][2]
	U_43 = U[3][2]

	C_11 = U_dagger[0][0]
	C_12 = U_dagger[0][1]
	C_13 = U_dagger[0][2]
	C_14 = U_dagger[0][3]

	beta = params['beta']
	nf_0 = fermi_function(eigen_vals[0],beta)
	nf_1 = fermi_function(eigen_vals[1],beta)
	nf_2 = fermi_function(eigen_vals[2],beta)
	nf_3 = fermi_function(eigen_vals[3],beta)
     
	return (U_13*C_11*nf_0 + C_12*U_23*nf_1 + C_13*U_33*nf_2 + C_14*U_43*nf_3)
def mean_field_function(params):
	disp = []
	Xi = []
	band_1 = []
	band_2 = []
	band_3 = []
	band_4 = []

	#params['antifm_const'] = j
	for kx in range(params['kx_start'],params['kx_end']):
		for ky in range(params['ky_start'],params['ky_end']):
			mu_c = 0
			mu_f = 0
			Xi_guess = -1 
			# H[0][2] = Xi_guess
			# H[1][3] = Xi_guess
                               
			# H[2][0] = np.conj(Xi_guess)
			# H[3][1] = np.conj(Xi_guess)
			# Xi_act = 0
			# eig_vals = []
			counter = 0

			# eig_vals,U = LA.eig(H)

			# D = np.diag(eig_vals)
			# U_dagger = LA.inv(U)
			Xi_act =  get_Xi(Xi_guess, params)
			while(abs(Xi_guess - Xi_act) > 1e-7):
				Xi_guess = .2*(Xi_act-Xi_guess) + .9*(Xi_guess) 		
				
				Xi_act =  get_Xi(Xi_guess,params)
				counter += 1
				if (counter % 10 ==0):
					print(counter , Xi_act, Xi_guess)
						
			if(abs(0-Xi_act) > 1e-6):
				print(counter, Xi_act)
			H = generate_hamiltonian(kx/100,ky/100, 0,0)
			H[0][2] = Xi_act  
			H[1][3] = Xi_act               
			H[2][0] = np.conj(Xi_act)  
			H[3][1] = np.conj(Xi_act) 

			eig_vals = LA.eigvalsh(H)
			Xi.append(Xi_act)
			
			print("kx: ",kx, "ky: ",ky, "Xi: ",Xi_act)
			band_1.append(eig_vals[0])
			band_2.append(eig_vals[1])
			band_3.append(eig_vals[2])
			band_4.append(eig_vals[3])
			disp.append(kx/100)


	# X, Y = np.meshgrid(disp, disp)
	# Z = np.reshape(band_1, X.shape)
	# Z1 = np.reshape(band_3, X.shape)
	# Z2 = np.reshape(band_2, X.shape)
	# Z3 = np.reshape(band_4, X.shape)
	# print(X.shape)	
	# fig = plt.figure()
	# ax = plt.axes(projection='3d')
	# ax.contour3D(X, Y, Z, 50, cmap='binary')
	# ax.contour3D(X, Y, Z1, 50, cmap='binary')
	# ax.contour3D(X, Y, Z2, 50, cmap='binary')
	# ax.contour3D(X, Y, Z3, 50, cmap='binary')
	# ax.set_xlabel('x')
	# ax.set_ylabel('y')
	# ax.set_zlabel('z')
	plt.plot(disp, Xi, label="Order parameter")
	# plt.plot(disp, band_2, label="band 2")
	# plt.plot(disp,band_3, label="band 3")
	# plt.plot(disp,band_4, label="band 4")
	plt.legend()
	# plt.show()
	plt.savefig("trial_1.png", format="png")


def main():
	params = {}
	params['kx_start'] = -100
	params['ky_start'] = 0
	params['kz_start'] = 0
	params['kx_end'] = 100
	params['ky_end'] = 1
	params['kz_end'] = 1
	params['antifm_const'] = -1
	params['epsilon'] = .01
	params['beta'] = 1000
	
	
	mean_field_function(params)
main()
