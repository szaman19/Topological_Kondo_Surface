import numpy as np 
from numpy import linalg as LA 
import numpy.matlib
import math 
import matplotlib.pyplot as plt
from multiprocessing import Pool


def fermi_function(energy,  beta, mu=0):
	return 1 / (1 + np.exp(beta * (energy - mu)))
#def integral()
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
def get_Xi(U,U_dagger,eigen_vals,params):
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
     
	return (3 * params['antifm_const'] / 2) *(U_11*C_13*fermi_function(eigen_vals[0]) + U_12*C_23*fermi_function(eigen_vals[1]) + U_13*C_33*fermi_function(eigen_vals[2]) + U_14*C_43*fermi_function(eigen_vals[3]))
def mean_field_function(params):
	disp = []
	# disp_y = []
	band_1 = []
	band_2 = []
	band_3 = []
	band_4 = []

	for j in range(-10,10):
		params['antifm_const'] = j
		for kx in range(params['kx_start'],params['kx_end']):
			for ky in range(params['ky_start'],params['ky_end']):
				mu_c = 1
				mu_f = 1
				#epsilon = 2 * params['epsilon']*(math.cos(kx)+math.cos(ky))
				
				params['mu_c'] = mu_c
				params['mu_f'] = mu_f
				params['beta'] = 100
				H = generate_hamiltonian(kx/100,ky/100, mu_f,mu_c)
				Xi_guess = -1 
				H[0][2] = Xi_guess
				H[1][3] = Xi_guess
                                
				H[2][0] = np.conj(Xi_guess)
				H[3][1] = np.conj(Xi_guess)
				Xi_act = 0
				eig_vals = []
				counter = 0
				while(abs(Xi_guess - Xi_act) > 1e-8):
					if(Xi_guess < Xi_act):
						if (Xi_act >0 and Xi_guess < 0):
							Xi_guess = -2 * Xi_guess
						else:
							Xi_guess = 2 * Xi_guess
					else:
						Xi_guess -= abs(Xi_act - Xi_guess)/2
							
					H[0][2] = Xi_guess
					H[1][3] = Xi_guess
					H[2][0] = np.conj(Xi_guess)
					H[3][1] = np.conj(Xi_guess)
					eig_vals,U = LA.eig(H)
					D = np.diag(eig_vals)
					U_dagger = LA.inv(U)
					Xi_act =  np.real(get_Xi(U, U_dagger,eig_vals,params))
					counter += 1
				if(abs(0-Xi_act) > 1e-6):
					print(Xi_act)
			# disp.append(kx/100)
			# disp_y.append(ky/100)
			# band_1.append(eig_vals[0])
			# band_2.append(eig_vals[1])
			# band_3.append(eig_vals[2])
			# band_4.append(eig_vals[3])


	# plt.plot(disp, band_1, label="band 1")
	# plt.plot(disp, band_2, label="band 2")
	# plt.plot(disp,band_3, label="band 3")
	# plt.plot(disp,band_4, label="band 4")
	# plt.legend()
	# plt.show()


def main():
	params = {}
	params['kx_start'] = -1000
	params['ky_start'] = -1000
	params['kz_start'] = 0
	params['kx_end'] = 1000
	params['ky_end'] = 1000
	params['kz_end'] = 1
	params['antifm_const'] = -1
	params['epsilon'] = .01
	
	
	mean_field_function(params)
main()
