import numpy as np 
from numpy import linalg as LA 
import numpy.matlib
import math 
import matplotlib.pyplot as plt

def generate_hamiltonian(kx,ky,mu_f, mu_c, epsilon):
	dims=(4,4)
	hamiltonian = np.zeros(dims, dtype=complex)
	A = ky + 1j*kx
	B = ky - 1j*kx


	hamiltonian[0][1] = B
	hamiltonian[1][0] = A
	hamiltonian[0][0] = epsilon + mu_c
	hamiltonian[1][1] = epsilon - mu_c 
	return hamiltonian
def mean_field_function(params):
	disp = []
	band_1 = []
	band_2 = []
	band_3 = []
	band_4 = []

	for kx in range(params['kx_start'],params['kx_end']):
		for ky in range(params['ky_start'],params['ky_end']):
			mu_c = 1
			mu_f = 1
			H = generate_hamiltonian(kx/100,ky, mu_f,mu_c, params['epsilon'])
			Xi = 1 
			H[0][2] = Xi
			H[1][3] = Xi

			H[2][0] = np.conj(Xi)
			H[3][1] = np.conj(Xi)

			eig_vals,U = LA.eig(H)
			D = np.diag(eig_vals)
			U_dagger = LA.inv(U)

			disp.append(kx/100)
			band_1.append(eig_vals[0])
			band_2.append(eig_vals[1])
			band_3.append(eig_vals[2])
			band_4.append(eig_vals[3])
			# print(U @ D @ U_dagger)
			# print(H)
	plt.plot(disp, band_1,disp, band_2,disp,band_3,disp,band_4)
	plt.show()


def main():
	params = {}
	params['kx_start'] = -1000
	params['ky_start'] = 0
	params['kz_start'] = 0
	params['kx_end'] = 1000
	params['ky_end'] = 1
	params['kz_end'] = 1
	params['antifm_const'] = 1
	params['epsilon'] = 1
	
	mean_field_function(params)
main()