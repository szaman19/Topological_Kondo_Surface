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

def calibtrate_moment(Xi, params):
	# print("Return the correct value for mu_f")
	num = 0
	k_range = params['cutoff']
	norm = params['cutoff_norm']
	for kx in range(-1*k_range,k_range):
		for ky in range(-1*k_range,k_range):
			kx /= norm
			ky /= norm
			H = generate_hamiltonian(kx,ky,params['mu_f'],params['mu_c'])
			eig_vals,U_dagger = LA.eig(H)
			U = LA.inv(U_dagger)
			num += moment_number_integral(U,U_dagger,eig_vals,params['mu_f'])
	num = num * (1 / (norm * norm))*(k_range **2)
	# params['mu_f'] = 0
	while(abs(num-36) > 1E-8):
		if(num > 36):
			params['mu_f_prev_prev'] = params['mu_f_prev']
			params['mu_f_prev'] = params['mu_f']
			if(params['mu_f_prev_prev'] == params['mu_f'] - params['mu_f_delta']):
				params['mu_f_delta'] /= 2
			params['mu_f'] -= params['mu_f_delta']
		else:
			params['mu_f_prev_prev'] = params['mu_f_prev']
			params['mu_f_prev'] = params['mu_f']
			if(params['mu_f_prev_prev'] == params['mu_f'] + params['mu_f_delta']):
				params['mu_f_delta'] /= 2
			params['mu_f'] +=params['mu_f_delta']
		num = 0

		for kx in range(-1*k_range,k_range):
			for ky in range(-1*k_range,k_range):
				kx /= norm
				ky /= norm
				H = generate_hamiltonian(kx,ky,params['mu_f'],params['mu_c'])
				eig_vals,U_dagger = LA.eig(H)
				U = LA.inv(U_dagger)
				num += moment_number_integral(U,U_dagger,eig_vals,params['mu_f'])
		num = num * (1 / (norm * norm))*(k_range ** 2)
		print(num, params['mu_f'])
	# if(num)
	# print("Mu Moment", params['mu_f'])
	params['mu_f_delta'] = .2

def moment_number_integral(U,U_dagger, eigen_vals, mu):
	return_val = 0
	beta = 1000
	# print(eigen_vals)
	for j in range(2,4):
		for i in range(4):
			return_val += U[i][j] * U_dagger[j][i] * fermi_function(eigen_vals[i],beta,mu)
	return return_val  




def self_consistent(j,parameters):
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
	params = {}
	params = parameters
	anti_f = []
	Xi_list= []
	
	params['antifm_const'] = j
	Xi_guess = params['Xi_guess'] 
	counter = 0
	Xi_act =  get_Xi(Xi_guess, params)
	while(abs(Xi_guess - Xi_act) > 1e-7):
		Xi_guess = .01*(Xi_act) + .99*(Xi_guess) 		
			
		calibtrate_moment(Xi_guess, params)

		Xi_act =  get_Xi(Xi_guess,params)

		counter += 1
		if (counter % 100 ==0):
			print(j, counter , Xi_act, Xi_guess)					
	if(abs(0-Xi_act) > 1e-6):
		print(j, Xi_act)
	return (j,Xi_act * (j * 3/2))
	# anti_f.append(j)
	# Xi_list.append(Xi_act)
	# plt.plot(anti_f, Xi_list, label="Phase Diagrams")
	# plt.savefig("Phase Diagram ", format="png")





def get_Xi(Xi_guess, params):
	Xi_act = 0
	k_range = params['cutoff']
	norm = params['cutoff_norm']
	for kx in range(-1*k_range,k_range):
		for ky in range(-1*k_range,k_range):
			kx /= norm
			ky /= norm
			H = generate_hamiltonian(kx,ky, params['mu_f'],params['mu_c'])
			H[0][2] = (3 * params['antifm_const'] / 2) *Xi_guess
			H[1][3] = (3 * params['antifm_const'] / 2) *Xi_guess
			H[2][0] = (3 * params['antifm_const'] / 2) *np.conj(Xi_guess)
			H[3][1] = (3 * params['antifm_const'] / 2) *np.conj(Xi_guess)
			eig_vals,U_dagger = LA.eig(H)
			U = LA.inv(U_dagger)
			thresh = 1e-16
			U.real[abs(U.real)<thresh] = 0.0
			U.imag[abs(U.imag) < thresh] = 0.0
			# D = np.diag(eig_vals)
			U_dagger.real[abs(U_dagger.real)<thresh] = 0.0
			U_dagger.imag[abs(U_dagger.imag) < thresh] = 0.0
			Xi_act +=  np.real(get_Xi_helper(U, U_dagger,eig_vals,params))
	return  Xi_act / (norm ** 4) * (k_range **2)
def generate_hamiltonian(kx,ky,mu_f, mu_c):
	dims=(4,4)
	hamiltonian = np.zeros(dims, dtype=complex)
	A = ky + 1j*kx
	B = ky - 1j*kx


	hamiltonian[0][1] = B
	hamiltonian[1][0] = A
	hamiltonian[0][0] = -mu_c
	hamiltonian[1][1] = -mu_c 
	hamiltonian[2][2] = -mu_f
	hamiltonian[3][3] = -mu_f
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
def trial(j, params):
	params['Xi_guess'] = j 
	return (j,params['Xi_guess'])
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
	params['mu_f'] = .4
	params['mu_f_prev'] = 0 
	params['mu_f_prev_prev'] = 0
	params['mu_f_delta'] = .2
	params['mu_c'] = .2
	params['Xi_guess'] = -1
	params['cutoff'] = 20
	params['cutoff_norm'] = 20

	#self_consistent(j, params)

	outputs = []
	for j in range(20):
		pool = Pool(processes=10)
		results = [pool.apply_async(self_consistent, args=((j/10)+x*.1,params)) for x in range(10)]
		output = [p.get() for p in results]
		print(output)
		outputs.append(output)
	log = open("output.txt", 'w')
	for row in outputs:
		for each in row:
			# string = 
			log.write(str(each))
			log.write(",")
		log.write("\n")

	log.close()



main()