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
	delta = params['delta']
	N = params['mesh_lines']
	for i in range(N):
		for j in range(N):
			kx = -delta + 2 * (delta * i / N)
			ky = -delta + 2 * (delta * j / N)

			H = generate_hamiltonian(kx,ky,params['mu_f'],params['mu_c'])
			eig_vals,U_dagger = LA.eig(H)
			U = LA.inv(U_dagger)
			num += moment_number_integral(U,U_dagger,eig_vals,params['mu_f'])
	num = num * (1 /(N ** 2) * (np.pi **2))*(delta** 2)
	# params['mu_f'] = 0
	while(abs(num-1) > 1E-8):
		if(num > 1):
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

		for i in range(N):
			for j in range(N):
				kx = -delta + 2 * (delta * i / N)
				ky = -delta + 2 * (delta * j / N)

				H = generate_hamiltonian(kx,ky,params['mu_f'],params['mu_c'])
				eig_vals,U_dagger = LA.eig(H)
				U = LA.inv(U_dagger)
				num += moment_number_integral(U,U_dagger,eig_vals,params['mu_f'])
		num = num * (1 /(N ** 2) * (np.pi **2))*(delta** 2)
		print(num, params['mu_f'])

	params['mu_f_delta'] = .1

def moment_number_integral(U,U_dagger, eigen_vals, mu):
	return_val = 0
	beta = 1000
	# print(eigen_vals)
	for j in range(2,4):
		for i in range(4):
			return_val += U[i][j] * U_dagger[j][i] * fermi_function(eigen_vals[i],beta,mu)
	return return_val 


def self_consistent(params):
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
	anti_f = []
	Xi_list= []
	out = open('xi_out_2.txt','w')
	out.write("Starting mean field calc \n")
	out.close()
	for j in range(1,10):
		out = open('xi_out_2.txt','a')
		j = 1 * j / 100
		params['antifm_const'] = j
		Xi_guess = params['Xi_guess'] 
		
		counter = 0

		calibtrate_moment(Xi_guess, params)
		
		Xi_act =  get_Xi(Xi_guess, j, params)
		
		while(abs(Xi_guess - Xi_act) > 1e-7):
			Xi_guess = .2*(Xi_act) + .8*(Xi_guess) 	
			Xi_act =  get_Xi(Xi_guess, j, params)
			counter += 1
			if (counter %10 == 0):
				print(j,counter ,"Calculated: ", Xi_act, "Guess:", Xi_guess, " - ", abs(Xi_act- Xi_guess))					
		if(abs(0-Xi_act) > 1e-6):
			print(j, Xi_act)
		
		params['Xi_guess'] = Xi_act
		
		anti_f.append(abs(j))
		Xi_list.append(abs(Xi_act))
		string = 'J = {} , Xi = {}'.format(j, Xi_act)
		print(string)
		out.write(string)
		out.write('\n')
		out.close()
	
	plt.plot(anti_f, Xi_list, label="Phase Diagrams")
	plt.savefig("Phase Diagram_2.png", format="png")



def get_Xi(Xi_guess, anti_f, params):
	Xi_act = 0

	delta = params['delta']
	N = params['mesh_lines']
	for i in range(N):
		for j in range(N):

			kx = -delta + 2 * (delta * i / N)
			ky = -delta + 2 * (delta * j / N)

			H = generate_hamiltonian(kx,ky, params['mu_f'],params['mu_c'])
			
			H[0][2] = -Xi_guess
			H[1][3] = -Xi_guess
			H[2][0] = -np.conj(Xi_guess)
			H[3][1] = -np.conj(Xi_guess)
			
			eig_vals,U_dagger = LA.eig(H)

			U = LA.inv(U_dagger)
			
			thresh = 1e-16
			
			U.real[abs(U.real)<thresh] = 0.0
			U.imag[abs(U.imag) < thresh] = 0.0

			U_dagger.real[abs(U_dagger.real)<thresh] = 0.0
			U_dagger.imag[abs(U_dagger.imag) < thresh] = 0.0
			
			if(np.real(get_Xi_helper(U, U_dagger,eig_vals,params)) > 1):
				print(k, eig_vals[0],eig_vals[1],eig_vals[2],eig_vals[3])
			
			Xi_act +=  np.real(get_Xi_helper(U, U_dagger,eig_vals,params))
	
	return  (Xi_act * (delta **2) * 3 * anti_f)/ (2 * (N ** 2) * (np.pi **2))

def generate_hamiltonian(kx,ky,mu_f, mu_c):
	dims=(4,4)
	hamiltonian = np.zeros(dims, dtype=complex)
	A = ky + 1j*kx
	B = ky - 1j*kx


	hamiltonian[0][1] = B
	hamiltonian[1][0] = A
	hamiltonian[0][0] = mu_c
	hamiltonian[1][1] = mu_c 
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


def main():
	params = {}
	params['beta'] = 1000
	params['mu_f'] = .4
	params['mu_f_prev'] = 0 
	params['mu_f_prev_prev'] = 0
	params['mu_f_delta'] = .1
	params['mu_c'] = .2
	params['Xi_guess'] = 1
	params['delta'] = 5
	params['mesh_lines'] = 500
	self_consistent(params)
main()
