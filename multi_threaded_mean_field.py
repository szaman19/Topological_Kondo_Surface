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

def mu_f_update(num, params):
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
	
	return params 


def calibtrate_moment(Xi, params):
	# print("Return the correct value for mu_f")
	num = 0

	L = params['num_sites']
	N = L ** 2
	PI = params['lim']
	MU_F = params['mu_f']
	MU_C = params['mu_c']
	M_U_D = {}

	M_U_D['mu_f_prev_prev'] = 0
	M_U_D['mu_f_prev'] = 0
	M_U_D['mu_f_delta'] = .05
	M_U_D['mu_f'] = MU_F

	for i in range(L):
		for j in range(L):
			kx = -PI  + 2 * (PI / L * i)
			ky =  -PI  + 2 * (PI / L * j)
			H = generate_hamiltonian(kx,ky,MU_F,MU_C)
			eig_vals,U_dagger = LA.eig(H)
			U = LA.inv(U_dagger)
			num += moment_number_integral(U,U_dagger,eig_vals,MU_F)

	num = num / N 


	# params['mu_f'] = 0

	
	while(abs(num-1) > 1E-8):
		M_U_D = mu_f_update(num, M_U_D)	
		MU_F = M_U_D['mu_f']
		num = 0
		for i in range(L):
			for j in range(L):
				kx = -PI  + 2 * (PI / L * i)
				ky =  -PI  + 2 * (PI / L * j)

				H = generate_hamiltonian(kx,ky,MU_F,MU_C)
				eig_vals,U_dagger = LA.eig(H)
				U = LA.inv(U_dagger)
				num += moment_number_integral(U,U_dagger,eig_vals,MU_C)
		num = num / N
		print("J={},val={:.9f},mu_f={:.9f}".format(params['antifm_const'], num, MU_F))
		

	check_val_down = 0
	check_val_up = 0 
	for i in range(L):
		for j in range(L):
			kx = -PI  + 2 * (PI / L * i)
			ky =  -PI  + 2 * (PI / L * j)
			
			H = generate_hamiltonian(kx,ky,MU_F,MU_C)
			eig_vals,U_dagger = LA.eig(H)
			U = LA.inv(U_dagger)
			
			check_val_up += up_moment(U,U_dagger,eig_vals,MU_F)
			check_val_down += down_moment(U,U_dagger,eig_vals,MU_F)

	check_val_up = check_val_up / N
	check_val_down = check_val_down / N 

	if (abs(check_val_down - check_val_up) > 1e-8):
		print("WARNING: INVALID MOMENT CALCULATED!",check_val_up, check_val_down)
	
	params['mu_f'] = MU_F
	return params['mu_f']


def up_moment(U,U_dagger, eigen_vals, mu):
	check_val_up = 0
	# check_val_down = 0
	beta = 1000
	j = 2
	for i in range(4):
		check_val_up += U[i][j] * U_dagger[j][i] * fermi_function(eigen_vals[i],beta,mu)
	return check_val_up
def down_moment(U,U_dagger, eigen_vals, mu):
	# check_val_up = 0
	check_val_down = 0
	beta = 1000
	j = 3
	for i in range(4):
		check_val_down += U[i][j] * U_dagger[j][i] * fermi_function(eigen_vals[i],beta,mu)
	return check_val_down
	# print(np.real(check_val_up), np.real(check_val_down))


def moment_number_integral(U,U_dagger, eigen_vals, mu):
	return_val = 0
	beta = 1000
	# print(eigen_vals)
	for j in range(2,4):
		for i in range(4):
			return_val += U[i][j] * U_dagger[j][i] * fermi_function(eigen_vals[i],beta,mu)
	return np.real(return_val)  


def self_consistent(j, mu_c = 0):
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
	params['beta'] = 1000
	params['mu_f'] = .4
	params['Xi_guess'] = 1
	params['lim'] = np.pi
	params['num_sites'] = 20
	params['mu_c'] = mu_c/8 
	
	params['antifm_const'] = j
	

	Xi_guess = params['Xi_guess']
	Xi_act = 0

	### Constants
	L = params['num_sites']
	N = L ** 2
	PI = params['lim']


	MU_F = calibtrate_moment(Xi_guess, params)
	# for i in range(L):
	# 	for j in range(L):
	# 		kx = -PI  + 2 * (PI / L * i)
	# 		ky =  -PI  + 2 * (PI / L * j)

	# 		print(kx,ky)
	
	params['mu_f'] = calibtrate_moment(Xi_guess, params)

	print(params['mu_f'])
	counter = 0
	Xi_act =  get_Xi(Xi_guess, params)
	while(abs(Xi_guess - Xi_act) > 5e-9):
		Xi_guess = .2*(Xi_act) + .8*(Xi_guess) 		
		# params['mu_f'] = calibtrate_moment(Xi_guess, params)
		Xi_act =  get_Xi(Xi_guess, params)
		counter += 1
		if (counter % 1000 ==0):
			print("J= {},{:.3f},act = {:.8f}, guess = {:.8f}".format(j, counter , Xi_act, Xi_guess))					
	
	print("J= {},{:3f},act = {:.8f}, guess = {:.8f}".format(j, counter , Xi_act, Xi_guess))
	if(abs(0-Xi_act) > 1e-6):
		print(j, Xi_act)
	return (j,Xi_act, params['mu_f'])

def get_Xi(Xi_guess, params):
	Xi_act = 0

	L = params['num_sites']
	N = L ** 2
	PI = params['lim']

	anti_f = params['antifm_const']
	for i in range(L):
		for j in range(L):
			kx = -PI  + 2 * (PI / L * i)
			ky =  -PI  + 2 * (PI / L * j)

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
			# D = np.diag(eig_vals)
			U_dagger.real[abs(U_dagger.real)<thresh] = 0.0
			U_dagger.imag[abs(U_dagger.imag) < thresh] = 0.0
			
			Xi_act +=  np.real(get_Xi_helper(U, U_dagger,eig_vals,params))
	
	return  (Xi_act * 3 * anti_f)/ (2 * N )
def generate_hamiltonian(kx,ky,mu_f, mu_c):
	dims=(4,4)
	hamiltonian = np.zeros(dims, dtype=complex)
	
	hamiltonian = np.zeros(dims, dtype=complex)
	A = np.sin(ky)-1j*np.sin(kx)
	A_star = np.sin(ky)+1j*np.sin(kx)

	epsilon_k = 0.3 * (np.sin(kx/2) ** 2 + np.sin(ky/2) ** 2) 
	hamiltonian[0][0] = epsilon_k - mu_c 
	hamiltonian[0][1] = A 
	hamiltonian[1][0] = A_star
	hamiltonian[1][1] = -epsilon_k - mu_c
	hamiltonian[2][2] = -mu_f
	hamiltonian[3][3] = -mu_f
	return hamiltonian
def integral_helper(U,U_dagger,eigen_vals,params):
	U_11 = U[0][2]
	C_13 = U_dagger[0][0]
	U_12 = U[1][2]
	C_23 = U_dagger[0][1]
	U_13 = U[2][2]
	C_33 = U_dagger[0][2]
	U_14 = U[3][2]
	C_43 = U_dagger[0][3]
	beta = params['beta']
	nf_0 = fermi_function(eigen_vals[0],beta)
	nf_1 = fermi_function(eigen_vals[1],beta)
	nf_2 = fermi_function(eigen_vals[2],beta)
	nf_3 = fermi_function(eigen_vals[3],beta)
	num = (U_11*C_13*nf_0 + U_12*C_23*nf_1 + U_13*C_33*nf_2 + U_14*C_43*nf_3)

	if(abs(num) > 1):
		print("WARNING: INVALID SUM", num)
	return num


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
	

	#self_consistent(j, params)
	NUM_PROCESS = 8
	for i in range(2,4):
		outputs = []
		file_name = "phase_diagrams_chiral_kondo_"+str(i) + ".csv"
		for j in range(8):
			pool = Pool(processes=NUM_PROCESS)
			results = [pool.apply_async(self_consistent, args=((j*0.16)+x*.02,i)) for x in range(NUM_PROCESS)]
			output = [p.get() for p in results]
			print(output)
			outputs.append(output)
		log = open(file_name, 'w')
		log.write("J, Xi, Mu_f \n")
		for row in outputs:
			for tup in row:
				for each in tup:
					log.write(str(each))
					log.write(",")
				log.write("\n")
		log.close()
main()
