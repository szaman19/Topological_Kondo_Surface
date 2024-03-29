import numpy as np 
from numpy import linalg as LA 
import numpy.matlib
import math 
from multiprocessing import Pool
import matplotlib.pyplot as plt

np.seterr(all='raise')

def gen_brillouin_zone(L = 10):
	# X_points = []
	# Y_points = []
	Points = []
	for i in range(L):
		for j in range(L):
			kx = -np.pi + 2 * (np.pi / L * i)
			ky = -np.pi + 2 * (np.pi / L * j)
			Points.append((kx,ky))
	return Points




def util_equal(a , b, threshold=1E-3):
	return not(abs(a - b) > threshold)

def order_param_equal(calculated_order_params, guess_order_params ):
	for params in calculated_order_params.keys():
		v1 = calculated_order_params[params]
		v2 = guess_order_params[params]
		if (not util_equal(v1, v2)):
			return False
	return True

def get_col(mat_U,column_num):
	return mat_U[:,column_num]

def get_row(mat_U, row_num):
	return mat_U[row_num]


def fermi_function(energy,  beta=10000, mu=0):
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


def generate_U(op, params, K_POINTS):
	mu_f = params['mu_f']
	mu_c = params['mu_c']
	eigen_vals = []
	U_dagger_list = []
	for i in range(len(K_POINTS)):
		kx = K_POINTS[i][0]
		ky = K_POINTS[i][1]

		# print(kx,ky)
		ham = gen_hamiltonian(kx, ky, mu_f,mu_c, True)
		ham  = hamiltonian_order_params(ham, op)
		eigs, U_dagger = LA.eigh(ham)
		U_dagger_list.append(U_dagger)
		eigen_vals.append(eigs)
	return eigen_vals, U_dagger_list


def update_mu_f(num, params):
	if (util_equal(num, 1)):
		return params
	if (params['mu_f_delta'] <1E-16):
		params['mu_f_delta'] = 0.0005
	if(num > 1):
		params['mu_f_prev_prev'] = params['mu_f_prev']
		params['mu_f_prev'] = params['mu_f']
		if(params['mu_f_prev_prev'] == params['mu_f'] - params['mu_f_delta']):
			params['mu_f_delta'] /= 4
		params['mu_f'] -= params['mu_f_delta']
	else:
		params['mu_f_prev_prev'] = params['mu_f_prev']
		params['mu_f_prev'] = params['mu_f']
		if(params['mu_f_prev_prev'] == params['mu_f'] + params['mu_f_delta']):
			params['mu_f_delta'] /= 4
		params['mu_f'] +=params['mu_f_delta']
	
	return params

def update_mu_c(num, params):
	if (util_equal(num, 2)):
		return params
	if (params['mu_c_delta'] <1E-16):
		params['mu_c_delta'] = 0.05	
	if(num > 2):
		params['mu_c_prev_prev'] = params['mu_c_prev']
		params['mu_c_prev'] = params['mu_c']
		if(params['mu_c_prev_prev'] == params['mu_c'] - params['mu_c_delta']):
			params['mu_c_delta'] /= 2
		params['mu_c'] -= params['mu_c_delta']
	else:
		params['mu_c_prev_prev'] = params['mu_c_prev']
		params['mu_c_prev'] = params['mu_c']
		if(params['mu_c_prev_prev'] == params['mu_c'] + params['mu_c_delta']):
			params['mu_c_delta'] /= 2
		params['mu_c'] +=params['mu_c_delta']
	
	return params

def calibrate_mu(op, params, K_POINTS):
	eigen_vals, U_dagger_list = generate_U(op,params, K_POINTS)

	conduction_number = 0
	moment_number = 0

	mu_f = params['mu_f']
	mu_c = params['mu_c']

	N = len(K_POINTS) 

	mu_c_data = {}
	mu_f_data = {}

	mu_f_data['mu_f_prev_prev'] = 0
	mu_f_data['mu_f_prev'] = 0
	mu_f_data['mu_f_delta'] = .05
	mu_f_data['mu_f'] = 0

	mu_c_data['mu_c_prev_prev'] = 0
	mu_c_data['mu_c_prev'] = 0
	mu_c_data['mu_c_delta'] = .05
	mu_c_data['mu_c'] = mu_c


	for i in range(len(eigen_vals)):
		# print(i)
		U_dagger = U_dagger_list[i]
		eig_val = eigen_vals[i]
		# print(U_dagger)
		# print(eig_val)
		U = np.transpose(np.conjugate(U_dagger))
		conduction_number += calc_conduction_number(U, U_dagger, eig_val, mu_c)
		moment_number += calc_moment_number(U, U_dagger, eig_val, mu_f)

	# print(type(conduction_number))
	conduction_number /= (2*N)
	moment_number /= (2*N) 

	# print("N_c: {:.9f}, N_f: {:.9f}", conduction_number, moment_number)

	is_equal_NC = util_equal(conduction_number,2)
	is_equal_NF = util_equal(moment_number,1)

	loop_condition = not (is_equal_NF and is_equal_NC)

	counter = 0
	while (loop_condition):
		mu_c_data = update_mu_c(conduction_number, mu_c_data)
		mu_f_data = update_mu_f(moment_number, mu_f_data)

		mu_c = mu_c_data['mu_c']
		mu_f = mu_f_data['mu_f']

		conduction_number = 0
		moment_number = 0

		params['mu_c'] = mu_c
		params['mu_f'] = mu_f

		# eigen_vals, U_dagger_list = generate_U(op,params, K_POINTS)


		# for i in range(L):
		# 	for j in range(L):
		# 		kx = -np.pi + 2 * (np.pi / L * i)
		# 		kx = -np.pi + 2 * (np.pi / L * i)
		eigen_vals, U_dagger_list = generate_U(op,params, K_POINTS)


		for i in range(len(eigen_vals)):
			U_dagger = U_dagger_list[i]
			eig_val = eigen_vals[i]
			U = np.transpose(np.conjugate(U_dagger))
			conduction_number += calc_conduction_number(U, U_dagger, eig_val, mu_c)
			moment_number += calc_moment_number(U, U_dagger, eig_val, mu_f)

		conduction_number /= (2 * N)
		moment_number /= (2 * N)

		# print(mu_c, mu_f)

		is_equal_NC = util_equal(conduction_number,2)
		is_equal_NF = util_equal(moment_number,1)

		loop_condition = not (is_equal_NF and is_equal_NC)
		# if(counter % 400 == 0 and counter > 0):
		# 	print("j: {:2f} N_c: {:9f}, N_f: {:9f}".format( params['j'],conduction_number, moment_number))
		# 	print("mu_c: ", mu_c)
		# 	print("mu_f: ", mu_f)
		# 	print("dalta f", mu_f_data['mu_f_delta'])
		# 	print("delta c", mu_c_data['mu_c_delta'])
			
		counter +=1
	# print(util_equal(conduction_number,1))

	# print("Conduction Number: ", conduction_number)
	# print("Moment Number: ", moment_number)

	params['mu_c'] = mu_c
	params['mu_f'] = mu_f
	


	return params

def calc_conduction_number(U, U_dagger, eigen_vals, mu):
	
	c_k_up = get_row(U, 0)
	c_k_down = get_row(U, 1)
	c_q_up = get_row(U, 4)
	c_q_down = get_row(U, 5)

	c_k_dagger_up = get_col(U_dagger, 0)
	c_k_dagger_down = get_col(U_dagger, 1)
	c_q_dagger_up = get_col(U_dagger, 4)
	c_q_dagger_down = get_col(U_dagger, 5)

	f_k_up = get_row(U, 2)
	f_k_down = get_row(U, 3)
	f_q_up = get_row(U,6)
	f_q_down =  get_row(U,7)

	f_k_dagger_up = get_col(U_dagger, 2)
	f_k_dagger_down = get_col(U_dagger, 3)	
	f_q_dagger_up = get_col(U_dagger, 6)
	f_q_dagger_down = get_col(U_dagger, 7)
 
	number = 0
	for i in range(len(eigen_vals)):
		energy = eigen_vals[i]	
		number += (np.absolute(c_k_up[i])**2   + np.absolute(c_q_up[i])**2
			+np.absolute(c_k_down[i])**2  + np.absolute(c_q_down[i])**2)* (fermi_function(energy, mu=mu))

	for i in range(len(eigen_vals)):
		energy = eigen_vals[i]
		number += (np.absolute(f_k_up[i])**2+ np.absolute(f_q_up[i])**2
			+np.absolute(f_k_down[i])**2+ np.absolute(f_q_down[i])**2) * (fermi_function(energy, mu=mu))
	
	# print("total number: ", number)
	return number	

def calc_moment_number(U, U_dagger, eigen_vals, mu):
	f_k_up = get_row(U, 2)
	f_k_down = get_row(U, 3)
	f_q_up = get_row(U,6)
	f_q_down =  get_row(U,7)

	f_k_dagger_up = get_col(U_dagger, 2)
	f_k_dagger_down = get_col(U_dagger, 3)	
	f_q_dagger_up = get_col(U_dagger, 6)
	f_q_dagger_down = get_col(U_dagger, 7)

	# print(len(eigen_vals) == 8)
	number = 0
	for i in range(len(eigen_vals)):
		energy = eigen_vals[i]
		number += (f_k_up[i] * f_k_dagger_up[i]+ f_q_up[i] * f_q_dagger_up[i]
			+f_k_down[i] * f_k_dagger_down[i]+ f_q_down[i] * f_q_dagger_down[i]) * (fermi_function(energy, mu=mu)) 
	
	# print("Moment number:", number)
	return number	

	# print("To be implemented")


def calc_xi_one(U_dagger, U, Eigs, J, spin):
	''' MUST divide by N before returning'''

	up_sum = 0
	down_sum = 0
	
	f_k_dagger_up = np.conjugate(get_row(U_dagger, 2))
	f_k_dagger_down = np.conjugate(get_row(U_dagger, 3))	
	f_q_dagger_up = np.conjugate(get_row(U_dagger, 6))
	f_q_dagger_down = np.conjugate(get_row(U_dagger, 7))


	c_k_up = get_row(U_dagger, 0)
	c_k_down = get_row(U_dagger, 1)
	c_q_up = get_row(U_dagger, 4)
	c_q_down = get_row(U_dagger, 5)

	# print(Eigs)

	for i in range(len(Eigs)):
		energy = Eigs[i]
		up_sum += f_k_dagger_up[i] * c_k_up[i] * fermi_function(energy)
		up_sum += f_q_dagger_up[i] * c_q_up[i] * fermi_function(energy)
		down_sum += f_k_dagger_down[i] * c_k_down[i] * fermi_function(energy)
		down_sum += f_q_dagger_down[i] * c_q_down[i] * fermi_function(energy)


	if(spin == 0):
		''' down '''
		up_sum = up_sum * (J / 2)
		down_sum = down_sum * (J / 4)
	
	else:
		''' up '''
		up_sum = up_sum * (J / 4)
		down_sum = down_sum * (J / 2)
	# print(up_sum + down_sum)
	return (up_sum + down_sum) 
	# print("To be implemented")

def calc_xi_two(U_dagger, U, Eigs, J, spin):

	up_sum = 0
	down_sum = 0
	
	f_k_dagger_up = np.conjugate(get_row(U_dagger, 2))
	f_k_dagger_down = np.conjugate(get_row(U_dagger, 3))	
	f_q_dagger_up = np.conjugate(get_row(U_dagger, 6))
	f_q_dagger_down = np.conjugate(get_row(U_dagger, 7))

	
	c_k_up = get_row(U_dagger, 0)
	c_k_down = get_row(U_dagger, 1)
	c_q_up = get_row(U_dagger, 4)
	c_q_down = get_row(U_dagger, 5)

	for i in range(len(Eigs)):
		energy = Eigs[i]
		up_sum += f_k_dagger_up[i] * c_q_up[i] * fermi_function(energy)
		up_sum += f_q_dagger_up[i] * c_k_up[i] * fermi_function(energy)

		down_sum += f_k_dagger_down[i] * c_q_down[i] * fermi_function(energy)
		down_sum += f_q_dagger_down[i] * c_k_down[i] * fermi_function(energy)
	if(spin == 0):
		''' down '''
		up_sum = up_sum * (J / 2)
		down_sum = down_sum * (J / 4)
	
	else:
		''' up '''
		up_sum = up_sum * (J / 4)
		down_sum = down_sum * (J / 2)

	return up_sum + down_sum
	# print("To be implemented")

def calc_M1_C(U_dagger, U, Eigs, J):

	up_sum = 0
	down_sum = 0

	c_k_up = get_row(U_dagger, 0)
	c_k_down = get_row(U_dagger, 1)
	c_q_up = get_row(U_dagger, 4)
	c_q_down = get_row(U_dagger, 5)

	c_k_dagger_up = get_col(U_dagger, 0)
	c_k_dagger_down = get_col(U_dagger, 1)
	c_q_dagger_up = get_col(U_dagger, 4)
	c_q_dagger_down = get_col(U_dagger, 5)


	for i in range(len(Eigs)):
		energy = Eigs[i]
		up_sum += c_k_up[i] * c_k_dagger_up [i]* fermi_function(energy)
		up_sum += c_q_up[i] * c_q_dagger_up[i] * fermi_function(energy)
		down_sum += c_k_down[i] * c_k_dagger_down[i] * fermi_function(energy)
		down_sum += c_q_down[i] * c_q_dagger_down[i] * fermi_function(energy)

	# print("To be implemented")
	return_val = (J /4) * (up_sum - down_sum)
	return return_val

def calc_M2_C(U_dagger, U, Eigs, J):


	up_sum = 0
	down_sum = 0

	c_k_up = get_row(U_dagger, 0)
	c_k_down = get_row(U_dagger, 1)
	c_q_up = get_row(U_dagger, 4)
	c_q_down = get_row(U_dagger, 5)

	c_k_dagger_up = np.conjugate(get_row(U_dagger, 0))
	c_k_dagger_down = np.conjugate(get_row(U_dagger, 1))
	c_q_dagger_up = np.conjugate(get_row(U_dagger, 4))
	c_q_dagger_down = np.conjugate(get_row(U_dagger, 5))

	for i in range(len(Eigs)):
		energy = Eigs[i]
		up_sum += c_k_up[i] * c_q_dagger_up[i] * fermi_function(energy)
		up_sum += c_q_up[i] * c_k_dagger_up[i] * fermi_function(energy)
		down_sum += c_k_down[i] * c_q_dagger_down[i] * fermi_function(energy)
		down_sum += c_q_down[i] * c_k_dagger_down[i] * fermi_function(energy)
	
	return_val = (J /4) * (up_sum - down_sum)
	return return_val
	# print("To be implemented")

def calc_M1_F(U_dagger, U, Eigs, J):

	up_sum = 0
	down_sum = 0

	f_k_up = get_row(U_dagger, 2)
	f_k_down = get_row(U_dagger, 3)
	f_q_up = get_row(U_dagger,6)
	f_q_down =  get_row(U_dagger,7)

	f_k_dagger_up = get_col(U_dagger, 2)
	f_k_dagger_down = get_col(U_dagger, 3)	
	f_q_dagger_up = get_col(U_dagger, 6)
	f_q_dagger_down = get_col(U_dagger, 7)


	for i in range(len(Eigs)):
		energy = Eigs[i]
		up_sum += f_k_up[i] * f_k_dagger_up[i] * fermi_function(energy)
		up_sum += f_q_up[i] * f_q_dagger_up[i] * fermi_function(energy)
		down_sum += f_k_down[i] * f_k_dagger_down[i] * fermi_function(energy)
		down_sum += f_q_down [i]* f_q_dagger_down[i] * fermi_function(energy)

	# print("To be implemented")

	return_val = (J /4) * (up_sum - down_sum)
	return return_val


def calc_M2_F(U_dagger, U, Eigs, J):

	up_sum = 0
	down_sum = 0

	f_k_up = get_row(U_dagger, 2)
	f_k_down = get_row(U_dagger, 3)
	f_q_up = get_row(U_dagger,6)
	f_q_down =  get_row(U_dagger,7)

	f_k_dagger_up = np.conjugate(get_row(U_dagger, 2))
	f_k_dagger_down = np.conjugate(get_row(U_dagger, 3))	
	f_q_dagger_up = np.conjugate(get_row(U_dagger, 6))
	f_q_dagger_down = np.conjugate(get_row(U_dagger, 7))


	for i in range(8):
		energy = Eigs[i]
		up_sum += f_k_up[i] * f_q_dagger_up[i] * fermi_function(energy)
		up_sum += f_q_up[i] * f_k_dagger_up[i] * fermi_function(energy)
		down_sum += f_k_down[i] * f_q_dagger_down[i] * fermi_function(energy)
		down_sum += f_q_down[i] * f_k_dagger_down[i] * fermi_function(energy)

	return_val = (J /4) * (up_sum - down_sum)
	return return_val
		

	# print("To be implemented")

def order_params_calculations(calc_op, guess_op, params, K_POINTS):
	
	# eigen_vals, U_dagger_list = generate_U(guess_op,params, K_POINTS)

	temp_xi1_up = 0
	temp_xi1_down = 0
	temp_xi2_up = 0
	temp_xi2_down = 0

	temp_m1_c = 0
	temp_m2_c = 0
	temp_m1_f = 0
	temp_m2_f = 0

	N = len(K_POINTS)
	# print(N)
	j = params['j']

	mu_f = params['mu_f']
	mu_c = params['mu_c']

	# print("*"*80)
	# print("Generating U matrices")
	# print("*"*80)

	for i in range(len(K_POINTS)):
		kx = K_POINTS[i][0]
		ky = K_POINTS[i][1]

		# print(kx,ky)
		ham = gen_hamiltonian(kx, ky, mu_f,mu_c, True)
		ham  = hamiltonian_order_params(ham, guess_op)
		eigs, U_dagger = LA.eigh(ham)
		U = LA.inv(U_dagger)
	
		temp_xi1_up += calc_xi_one(U_dagger, U, eigs, j, 1)
		temp_xi1_down += calc_xi_one(U_dagger, U, eigs, j, 0)
		
		temp_xi2_up += calc_xi_two(U_dagger, U, eigs, j, 1)
		temp_xi2_down += calc_xi_two(U_dagger, U, eigs, j, 0)

		temp_m1_c += calc_M1_C(U_dagger, U, eigs, j)
		temp_m2_c += calc_M2_C(U_dagger, U, eigs, j)
		
		temp_m1_f += calc_M1_F(U_dagger, U, eigs, j)
		temp_m2_f += calc_M2_F(U_dagger, U, eigs, j)

		# U = np.transpose(np.conjugate(U_dagger))


	'''
	Generate hamiltonian for each K using guess order parameters. After 
	'''	
	calc_op['xi1_up'] = temp_xi1_up  / (2 * N)
	calc_op['xi1_down'] = temp_xi1_down / (2 * N)

	calc_op['xi2_up'] = temp_xi2_up / (2*N)
	calc_op['xi2_down'] = temp_xi2_down / (2*N)
	
	# calc_op['xi2_down'] = 0
	# calc_op['xi2_up'] = 0


	# calc_op['M1_c'] = temp_m1_c / N
	calc_op['M1_c'] = 0

	calc_op['M2_c'] = temp_m2_c / (2*N)
	# calc_op['M2_c'] = 0


	# calc_op['M1_f'] = temp_m1_f / N
	calc_op['M1_f'] = 0

	calc_op['M2_f'] = temp_m2_f / (2*N)
	# calc_op['M2_f'] = 0


	return calc_op


def update_guess_calc(calc_op, guess_op):
	# for param in guess_op.keys():
	# 	guess_op[param] = .2* (calc_op[param]) + .80*(guess_op[param])
	'''
		Uncomment, in case we want to change to
		scaling each param individually
	'''
	guess_op['xi1_up'] = .2* (calc_op['xi1_up']) + .80*(guess_op['xi1_up'])
	guess_op['xi1_down'] = .2* (calc_op['xi1_down']) + .80*(guess_op['xi1_down'])
	guess_op['xi2_up'] = .2* (calc_op['xi2_up']) + .8*(guess_op['xi2_up'])
	guess_op['xi2_down'] = .2* (calc_op['xi2_down']) + .8*(guess_op['xi2_down'])
	# guess_op['M1_c'] = .2* (calc_op['M1_c']) + .8*(guess_op['M1_c'])
	guess_op['M2_c'] = .2* (calc_op['M2_c']) + .8*(guess_op['M2_c'])
	# guess_op['M1_f'] = .2* (calc_op['M1_f']) + .8*(guess_op['M1_f'])
	guess_op['M2_f'] = .2* (calc_op['M2_f']) + .8*(guess_op['M2_f'])
	
	return guess_op

def order_param_init(calculated_order_params, guess = False):
	if(guess):
		A = 2
		B = 3
	else:
		A = 1
		B = -3
	calculated_order_params['xi1_up']  = A
	calculated_order_params['xi1_down']  = A

	calculated_order_params['xi2_up']  = .2
	calculated_order_params['xi2_down']  = .3

	calculated_order_params['M1_c']  = 0
	calculated_order_params['M2_c']  = B

	calculated_order_params['M1_f']  = 0
	calculated_order_params['M2_f']  = B
	return calculated_order_params
def print_params_search(gp, cp):
	for each in gp.keys():
		print(each,"{:18f}".format(abs(gp[each]- cp[each])))

def self_consistent(j, K_POINTS):
	calculated_order_params = {}
	guess_order_params = {}
	params = {}

	params['mu_c'] = -.2
	params['mu_f'] = -.2
	params['j'] = j
	
	guess_order_params = order_param_init(guess_order_params, True)
	# generate_U(guess_order_params, params, K_POINTS)
	calculated_order_params = order_param_init(calculated_order_params)
	params = calibrate_mu(guess_order_params, params, K_POINTS)

	print("Params initalized")
	counter = 0
	while(not order_param_equal(calculated_order_params, guess_order_params)):
		guess_order_params =  update_guess_calc(calculated_order_params, guess_order_params)
		
		params = calibrate_mu(guess_order_params, params, K_POINTS)
		# print(guess_order_params)
		# print(params)

		calculated_order_params = order_params_calculations(calculated_order_params, guess_order_params, params, K_POINTS)


		if(counter %5 == 0):
			print("i = ",counter,'*' * 80)
			print_params_search(guess_order_params, calculated_order_params)
			print('*' * 80)

			
		# print(not order_param_equal(calculated_order_params, guess_order_params))
		counter += 1
	for each in calculated_order_params.keys():
		print(each, calculated_order_params[each])
	calculated_order_params['j'] = j

	for each in params.keys():
		print(each, params[each])

	return calculated_order_params



def gen_hamiltonian(kx,ky,mu_f, mu_c,  chiral, W = 0):
	
	if(chiral):
		epsilon_k = W * (np.sin(kx/2) **2 + np.sin(ky/2)**2)
		epsilon_k_q = W * (np.sin((kx + np.pi)/2) **2 + np.sin((ky+np.pi)/2)**2)
		a_k = np.sin(ky) - 1j* np.sin(kx)
		a_q =  np.sin(ky + np.pi) - 1j* np.sin(kx + np.pi)
		a_k_star = np.sin(ky) + 1j* np.sin(kx)
		a_q_star = np.sin(ky + np.pi) + 1j* np.sin(kx + np.pi)
	else:
		epsilon_k = -2 * (np.cos(kx)  + np.cos(ky))
		epsilon_k_q = -2 * (np.cos(kx + np.pi)  + np.cos(ky+np.pi))

		a_k = 0
		a_q = 0
		a_k_star = 0
		a_q_star = 0
	
	dims = (8,8)
	ham = np.zeros(dims, dtype=complex)

	ham[0][0] = epsilon_k - mu_c
	ham[1][1] = epsilon_k - mu_c

	ham[2][2] = -mu_f - mu_c
	ham[3][3] = -mu_f - mu_c

	ham[4][4] = epsilon_k_q -mu_c
	ham[5][5] = epsilon_k_q -mu_c

	ham[6][6] = -mu_f - mu_c
	ham[7][7] = -mu_f - mu_c

	ham[0][1] = a_k
	ham[1][0] = a_k_star

	ham[4][5] = a_q
	ham[5][4] = a_q_star
			
	return ham

def hamiltonian_order_params(hamiltonian, order_params):
	'''
	Must add the order parameters to hamiltonian. 

	Break Hamilhonian into Four 4 x 4 blocks: 

	| H1 H2 |
	| H3 H4 |

	'''
	#H1 Block Begin ############################################################################## 
	# hamiltonian [0][0] = hamiltonian [0][0] + order_params['M1_f']
	# hamiltonian [1][1] = hamiltonian [1][1] - order_params['M1_f']
	# hamiltonian [2][2] = hamiltonian [2][2] - order_params['M1_c']
	# hamiltonian [3][3] = hamiltonian [3][3] +  order_params['M1_c']


	hamiltonian [0][2] = - order_params['xi1_up']
	hamiltonian [1][3] = - order_params['xi1_down']
	hamiltonian [2][0] = - np.conjugate(order_params['xi1_up'])
	hamiltonian [3][1] = - np.conjugate(order_params['xi1_down'])

	#H4 Block Begin ##############################################################################
	# hamiltonian [4][4] = hamiltonian [4][4] + order_params['M1_f']
	# hamiltonian [5][5] = hamiltonian [5][5] - order_params['M1_f']
	# hamiltonian [6][6] = hamiltonian [6][6] - order_params['M1_c']
	# hamiltonian [7][7] = hamiltonian [7][7] + order_params['M1_c']


	hamiltonian [4][6] = - order_params['xi1_up']
	hamiltonian [5][7] = - order_params['xi1_down']
	hamiltonian [6][4] = - np.conjugate(order_params['xi1_up'])
	hamiltonian [7][5] = - np.conjugate(order_params['xi1_down'])	

	#H2 Block Begin ############################################################################## 
	hamiltonian [0][4] = order_params['M2_f']
	hamiltonian [1][5] =  -order_params['M2_f']
	hamiltonian [2][6] =  -order_params['M2_c']
	hamiltonian [3][7] =  order_params['M2_c']


	hamiltonian [0][6] = - order_params['xi2_up']
	hamiltonian [1][7] = - order_params['xi2_down']
	hamiltonian [2][4] = - np.conjugate(order_params['xi2_up'])
	hamiltonian [3][5] = - np.conjugate(order_params['xi2_down'])

	#H3 Block Begin ############################################################################## 
	hamiltonian [4][0] = order_params['M2_f']
	hamiltonian [5][1] = -order_params['M2_f']
	hamiltonian [6][2] = -order_params['M2_c']
	hamiltonian [7][3] =  order_params['M2_c']


	hamiltonian [4][2] = - order_params['xi2_up']
	hamiltonian [5][3] = - order_params['xi2_down']
	hamiltonian [6][0] = - np.conjugate(order_params['xi2_up'])
	hamiltonian [7][1] = - np.conjugate(order_params['xi2_down'])

	return hamiltonian

def main():
	K_POINTS = gen_brillouin_zone(30)
	# points = gen_brillouin_zone()

	# print(len(K_POINTS))	
	# for i in range(len(K_POINTS)):
	# 	print(K_POINTS[i])

	NUM_PROCESS = 8

	outputs = []
	for j in range(2):
		pool = Pool(processes=NUM_PROCESS)
		results = [pool.apply_async(self_consistent, args=(.5 + 1.6 * j + .2*x, K_POINTS)) for x in range(NUM_PROCESS)]
		output = [p.get() for p in results]
		outputs.append(output)
	log = open("chiral_afm_0.csv","w")
	string = ",".join([str(k) for k in sorted(outputs[0][0].keys())])
	log.write(string)
	log.write("\n")
	for vec in outputs:
		for dic in vec:
			string = ",".join([str(np.real(dic[k])) for k in sorted(dic.keys())])
			log.write(string)
			log.write("\n")
	log.close()

	# results = self_consistent(3.6, K_POINTS)
	# print(results)


# main() 