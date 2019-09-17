import numpy as np 
from numpy import linalg as LA 
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import figure
figure(num=None, figsize=(3.5, 4.5), dpi=200, facecolor='w', edgecolor='k')
import math

def generate_hamiltonian(kx,ky,mu_f, mu_c, Xi_guess):
	dims=(4,4)
	hamiltonian = np.zeros(dims, dtype=complex)
	
	epsilon_k = -2 * (np.cos(kx)  + np.cos(ky)) 
	hamiltonian[0][0] = epsilon_k - mu_c 
	hamiltonian[1][1] = -epsilon_k - mu_c
	hamiltonian[2][2] = -mu_f
	hamiltonian[3][3] = -mu_f
	hamiltonian[0][2] = -Xi_guess
	hamiltonian[1][3] = -Xi_guess
	hamiltonian[2][0] = -np.conj(Xi_guess)
	hamiltonian[3][1] = -np.conj(Xi_guess)
	return hamiltonian


def reader(mu):
	f_name = "kondo_data/phase_diagrams_kondo_" + str(mu)+".csv"
	f = open(f_name,"r")
	out = []
	counter = 0
	for line in f:
		if counter > 0:
			# print(line.rstrip("\n").split(","))
			out.append(line.rstrip("\n").split(","))
		counter +=1
	j = []
	Xi = []
	mu_f  = []
	for arr in out:
		j.append(float(arr[0]))
		Xi.append(float(arr[1]))
		mu_f.append(float(arr[2]))
	return j, Xi, mu_f

def main():
	mu_c = 0
	afm_c,Xi, mu_f = reader(mu_c)
	L= 200
	PI = np.pi
	band_1 = []
	band_2 = []
	band_3 = []
	band_4 = []
	kx_s = []

	rho_s = []
	counter = 30

	file = open("ck_with_dos_0.dat", 'w')
	file.write("j,xi,mu_c,mu_f,rho(0) \n")
	xs = []
	for each in range(-300,300):
		each /= 3000
		xs.append(each)
		rho = 0
		for i in range(L):
			for j in range(L):
				kx = -PI  + 2 * (PI / L * i)
				ky = -PI  + 2 * (PI / L * j)
				H_ti = generate_hamiltonian(kx,ky, mu_f[counter], mu_c/8 , Xi[counter])
				eig_vals = LA.eigvalsh(H_ti)
				for en in eig_vals:
					if (abs(en -each - (mu_c/8)) < 1e-4):
						rho += 1
				# print(en)
		rho = rho / (L**2)
		msg = str(each)+" "+str(afm_c[counter])+","+str(Xi[counter]) + "," +str(mu_c/8) + "," + str(mu_f[counter]) + "," + str(rho)+"\n"
		print(msg)
		# file.write(msg)
		rho_s.append(rho)
		# counter += 1
	file.close()



	# 	kx_s.append(kx)
	# 	band_1.append(eig_vals[0])
	# 	band_2.append(eig_vals[1])
	# 	band_3.append(eig_vals[2])
	# 	band_4.append(eig_vals[3])
	plt.plot(xs,rho_s)
	mu_c = "{0:.2f}".format(i / 8)
	# plt.plot(kx_s,band_2)
	# plt.plot(kx_s,band_2)
	# plt.plot(kx_s,band_4)
	plt.xlabel('$E$')
	plt.ylabel('DOS')
	# plt.title('TI Surface State Band Structure')
	# plt.show()
	plt.savefig("kondo_dos_3_2.png", format="png")			
main()


