import numpy as np 
from numpy import linalg as LA
import numpy.matlib
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from multiprocessing import Pool
import matplotlib.pyplot as plt

np.seterr(all='raise')


def generate_hamiltonian(kx,ky,mu_c):
	dims = (2,2)
	hamiltonian = np.zeros(dims, dtype=complex)
	A = np.sin(ky)-1j*np.sin(kx)
	A_star = np.sin(ky)+1j*np.sin(kx)

	epsilon_k = 0.3 * (np.sin(kx/2) ** 2 + np.sin(ky/2) ** 2) 
	hamiltonian[0][0] = epsilon_k - mu_c 
	hamiltonian[0][1] = A 
	hamiltonian[1][0] = A_star
	hamiltonian[1][1] = -epsilon_k - mu_c
	return hamiltonian

def dispersion():
	Pi_approx = 3145
	L = 2 * Pi_approx
	band_1 = []
	band_2 = []
	kx_s = []
	for i in range(L):
		kx = (-Pi_approx + i * 2 * Pi_approx/ L) / 1000
		ky = 0

		H_ti = generate_hamiltonian(kx,ky,0)
		eig_vals = LA.eigvalsh(H_ti)
		kx_s.append(kx)
		band_1.append(eig_vals[0])
		band_2.append(eig_vals[1])

	plt.plot(kx_s,band_1)
	plt.plot(kx_s,band_2)
	plt.xlabel('$k_x$')
	plt.ylabel('E')
	plt.title('TI Surface State Band Structure')
	plt.show()



def main():
	dispersion()

main()