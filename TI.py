import numpy as np 
from numpy import linalg as LA
import numpy.matlib
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from multiprocessing import Pool
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

# import matplotlib.pyplot as plt
# import numpy as np


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
	# fig = plt.figure()
	# ax = fig.gca(projection='3d')
	x, y, z = np.meshgrid(np.arange(-np.pi, 1, 0.2),
	                      np.arange(-np.pi, 1, 0.2),
	                      np.arange(-1, 1, 0.8))


	# ax.quiver(x, y, z, length=0.1, normalize=True)

	# plt.show()
	
	Pi_approx = np.pi
	L = 20
	band_1 = []
	band_2 = []
	kx_s = []
	ky_s = []

	x_moments = []

	sx = np.array([[0, 1],[ 1, 0]])
	sy = np.array([[0, -1j],[1j, 0]])
	sz = np.array([[1, 0],[0, -1]])

	for i in range(L):
		for j in range(L):
			kx = (-Pi_approx + i * 2 * Pi_approx/ L)
			ky = (-Pi_approx + j * 2 * Pi_approx/ L)
			H_ti = generate_hamiltonian(kx,ky,0)
			eig_vals, eigen_vecs = LA.eigh(H_ti)

			for i in range(2):
				x_moment = np.conjugate(eigen_vecs[:,i])@ sx @ eigen_vecs[:,i]
			x_moments.append(x_moment)
			kx_s.append(kx)
			ky_s.append(ky_s)
		# band_1.append(eig_vals[0])
		# band_2.append(eig_vals[1])

	# plt.plot(kx_s,band_1)
	# plt.plot(kx_s,band_2)
	plt.xlabel('$k_x$')
	plt.ylabel('E')
	plt.title('TI Surface State Band Structure')
	plt.show()



def main():
	dispersion()

main()