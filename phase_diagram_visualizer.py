import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import figure
figure(num=None, figsize=(3.5, 4.5), dpi=200, facecolor='w', edgecolor='k')
import math

def main():
	plt.rc('text', usetex=True)
	font = {
	'fontname':'Times New Roman',
	'color':'black',
	'weight':'normal',
	'size':13
	}
	
	plt.rcParams["axes.labelweight"] = "bold"
	plt.rcParams["font.family"] = "serif"
	plt.rcParams["font.serif"]="Times New Roman"
	for i in range(4):
		
		f_name = "kondo_data/phase_diagrams_kondo_" + str(i)+".csv"
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
		for arr in out:
			j.append(float(arr[0]))
			Xi.append(float(arr[1]))
		mu_c = "{0:.2f}".format(i / 8)
		plt.plot(j, Xi, label="$\\mu_c = "+mu_c+"$" )

	plt.gcf().subplots_adjust(left=0.15, bottom=.13, top = .93, right = .97)
	plt.xlabel('$J_0$')
	plt.ylabel('$\\Xi$')
	plt.title('Phase Diagram')
	plt.legend()
	plt.show()
'''
	# print(len(out)) 
def main():
	plt.rc('text', usetex=True)
	font = {
	'fontname':'Times New Roman',
	'color':'black',
	'weight':'normal',
	'size':13
	}
	
	plt.rcParams["axes.labelweight"] = "bold"
	plt.rcParams["font.family"] = "serif"
	plt.rcParams["font.serif"]="Times New Roman"
	for i in range(2):
		t = "Non-Chiral"
		if (i == 0):
			f_name = "phase_diagrams_kondo" +".csv"
		else:
			f_name = "phase_diagrams_chiral_kondo_0.csv"
			t = "Chiral"
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
		for arr in out:
			j.append(float(arr[0]))
			Xi.append(float(arr[1]))
		# mu_c = "{0:.2f}".format(.20 + .05 * i)
		mu_c = str(0)
		plt.plot(j, Xi, label=t + " $\\mu_c = "+mu_c+"$" )
	
	plt.gcf().subplots_adjust(left=0.15, bottom=.13, top = .93, right = .97)
	plt.xlabel('$J_0$')
	plt.ylabel('$\\Xi$')
	plt.title('Phase Diagram')
	plt.legend()
	plt.show()
'''
main()