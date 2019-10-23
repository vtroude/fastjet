import numpy as np
import time
import pylab as pl
import momentum
import sra

def jet_E_max(jets):
	E = np.zeros(len(jets))
	for n in range(len(jets)):
		E[n] = jets[n][0]
	
	return np.max(E)

def complexity(ja, ni, nf, nstep, M, E=1.):
	N = range(ni, nf, nstep)
	T = np.zeros((len(N), M))
	i = 0
	for n in N:
		for m in range(M):
			momenta = momentum.momentum_generator(E, n)
			t = time.time()
			_ = ja.launch(momenta)
			T[i,m] = T[i,m] + time.time() - t
		i = i+1
	pl.plot(N, np.mean(T, axis=1)**(1./2), 'o')
	pl.plot(N, T**(1./2), 'b.')
	pl.xlabel('N')
	pl.ylabel('$T^{1/3}$')
	pl.show()

def test_IR_safety(ja, N, M, n, n_):
	observable = np.zeros((M, n))
	k = 10.**(-1*np.geomspace(0.5, n_, num=n))
	for m in range(M):
		momenta = momentum.momentum_generator(1, N)
		N_j0 = float(len(ja.launch(momenta)))
		#E_j0 = jet_E_max(ja.launch(momenta))	
		for i in range(n):
			observable[m,i] = float(len(ja.launch(momenta.emission(k[i]))))/N_j0
			#observable[m,i] = jet_E_max(ja.launch(momenta.emission(k[i])))/E_j0
		pl.plot(np.log10(k), observable[m], '.')
	pl.xlabel('log(k)')
	pl.ylabel('N/N_0')
	#pl.ylabel('E/E_0')
	pl.title('IR-Safety')	
	pl.show()
		
def test_C_safety(ja, N, M, n, n_):
	observable = np.zeros((M, n))
	k = 10.**(-1*np.geomspace(0.01, n_, num=n))
	for m in range(M):
		momenta = momentum.momentum_generator(1, N)
		N_j0 = len(ja.launch(momenta))
		#E_j0 = jet_E_max(ja.launch(momenta))
		for i in range(n):
			observable[m,i] = float(len(ja.launch(momenta.split(k[i]))))/N_j0
			#observable[m,i] = jet_E_max(ja.launch(momenta.split(k[i])))/E_j0
		pl.plot(np.log10(k), observable[m], '.')
	pl.xlabel('log(k)')
	pl.ylabel('N/N_0')
	#pl.ylabel('E/E_0')
	pl.title('C-safety')
	pl.show()
	

ja = sra.FJA(.5)	# Jet algorithm

#complexity(ja, 100, 1000, 100, 10)
test_IR_safety(ja, 100, 3, 100, 10)
#test_C_safety(ja, 100, 3, 100, 1.5)






