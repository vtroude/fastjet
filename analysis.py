import numpy as np
import time
import scipy as sc
import pylab as pl
import momentum
import sra

def jet_E_max(jets):
	E = np.zeros(len(jets))
	for n in range(len(jets)):
		E[n] = jets[n][0]
	
	return np.max(E)

def complexity(ja, ni, nf, nstep, M, E=1.):
	N = np.array(range(ni, nf, nstep))
	T = np.zeros((len(N), M))
	i = 0
	for n in N:
		for m in range(M):
			momenta = momentum.momentum_generator(E, n)
			t = time.time()
			_ = ja.launch(momenta)
			T[i,m] = T[i,m] + time.time() - t
		i = i+1
	T_bar = np.mean(T, axis=1)
	#Y1 = N*np.log(N)
	#Y2 = N**2
	Y = N**3
	#a1 = (T_bar[0]-T_bar[-1])/(Y1[0]-Y1[-1])
	#a2 = (T_bar[0]-T_bar[-1])/(Y2[0]-Y2[-1])
	a = (T_bar[0]-T_bar[-1])/(Y[0]-Y[-1])
	#b1 = (T_bar[0]+T_bar[-1] - a1*(Y1[0]+Y1[-1]))*0.5
	#b2 = (T_bar[0]+T_bar[-1] - a2*(Y2[0]+Y2[-1]))*0.5
	b = (T_bar[0]+T_bar[-1] - a*(Y[0]+Y[-1]))*0.5
	pl.plot(N, T[:,0], 'b.', label='T(N)')	
	pl.plot(N, T[:,1:], 'b.')
	pl.plot(N, T_bar, 'ro', label='<T(N)>')
	#pl.plot(N, a1*Y1+b1, label='O(Nln(N))')
	#pl.plot(N, a2*Y2+b2, 'g-', label='O(N$^2$)')
	pl.plot(N, a*Y+b, label='O(N$^3$)')
	pl.xlabel('N')
	pl.ylabel('T')
	pl.legend()
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
		#N_j0 = len(ja.launch(momenta))
		E_j0 = jet_E_max(ja.launch(momenta))
		for i in range(n):
			#observable[m,i] = float(len(ja.launch(momenta.split(k[i]))))/N_j0
			observable[m,i] = jet_E_max(ja.launch(momenta.split(k[i])))/E_j0
		pl.plot(np.log10(k), observable[m], '.')
	pl.xlabel('log(k)')
	#pl.ylabel('N/N_0')
	pl.ylabel('E/E_0')
	pl.title('C-safety')
	pl.show()

def IRC_visual(ja, N, n, n_, N_split, N_emission):
	k = 10.**(-1*np.geomspace(0.5, n_, num=n))
	print k
	momenta = momentum.momentum_generator(1,N)
	momenta.plot()
	ja.list_to_momentum(ja.launch(momenta)).plot()
	N = []
	A = 0.8*np.random.rand(N_split)+0.1
	I = []
	P = momenta
	for i in range(N_split):
		I.append(np.random.randint(0, P.shape[0]))
		N.append(P.normal_to(P[I[i],1:]))
		P = P.split(k[0], N[i], A[i], I[i])
	for i in range(N_split,N_emission+N_split):
		I.append(P.index_emission())
		N.append(P.unitary())
		P = P.emission(k[0], n=N[i], i=I[i])
	P.plot()
	ja.list_to_momentum(ja.launch(P)).plot()
	for i in range(1,len(k)):
		P = momenta
		for j in range(N_split):
			P = P.split(k[i], N[j], A[j], I[j])
		for j in range(N_split, N_split+N_emission):
			P = P.emission(k[i], n=N[j], i=I[j])
		P.plot()
		ja.list_to_momentum(ja.launch(P)).plot()
	pl.show()
	

ja = sra.SRAG()	# Jet algorithm

complexity(ja, 100, 1000, 100, 10)
#test_IR_safety(ja, 100, 3, 100, 10)
#test_C_safety(ja, 100, 3, 100, 2.)
#IRC_visual(ja, 5, 3, 10, 10, 10)






