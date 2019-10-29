from abc import abstractmethod
import numpy as np
import math
import pylab as pl
import momentum

class JA:
	''' Jet Algorithm '''

	def __init__(self): pass

	@abstractmethod
	def beamDistance(self, momenta): pass
	@abstractmethod	
	def distance(self, momenta, index=None): pass
	@abstractmethod
	def cut_off(self, jet, cut_off): pass
	@abstractmethod
	def launch(self, momenta, cut_off=None): pass

	def list_to_momentum(self, jets):
		J = momentum.Momentum(np.zeros((len(jets), jets[0].shape[0]-1)))
		for n in range(len(jets)):
			J[n] = jets[n]

		return J

	def erase_momentum(self, momenta, i):
		return np.delete(momenta, i, axis=0)

	def erase_distance(self, d, i):
		return np.delete(np.delete(d, i, axis=0), i, axis=1)

	def erase(self, momenta, d, i):
		return self.erase_momentum(momenta, i), self.erase_distance(d, i)

	def adjust_distance(self, momenta, d, i):
		d_i = self.distance(momenta, i)
		d[:,i] = d_i
		d[i,:] = d_i

		return d

	def merge(self, momenta, d, i, j):
		I = np.minimum(i,j)
		momenta[I] = momenta[i]+momenta[j]
		momenta, d = self.erase(momenta, d, np.maximum(i,j))
		d = self.adjust_distance(momenta, d, I)
	
		return momenta, d

	def plot(self, jet, momenta):
		if len(jet)>0:
			momenta.add(self.list_to_momentum(jet)).plot()
		else:
			momenta.plot()

class SRA(JA):
	''' Sequential Recombination Algorithm '''

	def __init__(self): pass

	def launch(self, momenta, cut_off=None, show=False):
		d = self.distance(momenta)	# Distance between particles for i!=j & beam distance on the diagonal
		jets = []			# Table of Jet found by the algorithm
		if show:
			step = int(momenta.shape[0]/4)
			i=0
		while momenta.shape[0]>0:	# While the momentum table is not empty
			if show and i%step==0:
				self.plot(jets, momenta)
			argDmin = np.unravel_index(np.argmin(d, axis=None), d.shape)
			if argDmin[0]==argDmin[1]:	# Beam distance is the minimum
				if self.cut_off(momenta[argDmin[0]], cut_off):
					jets.append(momenta[argDmin[0]]) 	# add it to the list
				momenta, d = self.erase(momenta, d, argDmin[0])		# delete the corespounding distance
			else:
				momenta, d = self.merge(momenta, d, argDmin[0], argDmin[1])
			if show:
					i=i+1
			

		return jets
 

class SRA_trivial(SRA):
	''' SRA using the Trivial (Minkowski) metric '''

	def __init__(self, Dbeam=0.):
		self.Dbeam = Dbeam		# A constant value used as the beam distance

	def beamDistance(self, momenta):
		return self.Dbeam*np.ones(momenta)

	def distance(self, momenta, index=None):
		if index is None:
			d = 2*momenta.dot(momenta)
			d = np.diag(self.beamDistance(momenta.shape[0]))+d-np.diag(np.diag(d))
		else:
			d = 2*momenta.dot(momenta[index])
			d[index] = self.beamDistance(1)

		return d

	def cut_off(self, jet, cut_off):
		if (cut_off is None) or jet.p_norm()>cut_off:
			return True
		else:
			return False


class SRAG(SRA):
	''' SRA using a generalized metric  '''

	def __init__(self, alpha=1., R=1.):
		self.alpha = alpha		# exponent used for the transverse momentum
		self.R = R			# radius parameter

	def delta(self, momenta, index=None):
		azimuth = np.array(momenta.phi())
		rapidity = np.array(momenta.theta_cm())
		if index is None:
			return ((azimuth-azimuth.T)**2+(rapidity-rapidity.T)**2)/self.R/self.R
		else:
			return ((azimuth[index]-azimuth)**2+(rapidity[index]-rapidity)**2)/self.R/self.R

	def beamDistance(self, momenta):
		return np.array(momenta.pt2()**self.alpha)

	def distance(self, momenta, index=None):
		kt = self.beamDistance(momenta)
		if index is None:
			d = np.minimum(kt, kt.T)*self.delta(momenta) + np.diag(kt.reshape(-1))
		else:
			d = np.minimum(kt[index], kt)*self.delta(momenta, index)
			d[index] = kt[index]
			d = d.reshape(-1)

		return d

	def cut_off(self, jet, cut_off):
		if ((cut_off is None) or (jet.pt()>cut_off)):
			return True
		else:
			return False

class FJA(SRAG):
	''' FastJet Algorithm '''

	def __init__(self, alpha=1., R=1.):
		self.alpha = alpha		# exponent used for the transverse momentum
		self.R = R			# radius parameter

	def nearest_neighbour(self, momenta):
		delta = self.delta(momenta, 0).reshape(-1)
		g_i = np.where(delta<1)
		kt = self.beamDistance(momenta[g_i]).reshape(-1)
		d_i = np.minimum(kt, kt[0])*delta[g_i]
		d_i[0] = kt[0]

		return d_i, g_i[0]

	def merge(self, momenta, d, i, j):
		momenta[i] = momenta[i]+momenta[j]
		momenta = self.erase_momentum(momenta, j)
		d_i, g_i = self.nearest_neighbour(momenta)
	
		return momenta, d_i, g_i

	def launch(self, momenta, cut_off=None, show=False):
		jets = []
		if show:
			step = int(momenta.shape[0]/4)
			i=0
		while momenta.shape[0]>0:
			d_i, g_i = self.nearest_neighbour(momenta)
			while d_i.shape[0]>0:
				if show and i%step==0:
					self.plot(jets, momenta)
				argDmin = np.argmin(d_i)
				if argDmin==0:
					if self.cut_off(momenta[g_i[argDmin]], cut_off):
						jets.append(momenta[g_i[argDmin]]) 	# add it to the list
					momenta = self.erase_momentum(momenta, g_i[argDmin])		# delete the corespounding distance
					d_i = np.delete(d_i, argDmin)
				else:
					momenta, d_i, g_i = self.merge(momenta, d_i, 0, g_i[argDmin])
				if show:
					i=i+1

		return jets			

##########################################

''' Test algorithms '''


def test_algorithms(ja):
	phi = 2.*math.pi/3.
	P = momentum.Momentum(np.array([[1.,0.,0.], [math.cos(phi), math.sin(phi), 0.], [math.cos(phi*2.), math.sin(phi*2.), 0.]]))
	P.plot()
	print 'Momenta'
	print P
	print 
	print 'Jets'
	jets = ja.launch(P)
	for n in range(len(jets)):
		print jets[n]
	print
	ja.list_to_momentum(jets).plot()
	p = P
	for n in range(10):
		p = p.split(0.01)
	for n in range(10):
		p = p.emission(0.0001)
	p.plot()
	print 'New momenta'
	print p
	print np.sum(p, axis=0)
	print
	print 'new Jets'
	jets_new = ja.launch(p)
	for n in range(len(jets_new)):
		print jets_new[n]
	print
	ja.list_to_momentum(jets_new).plot()
	pl.show()

def test(ja, N):
	ja.list_to_momentum(ja.launch(momentum.momentum_generator(float(N),N), show=True)).plot()
	pl.show()

#ja = SRAG(R=1.5)
#test_algorithms(ja)
#test(ja, 50)




























