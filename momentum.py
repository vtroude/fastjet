import numpy as np
import math
import pylab as pl
from mpl_toolkits.mplot3d import axes3d
from matplotlib import style
from scipy.spatial import Voronoi, voronoi_plot_2d

class Momentum(np.ndarray):
	''' Array of Momentum in Minkowski space '''

    	def __new__(cls, p, m = None, info=None):
		if p.ndim > 1:
			if m is None:
				E = np.sqrt((p**2).sum(axis=1)).reshape(p.shape[0], -1)
			else:
				E = np.sqrt((p**2).sum(axis=1)+m**2).reshape(p.shape[0], -1)
			obj = np.asarray(np.concatenate((E,p), axis=1)).view(cls)
		else:
			if m is None:
				E = np.array([(p**2).sum()**0.5])
			else:
				E = np.array([((p**2).sum()+m**2)**0.5])
			obj = np.asarray(np.concatenate((E,p))).view(cls)
        	obj.info = info
        	return obj

    	def __array_finalize__(self, obj):
        	if obj is None: return
        	self.info = getattr(obj, 'info', None)

	def C(self):			# Pass from the covariant to the contravariant representation
		if self.ndim>1:
			g = -1*np.ones(self.shape[1])
		else:
			g = -1*np.ones(self.shape)
		g[0] = 1
		
		return (g*self).T

	def dot(self, p):		# Minkowski metric
		return np.ndarray.dot(self, p.C())

	def p2(self):			# impulsion norm squared
		if self.ndim>1:
			return (self[:,1:]**2).sum(axis=1).reshape(self.shape[0],-1)
		else:
			return (self[1:]**2).sum()

	def p_norm(self):		# impulsion norm
		return np.sqrt(self.p2())

	# For all the futher method, I consider the beam direection along the z-axis

	def theta_cm(self):		# angle in the xz-plane
		if self.ndim>1:
			sgn = np.sign(self[:,1]).reshape(self.shape[0],-1)
			sgn = np.where(sgn==0, 1, sgn)
			return sgn*np.arccos(self[:,3].reshape(self.shape[0],-1)/self.p_norm())
		else:
			sgn = np.sign(self[1])
			if sgn == 0: sgn=1
			return sgn*np.arccos(self[3]/self.p_norm())

	def rapidity(self):
		return -np.log(np.tan(self.theta_cm()/2))

	def pt2(self):			# norm squared of the transverse momentum (xy-plane)
		if self.ndim>1:
			return np.sum(self[:,1:3]**2, axis=1).reshape(self.shape[0],-1)
		else:
			return np.sum(self[1:3]**2)

	def pt(self):			# norm of the transverse momentum (xy-plane)
		return np.sqrt(self.pt2())

	def phi(self):			# angle in the xy-plane
		if self.ndim>1:
			sgn = np.sign(self[:,2]).reshape(self.shape[0],-1)
			sgn = np.where(sgn==0, 1, sgn)
			return sgn*np.arccos(self[:,1].reshape(self.shape[0],-1)/self.pt()).reshape(self.shape[0],-1)
		else:
			sgn = np.sign(self[2])
			if sgn == 0: sgn=1
			return sgn*np.arccos(self[1]/self.pt())

	def plot(self):
		fig = pl.figure()
		ax1 = fig.add_subplot(111, projection='3d')
		if self.ndim>1:
			d = 0.3*np.ones(self.shape[0])
			ax1.bar3d(self.phi().reshape(-1), self.theta_cm().reshape(-1), np.zeros(self.shape[0]), d, d, self.pt().reshape(-1))
		else:
			ax1.bar3d(self.phi(), self.theta_cm(), self[0], 0.3, 0.3, 1)
		ax1.set_xlabel('$\phi$')
		ax1.set_ylabel("$\Theta_{cm}$")
		ax1.set_zlabel('$p_t$')

	def add(self, p):
		if self.ndim>1:
			r1 = self.shape[0]
			c = self.shape[1]-1
		else:
			r1 = 1
			c = self.shape[0]-1
		if p.ndim>1:
			r2 = p.shape[0]
		else:
			r2 = 1
		a = Momentum(np.zeros((r1+r2,c)))
		a[:r1] = self
		a[r1:] = p
		
		return a

	def unitary(self):
		if self.ndim>1:
			n = np.random.rand(self.shape[1]-1)
		else:
			n = np.random.rand(self.shape[0]-1)
		
		return n/np.sqrt(np.sum(n**2))

	def index_emission(self, E=None):
		if E is None:
			E = np.sum(self[:,0])/self.shape[0]
		i = np.where(self[:,0]>E)
		if len(i)>1:
			return i[np.random.randint(0,i.shape[0])]
		else:
			return np.argmax(self[:,0])

	def emission(self, k=None, E=None, n=None, i=None):
		if n is None:
			n = self.unitary()
		if k is None:
			k = np.random.rand()
		if i is None and self.ndim>1:
			i = self.index_emission(E)
		elif self.ndim==1:
			i = 0
			self = self.reshape(-1, self.shape[0])
		k = k*self[i,0]
		p = Momentum(k*n)
		P = self[i]-p	

		return self[:i].add(P).add(p).add(self[i+1:])

	def normal_to(self, p):
		n = np.random.rand(p.size)
		i = 0
		while i<p.shape[0]-1: 
			if p[i] != 0:
				n[i] = -1.*np.sum(n[i+1:]*p[i+1:])/p[i]
				i = p.shape[0]
			i = i+1
		
		return n/np.sqrt(np.sum(n**2))

	def split(self, b=None, n=None, a=None, i=None):
		if a is None:
			a = 0.8*np.random.rand()+0.1
		if b is None:
			b = np.random.rand()
		b = b*(a*(1.-a))**0.5
		if i is None and self.ndim>1:
			i = np.random.randint(0, self.shape[0])
		elif self.ndim==1:
			self = self.reshape(-1,self.shape[0])
			i = 0
		b = b*self[i,0]
		if n is None:
			n = self.normal_to(self[i,1:])
		p1 = Momentum(a*self[i,1:]+b*n)
		p2 = Momentum((1.-a)*self[i,1:]-b*n)

		return self[:i].add(p1).add(p2).add(self[i+1:])
			
			


###################################################

''' Test the Momentum class '''



def test_momentum():

	Pa = Momentum(np.array([[1,0,0], [1,1,0]]))
	Pb = Momentum(np.array([0,1,1]))
	print "Two sets of Momentum"
	print Pa
	print Pb
	print
	print "Dot Product"
	print Pa.dot(Pb)
	print 
	print "Transverse Momentum"
	print Pa.pt()
	print Pb.pt()
	print
	print "Pseudo-rapidity"
	print Pa.rapidity()
	print Pb.rapidity()
	print
	print "Azimuth angle"
	print Pa.phi()
	print Pb.phi()
	print
	print "Concatenate the two"
	print Pa.add(Pb)
	print

def test_emission():
	P = Momentum(np.array([1,1,1]))
	print "Initial Momentum"
	print P
	print
	p = P.emission()
	print "After emission"
	print p
	print
	print "Energy conservation Ei-Ef"
	print P[0]-np.sum(p[:,0])
	print
	print "Momentum conservation |Pi-Pf|"
	print np.sqrt(np.sum((P[1:]-p[0,1:]-p[1,1:])**2))
	print

def test_split():
	P = Momentum(np.array([1,1,1]))
	print "Initial Momentum"
	print P
	print
	p = P.split()
	print "After emission"
	print p
	print
	print "Energy conservation Ei-Ef"
	print P[0]-np.sum(p[:,0])
	print
	print "Momentum conservation |Pi-Pf|"
	print np.sqrt(np.sum((P[1:]-p[0,1:]-p[1,1:])**2))
	print

#test_momentum()
#test_emission()
#test_split()

##################################################

''' Methods to generate random momenta '''




def create_impulsion(p):
	theta = math.pi*np.random.random()
	phi = 2*math.pi*np.random.random()

	return p*np.array([math.sin(theta)*math.cos(phi), math.sin(theta)*math.sin(phi), math.cos(theta)])

def create_rand_impulsion(E):
	p = E*(np.random.random()*0.5)
	E = E-p

	return E, create_impulsion(p)

def momentum_generator(E, N):	# E: Total Energy, N: number of particles
	# This method conserved the Energy but not the momenta
	P = np.zeros((N,3))
	for n in range(N-1):
		E, P[n] = create_rand_impulsion(E)
	P[N-1] = -np.sum(P, axis=0)

	return Momentum(P)


#######################################################

''' Test the generator '''

def test_generator():
	M = momentum_generator(1, 10)
	print M
	print M.sum(axis=0)

def plot_momentum():
	momentum_generator(1,3).plot()
	pl.show()

def test_voronoi():
	p = momentum_generator(1,10)
	mu_phi = np.concatenate((p.theta_cm().reshape(-1,1),p.phi().reshape(-1,1)), axis=1)
	vor = Voronoi(mu_phi)
	print 'Points'
	print vor.points
	print
	print 'Vertices'
	print vor.vertices
	print
	print 'Regions'
	print vor.regions
	print
	print 'Ridge Points'
	print vor.ridge_points
	print
	voronoi_plot_2d(vor)
	pl.show()
	
	

#test_generator()
#plot_momentum()
#test_voronoi()



















