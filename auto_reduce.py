# -*- coding: utf-8 -*-
from numpy import *
from numpy import linalg as la
import scipy
import scipy.linalg
import sympy
from scipy.integrate import odeint

""" 

 We want to solve the linear system du/dt = Au + h(t)
 
"""

### Setup problem

# Construct A
D = array([-25., -2., -50., 1.])
V = array([[-2., 3., 0., 1.], [-4., 10., 7., -2.], [1., -14., 4., 8.], [9., 1., -2, 1.]])
V_inv = la.inv(V)
A = dot(dot(V, diag(D)), V_inv)

# Non-homegeneity
def h(t):
  return array([-t**2 + 1., -t**2 + 2., cos(t), 5.0*sin(t)])

# Initial condition
u0 = array([1., 2., 3., 4.])
# Output increment  
dt = 0.0001
# End time
T = 1.
# Output times
ts = arange(0., T, dt)
# Number of original equations
N = len(A)
# max_eigen value 
max_eigen = -30.
# Small param
eps = 1. / abs(max_eigen)



"""

Change variables.
 
""" 

# Initial condition for x
x0 = dot(V_inv, u0)

# Non-homegeneity 
def h_tilde(t):
  return dot(V_inv, h(t))


### Identify fast, stable components
rs = D*eps
# U indexes that will be approximated asymptotically
x_a_indexes = rs <= -1.
x_d_indexes = rs > -1.
rs = abs(rs[x_a_indexes])

print x_a_indexes
print x_d_indexes

# Number or fast equations in the transformed system
K = x_a_indexes.sum()

if K == N:
  print "All time scales are fast and stable. Code will break. Code will break anyway."
  quit()



"""

Get asymptotic solutions for fast stable x_i's. 

"""

# Returns the asymptotic approximation for all fast stable x_i's at time t

def x_hat(t):
  tau = t / eps
  Pi0s = x0[x_a_indexes] * e**(-rs * tau)
  xi1s = h_tilde(t)[x_a_indexes] / rs
  Pi1s = (-h_tilde(0)[x_a_indexes] / rs) * e**(-rs * tau)
  return Pi0s + eps*(xi1s + Pi1s) 
  


""" 

Solve for K of the u_i's in terms of the remaining N-K u_i's as well as the 
fast x_i's.

"""
 
S = -zeros((K, N + K))
S[0:K, 0:N] = V_inv[x_a_indexes,:]
S[:,-K:] = diag(ones(K))
S, pivots = sympy.Matrix(S).rref()
S = array(S)

# Indexes of algebraic equations for u_i
u_a_indexes = pivots
# Indexes of differential equations for u_i
u_d_indexes = delete(array(range(len(A))), u_a_indexes)

M = zeros((K, N-K))
M[u_a_indexes, :] = -S[:, u_d_indexes]

J = zeros((K, K))
J[:,:] = S[:,-K:]

print ("u alg. indexes", u_a_indexes)
print ("u dif. indexes", u_d_indexes)



"""

Derive the reduced system: du_prime / dt = A_prime u_prime + h_prime(t)

"""

A1 = A[u_d_indexes,:][:, u_a_indexes]
A2 = A[u_d_indexes,:][:, u_d_indexes]

A_hat = dot(A1, M)  + A2
B_hat = dot(A1, J)

if A_hat.size == 1:
 A_hat = A_hat[0] 
if B_hat.size == 1:
 B_hat = B_hat[0]

def h_hat(t):
  return h(t)[u_d_indexes]


  
from pylab import *

""" 

Solve the original system. 

"""

from pylab import *

def F(u, t):
  du_dt = dot(A, u) + h(t)
  return du_dt

us, inf1 = odeint(F, u0, ts, full_output=True)



""" 

Solve the reduced system. 

"""

### Solve differential equations

# Will contain solution for all unknowns in reduced system
us_hat = zeros((len(ts), N))

u0_hat = u0[u_d_indexes]

def F_hat(u_hat, t):
  du_hat_dt = dot(A_hat, u_hat) + dot(B_hat, x_hat(t)) + h_hat(t)
  return du_hat_dt
  
sol_r, inf2 = odeint(F_hat, u0_hat, ts, full_output=True)
us_hat[:, u_d_indexes] = sol_r


"""

Derive algebraic unknowns.

"""

us_hat[:, u_a_indexes] = (dot(M, sol_r.T) + dot(J, array(map(x_hat, ts)).T)).T



"""

Plot

"""
figure(figsize=(16,8))
suptitle(r'$\lambda_m=' + str(max_eigen) +  '$, Reduced Equations: ' + str(N-K))
for i in range(N):
  subplot(N, 2, 2*i + 1)

  plot(ts, us[:,i], 'r', linewidth = 2.5)  
  plot(ts, us_hat[:,i], 'k--', linewidth = 2.5)
  xlabel(r'$t$')
  ylabel(r'$u_' + str(i) + '$')
  xlim([0., T])
  
  subplot(N, 2, 2*i + 2)
  plot(ts, abs(us[:,i] - us_hat[:,i]), 'r', linewidth = 2.5)  
  xlabel(r'$t$')
  ylabel(r'$u_' + str(i) + '$ error')
  xlim([0., T])

show()
