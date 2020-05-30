import math
import numpy as np
import scipy.linalg as nla
import matplotlib.pyplot as plt

def logit(p):
    return math.log(p/(1-p))
def s(p,x):
    return p[0]/(1 + exp(p[1]-p[2]*x))


#Initialisation des données

x = np.arange(10)
y = np.array([.53, .53, 1.53, 2.53, 12.53, 21.53, 24.53, 28.53, 28.53, 30.53])
param = np.array([30.54,0,0])#Initialisation de K pour trouver alpha et rho
m = len(x)
n = len(param)

#Question 1
#Méthode des moindres carrés
"""
a = np.array([[-1]*10,x]).T
b = np.array([logit(y[i]/param[0]) for i in range(len(y))])
print(b)
#Decomposition en QR
Q, R = np.linalg.qr(a)
print(Q)
print(R)
print(Q.T.dot(b))
#Solution
sol = nla.solve_triangular(R,Q.T.dot(b))
#nla.norm(A.dot(sol)-b,2)
print(sol)
"""
def OLS(f,x,y,kappa):
    a = np.array([[-1]*10,x]).T
    b = np.array([f(y[i]/kappa) for i in range(len(y))])
    Q, R = np.linalg.qr(a, mode = "full")
    alpha_rho = nla.solve_triangular(R,Q.T.dot(b))
    return alpha_rho
print(OLS(logit,x,y,param[0]))

def r(p):
    result = np.empty(m)
    for i in range(0,m):
        result[i] = s(p,x[i])-y[i]
    return result
        


def M_jT_r(J):
    for i in range(0,m):
        t1 = math.exp(-rho * x[i - 1] + alpha)
        t2 = 1 + t1
        t2 = 0.1e1 / t2
        t3 = kappa * t2 - y[i - 1]
        t1 = kappa * t2 ** 2 * t1 * t3
        b = numpy.mat([-t2 * t3,t1,-t1 * x[i - 1]])
    return b
