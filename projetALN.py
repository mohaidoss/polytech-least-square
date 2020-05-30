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
param = np.array([30.54])#Initialisation de K pour trouver alpha et rho
m = len(x)
n = len(param)

#Question 1
#Méthode des moindres carrés
def OLS(f,x,y,kappa):   #Fonction pour trouver alpha et rho sachant kappa = 30.54
    a = np.array([[-1]*10,x]).T
    b = np.array([f(y[i]/kappa) for i in range(len(y))])
    Q, R = np.linalg.qr(a)
    alpha_rho = nla.solve_triangular(R,Q.T.dot(b))
    return alpha_rho
param = np.append(param,OLS(logit,x,y,param[0]))#Ajout des nouveaux paramètres trouvés

def r(p):
    result = np.empty(m)
    for i in range(0,m):
        result[i] = s(p,x[i])-y[i]
    return result
        
#Methode de Newton
#Definition de la jacobienne
def m_J(x,y,p):
    b = 0
    for i in range(len(x)):
        t1 = math.exp(-p[2] * x[i] + p[1])
        t2 = 1 + t1
        t2 = 0.1e1 / t2
        t3 = 2
        t3 = t3 * (p[0] * t2 - y[i])
        t4 = t3 * p[0] * t2 ** 2
        b = b + np.array([t3*t2,-t4*t1,t4*x[i]*t1])
    return b
#Definition de la Hessienne
def Hes(x,y,p):
    H = 0
    for i in range(len(x)):
        t1 = math.exp(-p[2]*x[i]+p[1])
        t2 = 1 + t1
        t2 = 0.1e1/t2
        t3 = p[0] * t2
        t4 = t3 - y[i]
        t5 = t2 ** 2
        t6 = t5 * t1
        t3 = t6 * (t3 + t4)
        t7 = 2
        t8 = t7 * t3
        t3 = t7 * t3 * x[i]
        t6 = t6 * p[0]
        t6 = t6 * (t6 - t4)
        t1 = 4 * t4 * p[0] * t2 * t5 * t1 ** 2
        t2 = -t6 * t7 * x[i] - t1 * x[i]
        t4 = x[i] ** 2
        H = H + np.mat([[t7 * t5,-t8,t3],[-t8,t6 * t7 + t1,t2],[t3,t2,t4 * t6 * t7 + t1 * t4]])
    return H

H = Hes(x,y,param)
J = m_J(x,y,param)
#Résolution en utilisant la méthode QR
Q,R=nla.qr_multiply(H,-J)
erreur = nla.solve_triangular(R,Q.T)
param = np.add(param,erreur)
#Résultat plus précis
print(param)
