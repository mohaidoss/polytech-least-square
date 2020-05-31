import math
import numpy as np
import scipy.linalg as nla
import matplotlib.pyplot as plt

def logit(p):
    return math.log(p/(1-p))
def s(p,x):
    return p[0]/(1 + math.exp(p[1]-p[2]*x))


#Initialisation des données

x = np.arange(10)
y = np.array([.53, .53, 1.53, 2.53, 12.53, 21.53, 24.53, 28.53, 28.53, 30.53])
param = np.array([30.54])#Initialisation de K pour trouver alpha et rho

#Question 1
#Méthode des moindres carrés
def OLS(f,x,y,kappa):   #Fonction pour trouver alpha et rho sachant kappa = 30.54
    a = np.array([[-1]*10,x]).T
    b = np.array([f(y[i]/kappa) for i in range(len(y))])
    Q, R = np.linalg.qr(a)
    alpha_rho = nla.solve_triangular(R,Q.T.dot(b))
    return alpha_rho
param = np.append(param,OLS(logit,x,y,param[0]))#Ajout des nouveaux paramètres trouvés
        
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
#Résolution en utilisant la méthode QR, application de la méthode de Newton
Q,R=nla.qr_multiply(H,-J)
erreur = nla.solve_triangular(R,Q.T)
param_Newton = np.add(param,erreur)

#Méthode de Gauss-Newton
def m_JT_r(x,y,p):
    b = 0
    for i in range(len(x)):
        t1 = math.exp(-p[2] * x[i] + p[1])
        t2 = 1 + t1
        t2 = 0.1e1 / t2
        t3 = p[0] * t2 - y[i]
        t1 = p[0] * t2 ** 2 * t1 * t3
        b = b + np.array([-t2 * t3,t1,-t1 * x[i]])
    return b
Jr = m_JT_r(x,y,param)
def JT_J(x,y,p):
    A = 0
    for i in range(len(x)):
        t1 = math.exp(-p[2] * x[i] + p[1])
        t2 = 1 + t1
        t2 = 0.1e1 / t2
        t3 = t2 ** 2
        t2 = t2 * t3 * p[0]
        t4 = t2 * t1
        t2 = t2 * x[i] * t1
        t1 = p[0] ** 2 * t3 ** 2 * t1 ** 2
        t5 = t1 * x[i]
        A = A + np.array([[t3,-t4,t2],[-t4,t1,-t5],[t2,-t5,t1 * x[i] **2]])
    return A
Jt = JT_J(x,y,param)

Q,R=nla.qr_multiply(Jt,Jr)
erreur = nla.solve_triangular(R,Q)
paramGaussNewton = np.add(param,erreur)


print("En utilisant la méthode des moindres carrés kappa,alpha,rho = ", param)
print("En utilisant la méthode de Newton kappa,alpha,rho = ", param_Newton)
print("En utilisant la méthode de Gauss-Newton kappa,alpha,rho = ", paramGaussNewton)

Y=np.empty(len(x))
Y_Newton = np.empty(len(x))
Y_GaussNewton = np.empty(len(x))

for i in range(len(x)):
    Y[i] = np.array([s(param,x[i])])
    Y_Newton[i] = np.array([s(param_Newton,x[i])])
    Y_GaussNewton[i] = np.array([s(paramGaussNewton,x[i])])

plt.plot(x,y,label = "Courbe 0")
plt.plot(x,Y,label = "Courbe avec méthode des moindres carrés")
plt.plot(x,Y_Newton,label = "Courbe avec méthode de Newton")
plt.plot(x,Y_GaussNewton,label = "Courbe avec méthode de GaussNewton")
plt.legend()
plt.show()
#Question 3 
#Comparaison graphique des courbes
plt.plot(x,Y_Newton,label = "Courbe avec méthode de Newton")
plt.plot(x,Y_GaussNewton,label = "Courbe avec méthode de GaussNewton")
plt.legend()
plt.show()

###################################
#Question 4
#Initialisation des données pour la Question 4

x = np.arange(10)
y = np.array([51, 51, 52, 53, 63.53, 72, 75, 79, 79, 81])
param = np.empty(4)
param[0] = 30.54  #Initialisation de kappa
param[3] = 50.47    #Initialisation de Lambda
def s4(p,x):
    return p[0]/(1 + math.exp(p[1]-p[2]*x)) + p[3]


#MOINDRE CARRE ENCORE
   
#Méthode des moindres carrés
#Fonction pour trouver alpha et rho sachant kappa = 51 et lambda = 50.47
def OLS4(f,x,y,kappa,lm):   
    a = np.array([[-1]*10,x]).T
    b = np.array([f((y[i]-lm)/kappa) for i in range(len(y))])
    Q, R = np.linalg.qr(a)
    alpha_rho = nla.solve_triangular(R,Q.T.dot(b))
    return alpha_rho
param[1:3] = OLS4(logit,x,y,param[0],param[3])

#En utilisant la méthode de Gauss-Newton
def m_JT_r4(x,y,p):
    b = 0
    for i in range(len(x)):
        t1 = math.exp(-p[2] * x[i] + p[1])
        t2 = 1 + t1
        t2 = 0.1e1 / t2
        t3 = p[0] * t2 + p[3] - y[i]
        t1 = p[0] * t2 ** 2 * t1 * t3
        b = b + np.array([-t2 * t3,t1,-t1 * x[i],-t3])
    return b

def JT_J4(x,y,p):
    A = 0
    for i in range(len(x)):
        t1 = math.exp(-p[2] * x[i] + p[1])
        t2 = 1 + t1
        t2 = 0.1e1 / t2
        t3 = t2 ** 2
        t4 = t2 * t3 * p[0]
        t5 = t4 * t1
        t4 = t4 * x[i]* t1
        t6 = p[0] * t3 * t1
        t1 = p[0] ** 2 * t3 ** 2 * t1 ** 2
        t7 = t1 * x[i]
        t8 = t6 * x[i]
        A = A + np.mat([[t3,-t5,t4,t2],[-t5,t1,-t7,-t6],[t4,-t7,t1 * x[i] ** 2,t8],[t2,-t6,t8,1]])
    return A
Jr4 = m_JT_r4(x,y,param)
Jt4 = JT_J4(x,y,param)
Q,R=nla.qr_multiply(Jt4,Jr4)
erreur = nla.solve_triangular(R,Q)
paramGaussNewton4 = np.add(param,erreur)
print("Les paramètres kappa,alpha,rho,lambda =",paramGaussNewton4)
Y_GaussNewton4=np.empty(len(y))
for i in range(len(y)):
    Y_GaussNewton4[i] = np.array([s4(paramGaussNewton4,x[i])])

plt.plot(x,y,label = "Courbe de la question 4")
plt.plot(x,Y_GaussNewton4,label = "Courbe de la question 4 en utilisant Gauss-Newton")

plt.legend()
plt.show()
