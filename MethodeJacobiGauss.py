#author Gerbaud FLorent
#Algorithme LU
#03/03/2023

import numpy as np
import math
import matplotlib.pyplot as plt
from copy import deepcopy

def invMatLower(A):
    dim=A.shape[1]
    Ac=np.copy(A)
    X=np.zeros_like(A)
    somme=0
    for i in range(0,dim):
        X[i,i]=1/Ac[i,i]
        for j in range(i):
            for k in range(j,i):
                # if(i==4 and j==3):
                #     print(k)
                #     print(Ac[i,k],"*",X[k,j],"/",Ac[i,i])
                somme=somme+Ac[i,k]*X[k,j] 
            X[i,j]=-somme/Ac[i,i]
            somme=0
    return X

def genereMat(n):
    return np.diag([-1.]*(n-1),-1)+np.diag([-1.]*(n-1),1)+np.diag([2.]*n)

#Méthode qui applique la résolution de jacobi
def jacobi(A,b,x0,eps,Nmax):
    
    x=deepcopy(x0)  #valeur de x a la 0eme iteration
    Ac=deepcopy(A) 
    dim=Ac.shape[1]
    N=-np.tril(Ac,-1)-np.triu(Ac,1) #we define the matrix N (-E-F) -E:lower matrix -F upper matrix
    M=np.diag(1.0/np.diag(Ac)) #we define Dinv to calculate x at the k-eme etape
    k=0
    residu=[]
    residu.append(np.linalg.norm(np.dot(Ac,x)-b)/np.linalg.norm(b))
    while(np.linalg.norm(np.dot(Ac,x)-b)/np.linalg.norm(b)>eps and k<Nmax): #conditions to stop the prog
        x=np.dot(M,(np.dot(N,x)+b)) #x to the k-ieme etape
        k=k+1
        residu.append(np.linalg.norm(np.dot(Ac,x)-b)/np.linalg.norm(b))
    return x, k,residu

def gaussSeidel(A,b,x0,eps,Nmax):
    
    x=deepcopy(x0)  #valeur de x a la 0eme iteration
    Ac=deepcopy(A) 
    dim=Ac.shape[1]
    N=-np.triu(Ac,1) #we define the matrix N (-E-F) -E:lower matrix -F upper matrix
    M=np.diag(np.diag(Ac))+np.tril(Ac,-1)
    # Dinv=np.linalg.inv(M) #we define Dinv to calculate x at the k-eme etape
    Dinv=invMatLower(M)
    # print(Dinv)
    k=0
    residu=[]
    residu.append(np.linalg.norm(np.dot(Ac,x)-b)/np.linalg.norm(b))
    while(np.linalg.norm(np.dot(Ac,x)-b)/np.linalg.norm(b)>eps and k<Nmax): #conditions to stop the prog
        x=np.dot(Dinv,(np.dot(N,x)+b)) #x to the k-ieme etape
        k=k+1
        residu.append(np.linalg.norm(np.dot(Ac,x)-b)/np.linalg.norm(b))
    return x,k,residu

def SOR(A,b,x0,eps,Nmax,w):
    x=deepcopy(x0)  #valeur de x a la 0eme iteration
    Ac=deepcopy(A) 
    dim=Ac.shape[1]
    N=-np.triu(Ac,1) #we define the matrix N (-E-F) -E:lower matrix -F upper matrix
    M=np.diag(np.diag(Ac))+w*np.tril(Ac,-1)
    # Dinv=np.linalg.inv(M) #we define Dinv to calculate x at the k-eme etape
    Dinv=invMatLower(M)
    # print(Dinv)
    k=0
    residu=[]
    residu.append(np.linalg.norm(np.dot(Ac,x)-b)/np.linalg.norm(b))
    while(np.linalg.norm(np.dot(Ac,x)-b)/np.linalg.norm(b)>eps and k<Nmax): #conditions to stop the prog
        x=np.dot(Dinv,(np.dot(np.dot(w,N),x)+np.dot(w,b)+np.dot(1.0-w,np.dot(np.diag(np.diag(Ac)),x)))) #x to the k-ieme etape
        residu.append(np.linalg.norm(np.dot(Ac,x)-b)/np.linalg.norm(b))
        k=k+1
    return x,k,residu
    
############################# test ####################################
############################# Initialisation of param ################

n=10
# A = np.array([[3., 1., -1.],[1., 2., 0.],[-1., 1., 4.]]) #avec les deux matrices A, les deux méthodes sont cv
A=genereMat(n) #tridiagonale Matrix
B = np.array([[1., 3/4, 3/4],[3/4, 1., 3/4],[3/4, 3/4, 1.]]) # avec la matrice B, Jacobi DV, GS cv
C = np.array([[1., 2., -2.],[1., 1., 1.],[2., 2., 1.]]) #avec la C, Jacobi cv, GS dv
# print(A)
x0 = np.zeros((n, 1))
b= np.ones((n, 1))

############################# Jacobi ####################################    

x1,k1,residues1=jacobi(C,b,x0,1e-8,300)
print(x1,k1,residues1[-1])

############################# Gauss Seidel ####################################  

x2,k2,residues2=gaussSeidel(A,b,x0,1e-8,3000)
print(x2,k2)


############################# SOR ####################################  

# x3,k3,residues3=SOR(A,b,x0,1e-8,3000,0.05)
# print(residues3[-1])
# print(k3)

########################### plot eps ########################################

# x1,k1,residues1=jacobi(A,b,x0,1e-7,1000)
# x2,k2,residues2=gaussSeidel(A,b,x0,1e-7,10000)
   
# plt.plot(np.arange(0,k1+1),residues1, label='Residus méthodes de Jacobi')
# plt.plot(np.arange(0,k2+1),residues2, label='Résidus méthode Gauss Seidel')
# plt.yscale('log')
# plt.legend()
# plt.title('Convergence rates comparison')
# plt.xlabel('Number of iterations')
# plt.ylabel('Precision')
# plt.show()

########################### find w ########################################

# tour=400
# tabIteration=[]
# tabW=[]
# rayonSpect=[]

# iteration=0
# w=0
# for i in range(1,tour):
#     w=i*(1/200.0)
#     x3,iteration,residues3=SOR(A,b,x0,1e-6,1000,w) #move the number of iteration max to have the more precision as possible and find the w
#     tabIteration.append(iteration)
#     tabW.append(w)
#     # radius spectral 
    
#     N=-np.triu(A,1)+(1.0-w)/w*np.diag(np.diag(A))
#     M=np.diag((1.0/w)*np.diag(A))+np.tril(A,-1)
#     B=np.dot(np.linalg.inv(M),N)
#     (valp,vectp)=np.linalg.eig(B)
#     rayonSpect.append(max(abs(valp)))
    

# plt.figure()
# plt.plot(tabW,tabIteration, label='w to reach the convergence or reach Nmax')
# plt.figure()
# plt.plot(tabW,rayonSpect, label='radius spectral ')
# plt.legend()
# plt.show()

# N=-np.tril(A,-1)-np.triu(A,1) #we define the matrix N (-E-F) -E:lower matrix -F upper matrix
# Dinv=np.diag(1.0/np.diag(A))
# Bj=np.dot(Dinv,N)
# (valpBj,vectpBj)=np.linalg.eig(Bj)
# rhoBj=max(np.abs(valpBj))
# wopt=2/(1+np.sqrt(1-pow(rhoBj,2)))
# print("la valeur optimale de w est : ", wopt)

# #Questions 6 & 7
# print('Question 6 & 7')
# iterations=[]
# valw=[]
# rayonspec=[]
# for k in range(1,200):
#     w=1.0/100.0*k
#     x0=np.matrix(np.zeros((n,1)))
#     (x_SOR,k_SOR,erreur_SOR)=SOR(A,b,x0,w,1e-10,1000)
#     valw.append(w)
#     iterations.append(k_SOR)
#     #Spectral radius calculation depending on w
#     N=np.matrix(-np.triu(A,1)+(1.0-w)/w*np.diag(np.diag(A)))
#     M=np.matrix(np.diag((1.0/w)*np.diag(A))+np.tril(A,-1))
#     B=np.linalg.inv(M)*N
#     (valp,vectp)=np.linalg.eig(B)
#     rayonspec.append(max(abs(valp)))
    


# N=np.matrix(-np.triu(A,1)-np.tril(A,-1))
# invM=np.matrix(np.diag(1.0/np.diag(A)))
# B=invM*N
# (valp,vectp)=np.linalg.eig(B)
# rhoBJ=max(abs(valp))
# wopt=2.0/(1.0+np.sqrt(1.0-rhoBJ*rhoBJ))
# print('wopt=',wopt)
# plt.figure()
# plt.plot(valw,iterations)
# plt.legend(['Convergence reached'])
# plt.title('Number of required iterations for convergence depending on w values')
# plt.xlabel('w values')
# plt.ylabel('Numer of iterations')
# ind=np.argmin(iterations) #Calculus of the index for the min number of iterations
# print('La valeur optimale de w parmi les valeurs choisies vaut w=',valw[ind])
# plt.figure()
# plt.plot(valw,rayonspec)
# plt.legend(['Spectral radius'])
# plt.title('Spectral radius of L_w depending on w values')
# plt.xlabel('w values')
# plt.ylabel('Spectral radius values')
# plt.show()

