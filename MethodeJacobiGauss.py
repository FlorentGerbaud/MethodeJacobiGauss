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
    Dinv=np.diag(1.0/np.diag(Ac)) #we define Dinv to calculate x at the k-eme etape
    k=0
    residu=[]
    residu.append(np.linalg.norm(np.dot(Ac,x)-b)/np.linalg.norm(b))
    while(np.linalg.norm(np.dot(Ac,x)-b)/np.linalg.norm(b)>eps and k<Nmax): #conditions to stop the prog
        x=np.dot(Dinv,(np.dot(N,x)+b)) #x to the k-ieme etape
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

# x1,k1,residues1=jacobi(C,b,x0,10^-8,300)
# print(x1,k1,residues1)

############################# Gauss Seidel ####################################  

# x2,k2,residues2=gaussSeidel(A,b,x0,1e-8,3000)
# print(k2)
# print(residues2[-1]>10^-8)

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

tour=400
tabIteration=[]
tabW=[]
rayonSpect=[]

iteration=0
w=0
for i in range(1,tour):
    w=i*(1/200.0)
    x3,iteration,residues3=SOR(A,b,x0,1e-6,1000,w) #move the number of iteration max to have the more precision as possible and find the w
    tabIteration.append(iteration)
    tabW.append(w)
    # radius spectral 
    
    N=-np.triu(A,1)+(1.0-w)/w*np.diag(np.diag(A))
    M=np.diag((1.0/w)*np.diag(A))+np.tril(A,-1)
    B=np.dot(np.linalg.inv(M),N)
    (valp,vectp)=np.linalg.eig(B)
    rayonSpect.append(max(abs(valp)))
    

plt.figure()
plt.plot(tabW,tabIteration, label='w to reach the convergence or reach Nmax')
plt.figure()
plt.plot(tabW,rayonSpect, label='radius spectral ')
plt.legend()
plt.show()

N=-np.tril(A,-1)-np.triu(A,1) #we define the matrix N (-E-F) -E:lower matrix -F upper matrix
Dinv=np.diag(1.0/np.diag(A))
Bj=np.dot(Dinv,N)
(valpBj,vectpBj)=np.linalg.eig(Bj)
rhoBj=max(np.abs(valpBj))
wopt=2/(1+np.sqrt(1-pow(rhoBj,2)))
print("la valeur optimale de w est : ", wopt)

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
    # N=np.matrix(-np.triu(A,1)+(1.0-w)/w*np.diag(np.diag(A)))
    # M=np.matrix(np.diag((1.0/w)*np.diag(A))+np.tril(A,-1))
    # B=np.linalg.inv(M)*N
    # (valp,vectp)=np.linalg.eig(B)
    # rayonspec.append(max(abs(valp)))
    


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

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Tue Mar 21 11:18:40 2023

# @author: boulbe
# """

# import numpy as np
# from numpy import *
# from matplotlib import pyplot as plt

# #Question 1
# print('Question 1')
# def Jacobi(A,b,x0,eps,Nmax):
#     k=0
#     x=x0
#     N=-triu(A,1)-tril(A,-1)
#     invM=diag(1.0/diag(A))
#     normb=linalg.norm(b)
#     residu=[]
#     residu.append(linalg.norm(A*x-b)/normb)
#     while(linalg.norm(A*x-b)/normb>eps and k<Nmax):
#         x=invM*(N*x+b)
#         residu.append(linalg.norm(A*x-b)/normb)
#         k=k+1
        
#     return (x,k,residu)

# def Descente(L,b):
#     N=L.shape
#     x=matrix(zeros((N[0],1)))
#     x[0]=b[0]/L[0,0]
    
#     for i in range(1,N[0]):
#         # x[i]=(b[i]-sum(L[i,0:i]*x[0:i]))/L[i,i]
#         x[i]=(b[i]-L[i,0:i]*x[0:i])/L[i,i]
#     return x

# def GaussSeidel(A,b,x0,eps,Nmax):
#     k=0
#     x=x0
#     N=-triu(A,1)
#     M=tril(A)
#     normb=linalg.norm(b)
#     residu=[]
#     residu.append(linalg.norm(A*x-b)/normb)
#     while(linalg.norm(A*x-b)/normb>eps and k<Nmax):
#         x=Descente(M,(N*x+b))  #M=D-E triangular inferior
#         residu.append(linalg.norm(A*x-b)/normb)
#         k=k+1
    
#     return (x,k,residu)


# #Questions 2 & 3
# print('Question 2 et 3')
    
# n=10
# c=ones(9)
# A=matrix(2*eye(n,n)-diag(c,1)-diag(c,-1))
# b=matrix(ones((n,1)))
# x0=matrix(zeros((n,1)))
# (x,k,erreur_J)=Jacobi(A,b,x0,1e-7,1000)
# (x_GS,k_GS,erreur_GS)=GaussSeidel(A,b,x0,1e-7,10000)
# plt.figure()
# plt.yscale('log')
# plt.plot(arange(0,k+1),erreur_J,arange(0,k_GS+1),erreur_GS)
# plt.legend(['Jacobi','Gauss-Seidel'])
# plt.title('Convergence rates comparison')
# plt.xlabel('Number of iterations')
# plt.ylabel('Precision')


# #Question 4
# print('Question 4')
# #Matrix B
# Bexo = array([[1., 0.75, 0.75], [0.75, 1., 0.75], [0.75, 0.75, 1.]])
# eigen_val_B, eigen_vectors_B = linalg.eig(Bexo)
# print(eigen_val_B, 'Eigen values of B')
# print(max(abs(eigen_val_B)), 'Spectral radius of B')
# mat_iter_B_Jacobi = array([[0., -0.75, -0.75], [-0.75, 0., -0.75], [-0.75, -0.75, 0.]])
# eigen_val_mat_iter_B, eigen_vectors_mat_iter_B = linalg.eig(mat_iter_B_Jacobi)
# print(max(abs(eigen_val_mat_iter_B)), 'Spectral radius of Jacobi iter matrix of B')
# mat_iter_B_GS = dot(linalg.inv(array([[1., 0., 0.], [0.75, 1., 0.], [0.75, 0.75, 1.]])),array([[0., -0.75, -0.75], [0., 0., -0.75], [0., 0., 0.]]))
# eigen_val_mat_iter_B_GS, eigen_vectors_mat_iter_B_GS = linalg.eig(mat_iter_B_GS)
# print(max(abs(eigen_val_mat_iter_B_GS)), 'Spectral radius of Gauss Seidel iter matrix of B')
# print('Jacobi does not converge, Gauss Seidel does (remark: matrix sym. def. pos. so makes sense)')

# x4=matrix(zeros((3,1)))
# b4=matrix(ones((3,1)))
# (x,k,erreur_J)=GaussSeidel(Bexo,b4,x4,1e-7,1000)
# print('La méthode de Gauss Seidel a convergé en',k,' iterations')
# (x,k,erreur_J)=Jacobi(Bexo,b4,x4,1e-7,1000)
# print('La méthode diverge -  Nb iterations effectués:',k,' iterations')


# #Matrix C
# Cexo = array([[1, 2, -2], [1, 1, 1], [2, 2, 1]])
# eigen_val_C, eigen_vectors_C = linalg.eig(Cexo)
# print(eigen_val_C, 'Eigen values of C')
# print(max(abs(eigen_val_C)), 'Spectral radius of C')
# mat_iter_C = array([[0., -2., 2.], [-1., 0., -1.], [-2., -2., 0.]])
# eigen_val_mat_iter_C, eigen_vectors_mat_iter_C = linalg.eig(mat_iter_C)
# print(max(abs(eigen_val_mat_iter_C)), 'Spectral radius of Jacobi iter matrix C')
# mat_iter_C_GS = dot(linalg.inv(array([[1., 0., 0.], [1., 1., 0.], [2., 2., 1.]])),array([[0., -2., 2.], [0., 0., -1.], [0., 0., 0.]]))
# eigen_val_mat_iter_C_GS, eigen_vectors_mat_iter_C_GS = linalg.eig(mat_iter_C_GS)
# print(max(abs(eigen_val_mat_iter_C_GS)), 'Spectral radius of Gauss Seidel iter matrix of C')
# print('Jacobi converges, Gauss Seidel does not')




# #Question 5
# print('Question 5')

# def SOR(A,b,x0,w,eps,Nmax):
#     k=0
#     x=x0
#     N=matrix(-triu(A,1)+(1.0-w)/w*matrix(diag(diag(A))))
#     M=matrix(diag((1.0/w)*diag(A))+tril(A,-1))
#     normb=linalg.norm(b)
#     residu=[]
#     residu.append(linalg.norm(A*x-b)/normb)
#     while(linalg.norm(A*x-b)/normb>eps and k<Nmax):
#         x=Descente(M,(N*x+b))
#         residu.append(linalg.norm(A*x-b)/normb)
#         k=k+1
#     return (x,k,residu)

# (x_SOR,k_SOR,erreur_SOR)=SOR(A,b,x0,0.33,1e-7,10000)  #play with w values, what do you notice?
# #print(erreur_SOR)
# plt.figure()
# plt.yscale('log')
# plt.plot(arange(0,k_SOR+1),erreur_SOR)
# plt.legend(['SOR'])
# plt.title('Convergence rate of SOR method')
# plt.xlabel('Number of iterations')
# plt.ylabel('Precision')
# # plt.show()


# #Questions 6 & 7
# print('Question 6 & 7')
# iterations=[]
# valw=[]
# rayonspec=[]
# for k in range(1,200):
#     w=1.0/100.0*k
#     x0=matrix(zeros((n,1)))
#     (x_SOR,k_SOR,erreur_SOR)=SOR(A,b,x0,w,1e-10,1000)
#     valw.append(w)
#     iterations.append(k_SOR)
#     #Spectral radius calculation depending on w
#     N=matrix(-triu(A,1)+(1.0-w)/w*diag(diag(A)))
#     M=matrix(diag((1.0/w)*diag(A))+tril(A,-1))
#     B=linalg.inv(M)*N
#     (valp,vectp)=linalg.eig(B)
#     rayonspec.append(max(abs(valp)))
    


# N=matrix(-triu(A,1)-tril(A,-1))
# invM=matrix(diag(1.0/diag(A)))
# B=invM*N
# (valp,vectp)=linalg.eig(B)
# rhoBJ=max(abs(valp))
# wopt=2.0/(1.0+sqrt(1.0-rhoBJ*rhoBJ))
# print('wopt=',wopt)
# plt.figure()
# plt.plot(valw,iterations)
# plt.legend(['Convergence reached'])
# plt.title('Number of required iterations for convergence depending on w values')
# plt.xlabel('w values')
# plt.ylabel('Numer of iterations')
# ind=argmin(iterations) #Calculus of the index for the min number of iterations
# print('La valeur optimale de w parmi les valeurs choisies vaut w=',valw[ind])
# plt.figure()
# plt.plot(valw,rayonspec)
# plt.legend(['Spectral radius'])
# plt.title('Spectral radius of L_w depending on w values')
# plt.xlabel('w values')
# plt.ylabel('Spectral radius values')
# plt.show()