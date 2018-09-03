# -*- coding: utf-8 -*-
"""
doc
"""

import numpy as np


def tm(index,ds,theta,lambd, polarization):
    
    N=index.shape[0]-2
    #set-up wave vector
    ks = index*2*np.pi/lambd
    
    #calculation of thetha en each layer (snell's law)
    thetas=np.zeros(index.shape[0])+0j
    thetas[0]=theta*np.pi/180
    for i in range(N+1):
        thetas[i+1] = np.arcsin(np.sin(thetas[i])*index[i]/index[i+1])
        
    #calculation of phase shifts associated with propagation in the middle regions  
    phis = ks*ds*np.cos(thetas)
    
    if polarization == "s":
        ones=np.ones(N+2)
        nc=index*np.cos(thetas)
        ephi=np.exp(1j*phis)
        n_ephi=np.exp(-1j*phis)

        M1=np.reshape(np.array([ones,ones,nc,-nc]).T,[N+2,2,2])
        M2=np.linalg.inv(np.reshape(np.array([ephi,n_ephi,nc*ephi,-nc*n_ephi]).T,[N+2,2,2]))
        M=np.matmul(M1, M2)
        
        prod=np.linalg.multi_dot(M[1:N+1])
        
        A1=np.linalg.inv([[1,1],[index[0]*np.cos(thetas[0]),-index[0]*np.cos(thetas[0])]])
        A2=np.array([[1,1],[index[N+1]*np.cos(thetas[N+1]),-index[N+1]*np.cos(thetas[N+1])]])
        A=A1@prod@A2
    else:
        cos=np.cos(thetas)
        ephi=np.exp(1j*phis)
        n_ephi=np.exp(-1j*phis)

        M1=np.reshape(np.array([cos,cos,index,-index]).T,[N+2,2,2])
        M2=np.linalg.inv(np.reshape(np.array([cos*ephi,cos*n_ephi,index*ephi,-index*n_ephi]).T,[N+2,2,2]))
        M=np.matmul(M1, M2)
        
        prod=np.linalg.multi_dot(M[1:N+1])
        
        A1=np.linalg.inv([[np.cos(thetas[0]),np.cos(thetas[0])],[index[0], -index[0]]])
        A2=np.array([[np.cos(thetas[N+1]),np.cos(thetas[N+1])],[index[N+1],-index[N+1]]])
        A = A1@prod@A2
        
    t_tot=1/A[0,0]
    r_tot=A[1,0]/A[0,0]
    return {'t': t_tot, 'r': r_tot, 'T': np.absolute(t_tot)**2, 'R': np.absolute(r_tot)**2, 'A': 1-np.absolute(t_tot)**2-np.absolute(r_tot)**2}      