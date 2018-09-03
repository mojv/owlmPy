# -*- coding: utf-8 -*-
import numpy as np


def tm(index,ds,theta,lambd, polarization):
    
    N=index.shape[0]-2
    #check step 01 of owlmp_illustration.ipynb
    ks = index*2*np.pi/lambd
    
    #calculation of thetha en each layer (snell's law) (check step 02 of owlmp_illustration.ipynb)
    thetas=np.zeros(index.shape[0])+0j
    thetas[0]=theta*np.pi/180
    for i in range(N+1):
        thetas[i+1] = np.arcsin(np.sin(thetas[i])*index[i]/index[i+1])
        
    #calculation of phase shifts associated with propagation in the middle regions (check step 03 of owlmp_illustration.ipynb)
    phis = ks*ds*np.cos(thetas)
    
    if polarization == "s":
        #check step 04 of owlmp_illustration.ipynb
        ones=np.ones(N+2)
        nc=index*np.cos(thetas)
        ephi=np.exp(1j*phis)
        n_ephi=np.exp(-1j*phis)

        M1=np.reshape(np.array([ones,ones,nc,-nc]).T,[N+2,2,2])
        M2=np.linalg.inv(np.reshape(np.array([ephi,n_ephi,nc*ephi,-nc*n_ephi]).T,[N+2,2,2]))
        M=np.matmul(M1, M2)

        # check step 05 of owlmp_illustration.ipynb
        prod=np.linalg.multi_dot(M[1:N+1])

        # check step 06 of owlmp_illustration.ipynb
        A1=np.linalg.inv([[1,1],[index[0]*np.cos(thetas[0]),-index[0]*np.cos(thetas[0])]])
        A2=np.array([[1,1],[index[N+1]*np.cos(thetas[N+1]),-index[N+1]*np.cos(thetas[N+1])]])
        A=A1@prod@A2
    else:
        # check step 08 of owlmp_illustration.ipynb
        cos=np.cos(thetas)
        ephi=np.exp(1j*phis)
        n_ephi=np.exp(-1j*phis)

        M1=np.reshape(np.array([cos,cos,index,-index]).T,[N+2,2,2])
        M2=np.linalg.inv(np.reshape(np.array([cos*ephi,cos*n_ephi,index*ephi,-index*n_ephi]).T,[N+2,2,2]))
        M=np.matmul(M1, M2)

        # check step 09 of owlmp_illustration.ipynb
        prod=np.linalg.multi_dot(M[1:N+1])

        # check step 10 of owlmp_illustration.ipynb
        A1=np.linalg.inv([[np.cos(thetas[0]),np.cos(thetas[0])],[index[0], -index[0]]])
        A2=np.array([[np.cos(thetas[N+1]),np.cos(thetas[N+1])],[index[N+1],-index[N+1]]])
        A = A1@prod@A2

    # check step 07 of owlmp_illustration.ipynb
    t_tot=1/A[0,0]
    r_tot=A[1,0]/A[0,0]
    return {'t': t_tot, 'r': r_tot, 'T': np.absolute(t_tot)**2, 'R': np.absolute(r_tot)**2, 'A': 1-np.absolute(t_tot)**2-np.absolute(r_tot)**2}      