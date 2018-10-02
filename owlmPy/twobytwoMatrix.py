# -*- coding: utf-8 -*-
import numpy as np
import numba as nb

def twobytwo(index,ds,theta,lambd, polarization):

    N=index.shape[0]-2
    #check step 01 of towbytwo_illustration.ipynb
    ks = index*2*np.pi/lambd

    #calculation of thetha en each layer (snell's law) (check step 02 of towbytwo_illustration.ipynb)
    thetas=np.zeros(index.shape[0])+0j
    thetas[0]=theta*np.pi/180
    for i in range(N+1):
        thetas[i+1] = np.arcsin(np.sin(thetas[i])*index[i]/index[i+1])

    #calculation of phase shifts associated with propagation in the middle regions (check step 03 of towbytwo_illustration.ipynb)
    phis = ks*ds*np.cos(thetas)

    if polarization == "s":
        #check step 04 of towbytwo_illustration.ipynb
        ones=np.ones(N+2)
        nc=index*np.cos(thetas)
        ephi=np.exp(1j*phis)
        n_ephi=np.exp(-1j*phis)

        M1=np.reshape(np.array([ones,ones,nc,-nc]).T,[N+2,2,2])
        M2=np.linalg.inv(np.reshape(np.array([ephi,n_ephi,nc*ephi,-nc*n_ephi]).T,[N+2,2,2]))
        M=np.matmul(M1, M2)

        # check step 05 of towbytwo_illustration.ipynb
        prod=np.linalg.multi_dot(M[1:N+1])

        # check step 06 of towbytwo_illustration.ipynb
        A1=np.linalg.inv([[1,1],[index[0]*np.cos(thetas[0]),-index[0]*np.cos(thetas[0])]])
        A2=np.array([[1,1],[index[N+1]*np.cos(thetas[N+1]),-index[N+1]*np.cos(thetas[N+1])]])
        A=A1@prod@A2
    else:
        # check step 08 of towbytwo_illustration.ipynb
        cos=np.cos(thetas)
        ephi=np.exp(1j*phis)
        n_ephi=np.exp(-1j*phis)

        M1=np.reshape(np.array([cos,cos,index,-index]).T,[N+2,2,2])
        M2=np.linalg.inv(np.reshape(np.array([cos*ephi,cos*n_ephi,index*ephi,-index*n_ephi]).T,[N+2,2,2]))
        M=np.matmul(M1, M2)

        # check step 09 of towbytwo_illustration.ipynb
        prod=np.linalg.multi_dot(M[1:N+1])

        # check step 10 of towbytwo_illustration.ipynb
        A1=np.linalg.inv([[np.cos(thetas[0]),np.cos(thetas[0])],[index[0], -index[0]]])
        A2=np.array([[np.cos(thetas[N+1]),np.cos(thetas[N+1])],[index[N+1],-index[N+1]]])
        A = A1@prod@A2

    # check step 07 of towbytwo_illustration.ipynb
    t_tot=1/A[0,0]
    r_tot=A[1,0]/A[0,0]
    return {'t': t_tot, 'r': r_tot, 'T': np.absolute(t_tot)**2, 'R': np.absolute(r_tot)**2, 'A': 1-np.absolute(t_tot)**2-np.absolute(r_tot)**2}


@nb.jit(nopython=True)
def numba_prod(matrix):
    prod = np.zeros((matrix.shape[0], 2, 2)) + 0j
    for i in nb.prange(matrix.shape[0]):
        prod[i] = np.identity(2)
        for ii in nb.prange(matrix.shape[1]):
            temp = matrix[i, ii + 1, :, :]
            prod[i] = np.dot(prod[i], temp)

    return prod

def twobytwo_thetas(index, ds, theta, lambd, polarization):
    N = index.shape[0] - 2
    # check step 01 of towbytwo_thetas_illustration.ipynb
    conv = theta.shape[0]
    ks = index * 2 * np.pi / lambd

    # calculation of thetha en each layer (snell's law) (check step 02 of towbytwo_thetas__illustration.ipynb)
    theta = theta * np.pi / 180
    thetas = np.zeros((conv, N + 2)) + 0j  # defining thetas vector (see figure below)
    thetas[:, 0] = theta
    for i in range(N + 1):
        thetas[:, i + 1] = np.arcsin(np.sin(thetas[:, i]) * index[i] / index[i + 1])

    # calculation of phase shifts associated with propagation in the middle regions (check step 03 of towbytwo_thetas_illustration.ipynb)
    phis = np.multiply(ks * ds, np.cos(thetas))

    if polarization == "s":
        # check step 04 of towbytwo_thetas_illustration.ipynb
        n_cos = np.multiply(index, np.cos(thetas))
        e_phis = np.exp(1j * phis)
        e_phis_n = np.exp(-1j * phis)

        M1 = np.ones((conv, N + 2, 2, 2)) + 0j
        M1[:, :, 1, 0] = n_cos
        M1[:, :, 1, 1] = -n_cos

        M2 = np.ones((conv, N + 2, 2, 2)) + 0j
        M2[:, :, 0, 0] = e_phis
        M2[:, :, 0, 1] = e_phis_n
        M2[:, :, 1, 0] = n_cos * e_phis
        M2[:, :, 1, 1] = -n_cos * e_phis_n
        M2 = np.linalg.inv(M2)
        M = np.matmul(M1, M2)

        # check step 05 of towbytwo_thetas_illustration.ipynb
        prod = numba_prod(M)

        # check step 06 of towbytwo_thetas_illustration.ipynb
        A1 = np.ones((conv, 2, 2)) + 0j
        A1[:, 1, 0] = n_cos[:, 0]
        A1[:, 1, 1] = -n_cos[:, 0]
        A1 = np.linalg.inv(A1)

        A2 = np.ones((conv, 2, 2)) + 0j
        A2[:, 1, 0] = n_cos[:, N + 1]
        A2[:, 1, 1] = -n_cos[:, N + 1]

        A = A1 @ prod @ A2
    else:
        # check step 08 of towbytwo_thetas_illustration.ipynb
        cos = np.cos(thetas)
        e_phis = np.exp(1j * phis)
        e_phis_n = np.exp(-1j * phis)

        M1 = np.ones((conv, N + 2, 2, 2)) + 0j
        M1[:, :, 0, 0] = cos
        M1[:, :, 0, 1] = cos
        M1[:, :, 1, 0] = index
        M1[:, :, 1, 1] = -index

        M2 = np.ones((conv, N + 2, 2, 2)) + 0j
        M2[:, :, 0, 0] = cos * e_phis
        M2[:, :, 0, 1] = cos * e_phis_n
        M2[:, :, 1, 0] = index * e_phis
        M2[:, :, 1, 1] = -index * e_phis_n
        M2 = np.linalg.inv(M2)
        M = np.matmul(M1, M2)

        # check step 09 of towbytwo_thetas_illustration.ipynb
        prod = numba_prod(M)

        # check step 10 of towbytwo_thetas_illustration.ipynb
        A1 = np.ones((conv, 2, 2)) + 0j
        A1[:, 0, 0] = cos[:, 0]
        A1[:, 0, 1] = cos[:, 0]
        A1[:, 1, 0] = index[0]
        A1[:, 1, 1] = -index[0]
        A1 = np.linalg.inv(A1)

        A2 = np.ones((conv, 2, 2)) + 0j
        A2[:, 0, 0] = cos[:, N + 1]
        A2[:, 0, 1] = cos[:, N + 1]
        A2[:, 1, 0] = index[N + 1]
        A2[:, 1, 1] = -index[N + 1]

        A = A1 @ prod @ A2

    # check step 07 of towbytwo_thetas_illustration.ipynb
    t_tot = 1 / A[:, 0, 0]
    r_tot = A[:, 1, 0] / A[:, 0, 0]
    t_tot[-1] = t_tot[-2]
    r_tot[-1] = r_tot[-2]
    return {'t': t_tot, 'r': r_tot, 'T': np.absolute(t_tot) ** 2, 'R': np.absolute(r_tot) ** 2,
            'A': 1 - np.absolute(t_tot) ** 2 - np.absolute(r_tot) ** 2}