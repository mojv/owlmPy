import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from mpl_toolkits.mplot3d import Axes3D

def freq_plot(sim_name, moving_avg, lines):
    file = open(sim_name + '.obj', 'rb')
    data = pickle.load(file)

    flux_freqs = np.asarray(data['flux_freqs'])
    disk_refl_flux = np.asarray(data['disk_refl_flux'])
    disk_tran_flux = np.asarray(data['disk_tran_flux'])
    straight_tran_flux = np.asarray(data['straight_tran_flux'])

    wl = []
    Rs = []
    Ts = []
    St = []

    wl = moving_average(np.append(wl, 1 / flux_freqs), n=moving_avg)
    Rs = moving_average(np.append(Rs, -disk_refl_flux / straight_tran_flux), n=moving_avg)
    Ts = moving_average(np.append(Ts, disk_tran_flux / straight_tran_flux), n=moving_avg)
    abs = 1 - Rs - Ts
    St = moving_average(np.append(St, straight_tran_flux), n=moving_avg)

    # limit = np.where(np.flip(abs) < -0.15)[0][0]
    # wl = np.delete(np.flip(wl), np.arange(limit, wl.size, 1))
    # Rs = np.delete(np.flip(Rs), np.arange(limit, Rs.size, 1))
    # Ts = np.delete(np.flip(Ts), np.arange(limit, Ts.size, 1))
    # abs = np.delete(np.flip(abs), np.arange(limit, abs.size, 1))

    plt.figure()
    if 'pulse' in lines:
        plt.plot(wl, St, 'bo-', label='pulso')

    if 'refl' in lines:
        plt.plot(wl, Rs, 'bo-', label='reflectance')

    if 'trans' in lines:
        plt.plot(wl, Ts, 'ro-', label='transmittance')

    if 'abs' in lines:
        plt.plot(wl, abs, 'go-', label='Absorptance')

    plt.xlabel("wavelength (μm)")
    plt.legend(loc="upper right")
    plt.show()

def load_folder(folder):
    files_names = np.asarray(os.listdir(folder))
    files_names = np.sort(files_names)
    second_param = []

    file = open(folder + files_names[0], 'rb')
    data = pickle.load(file)
    flux = np.asarray(data['flux_freqs'])

    flux_freqs = np.zeros((files_names.size, flux.size))
    disk_refl_flux = np.zeros((files_names.size, flux.size))
    disk_tran_flux = np.zeros((files_names.size, flux.size))
    straight_tran_flux = np.zeros((files_names.size, flux.size))

    for i in range(files_names.size):
        if files_names[i].endswith(".obj") or files_names[i].endswith(".py"):
            file = open(folder + files_names[i], 'rb')
            data = pickle.load(file)
            flux_freqs[i] = np.asarray(data['flux_freqs'])
            disk_refl_flux[i] = np.asarray(data['disk_refl_flux'])
            disk_tran_flux[i] = np.asarray(data['disk_tran_flux'])
            straight_tran_flux[i] = np.asarray(data['straight_tran_flux'])
            second_param = np.append(second_param, float(files_names[i].split('.obj')[0].split('_')[2]) * 1000)
        else:
            continue

    return second_param, flux_freqs, disk_refl_flux, disk_tran_flux, straight_tran_flux


def heat_map(folder, type):

    second_param, flux_freqs, disk_refl_flux, disk_tran_flux, straight_tran_flux = load_folder(folder)

    # Make data.
    Y = second_param
    X = (1/flux_freqs[0]) * 1000
    X, Y = np.meshgrid(X, Y)
    R = -disk_refl_flux/ straight_tran_flux
    T = disk_tran_flux / straight_tran_flux
    A = 1 - R - T

    if type=='3D':
        fig = plt.figure()
        ax = Axes3D(fig) #<-- Note the difference from your original code...
        cset1 = ax.contour(X, Y, R, 100, extend3d=False)
        cset2 = ax.contour(X, Y, T, 100, extend3d=True)
        # cset3 = ax.contour(X, Y, A, 100, extend3d=True)
        ax.clabel(cset1, fontsize=9, inline=0)
        ax.clabel(cset2, fontsize=9, inline=1)
        # ax.clabel(cset3, fontsize=9, inline=1)
    elif type=='heat_map':
        plt.pcolormesh(X, Y, T)
        plt.xlabel("wavelength (nm)")
        plt.ylabel("cell size (nm)")
        plt.colorbar()
        plt.show()

def param_plot(folder, lambd, lines, label):

    second_param, flux_freqs, disk_refl_flux, disk_tran_flux, straight_tran_flux = load_folder(folder)

    freq_id = np.where(flux_freqs[0] >= 1/lambd)[0][0]

    Rs = []
    Ts = []

    Rs = np.append(Rs, -disk_refl_flux[:, freq_id] / straight_tran_flux[:, freq_id])
    Ts = np.append(Ts, disk_tran_flux[:, freq_id] / straight_tran_flux[:, freq_id])

    plt.figure()
    if 'refl' in lines:
        plt.plot(second_param, Rs, 'bo-', label='reflectance')

    if 'trans' in lines:
        plt.plot(second_param, Ts, 'ro-', label='transmittance')

    if 'abs' in lines:
        plt.plot(second_param, 1 - Rs - Ts, 'go-', label='Absorptance')

    plt.xlabel(label)
    plt.legend(loc="upper right")
    plt.show()


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
