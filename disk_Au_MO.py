import numpy as np
from owlmPy import nanodisks as nd
import meep.materials as mt
from meep import inf, Vector3
import meep as mp
import matplotlib.pyplot as plt
from multiprocessing import Process as ps


def async_func(disk, s, name):
    disk.empty_run(s, s)
    disk.disk_run(s, s)
    disk.save_obj(name)


if __name__ == '__main__':
    theta = 0
    resolution = 100  # pixels/um
    sx = 0.55  # lattice periodicity
    sy = 0.55  # Incident source inclination
    abs_layer = 2  # PML thickness
    air_layer = 0.5  # air thickness
    lmin = 0.4  # source min wavelength
    lmax = 1  # source max wavelength
    disk_radius = 0.088

    ## Parameters for a gyrotropic Lorentzian medium
    epsn = 1.5  # background permittivity
    f0 = 1.0  # natural frequency
    gamma = 1e-6  # damping rate
    sn = 0.1  # sigma parameter
    b0 = 0.15  # magnitude of bias vector

    susc = [mp.LorentzianSusceptibility(frequency=f0, gamma=gamma, sigma=sn,
                                        )]
    mt.Au = mp.Medium(epsilon=epsn, mu=1, E_susceptibilities=susc)

    heights_vector = np.array([0.006, 0.010, 0.016, 0.002])
    materials_vector = [mt.Au, mt.Au, mt.Au, mt.Au]
    geometries_vector = ['block', 'block', 'block', 'block', 'block']
    characteristic_vector = [
        [inf, inf],  # For first Au cylinder it is specified the radius
        [inf, inf],  # For Co cylinder it is specified the radius
        [inf, inf],  # For 2nd Au cylinder it is specified the radius
        [inf, inf],  # For Ti cylinder it is specified the radius
        [inf, inf]    # For BK7 substrate block it is specified its width and length
    ]

    disk = nd.nanodisk(theta, resolution, heights_vector, materials_vector, geometries_vector, characteristic_vector, abs_layer, air_layer, lmin, lmax, Vector3(0, 0, 0))

    s = 0
    disk.empty_run(s, s)
    disk.disk_run(s, s)
    disk.save_obj('borrar3')