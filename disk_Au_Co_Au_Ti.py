import numpy as np
from owlmPy import nanodisks as nd
import meep.materials as mt
from meep import inf
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

    heights_vector = np.array([0.006, 0.010, 0.016, 0.002, 0.033])
    materials_vector = [mt.Au, mt.Co, mt.Au, mt.Ti, mt.BK7]
    geometries_vector = ['cylinder', 'cylinder', 'cylinder', 'cylinder', 'block']
    characteristic_vector = [
        disk_radius,  # For first Au cylinder it is specified the radius
        disk_radius,  # For Co cylinder it is specified the radius
        disk_radius,  # For 2nd Au cylinder it is specified the radius
        disk_radius,  # For Ti cylinder it is specified the radius
        [inf, inf]    # For BK7 substrate block it is specified its width and length
    ]

    disk = nd.nanodisk(theta, resolution, heights_vector, materials_vector, geometries_vector, characteristic_vector, abs_layer, air_layer, lmin, lmax)

    jobs = []

    for dis in np.arange(0.54, 0.61, 0.01):
        s = np.round(dis, 2)
        print(s)
        # disk.empty_run(s, s)
        # disk.disk_run(s, s)
        # disk.save_obj('disk_lati_' + str(s))
        p = ps(target=async_func, args=(disk, s, 'disk_lati_' + str(s)), )
        jobs.append(p)
        p.start()


