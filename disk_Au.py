import numpy as np
import owlmPy.nanodisks as nd
import meep.materials as mt
from meep import inf

theta = 0
resolution = 100  # pixels/um
sx = 0.3  # lattice periodicity
sy = 0.3  # Incident source inclination
abs_layer = 4  # PML thickness
air_layer = 5  # air thickness
lmin = 0.4  # source min wavelength
lmax = 1  # source max wavelength
disk_radius = 0.88

heights_vector = np.array([0.033, 2])
materials_vector = [mt.Au, mt.BK7]
geometries_vector = ['cylinder', 'block']
characteristic_vector = [
    disk_radius,  # For 2nd Au cylinder it is specified the radius
    [inf, inf]    # For BK7 substrate block it is specified its width and length
]

nd.abs_layer(theta, resolution, sx, sy, heights_vector, materials_vector, geometries_vector, characteristic_vector, abs_layer, air_layer, lmin,lmax,'disk_Au_0')
