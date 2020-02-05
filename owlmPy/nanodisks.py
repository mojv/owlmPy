import meep as mp
import math
import cmath
import numpy as np
import pickle
import owlmPy.gyrotropic_materials as gyr


class nanodisk:
    # The default unit length is 1 um
    um_scale = 1
    nfreq = 500

    def __init__(self, theta, resolution, heights_vector, materials_vector, geometries_vector,
                 characteristic_vector, abs_layer, air_layer, lmin, lmax, magnetic_field=mp.Vector3(0, 0, 0)):

        self.sz = abs_layer + air_layer + np.sum(heights_vector) + abs_layer
        heights_vector[-1] = heights_vector[
                                 -1] + abs_layer  # The las item is  substrate and will be extended to the abs layer
        materials_position = np.zeros(heights_vector.size)

        for i in range(heights_vector.size):
            materials_position[i] = 0.5 * self.sz - abs_layer - air_layer - np.sum(heights_vector[0:i]) - 0.5 * \
                                    heights_vector[i]

        # -----------------------------------------------------------------------------------------------------
        # ------------------------------- Simulation Variables -------------------------------------------------
        # ----------------------------------------------------------------------------------------------
        self.z_flux_trans = -0.5 * self.sz + abs_layer
        self.z_flux_refl = 0.5 * self.sz - abs_layer
        self.pt = mp.Vector3(0, 0, self.z_flux_refl)  # Field Decay point`

        self.src_pos = 0.5 * self.sz - abs_layer - 0.1 * air_layer

        # -----------------------------------------------------------------------------------------------
        # ------------------------------- Geometry Setup -------------------------------------------------
        # ----------------------------------------------------------------------------------------------

        self.pml_layers = [mp.PML(thickness=abs_layer, direction=mp.Z, side=mp.High),
                           mp.PML(thickness=abs_layer, direction=mp.Z, side=mp.Low)]

        self.geometry = []

        for i in range(materials_position.size):
            mat = gyr.gyrotropic_conversion(materials_vector[i], magnetic_field)
            if geometries_vector[i] == 'cylinder':
                self.geometry.append(
                    mp.Cylinder(material=mat, radius=characteristic_vector[i], height=heights_vector[i],
                                center=mp.Vector3(0, 0, materials_position[i])))
            elif geometries_vector[i] == 'block':
                self.geometry.append(mp.Block(material=mat,
                                              size=mp.Vector3(characteristic_vector[i][0], characteristic_vector[i][1],
                                                              heights_vector[i]),
                                              center=mp.Vector3(0, 0, materials_position[i])))

        susc = [mp.GyrotropicLorentzianSusceptibility(bias=magnetic_field)]
        self.default_material = mp.Medium(epsilon=1, mu=1, E_susceptibilities=susc)
        # -----------------------------------------------------------------------------------------------
        # ------------------------------- Source Setup -------------------------------------------------
        # ----------------------------------------------------------------------------------------------
        fmin = self.um_scale / lmax  # source min frequency
        fmax = self.um_scale / lmin  # source max frequency
        self.fcen = 0.5 * (fmin + fmax)
        self.df = fmax - fmin

        # CCW rotation angle (degrees) about Y-axis of PW current source
        self.k = mp.Vector3(math.sin(math.radians(theta)), 0, math.cos(math.radians(theta))).scale(self.fcen)

        if theta == 0:
            k = mp.Vector3(0, 0, 0)

        self.resolution = resolution
        self.store = {}

    def set_source(self, sx, sy):
        sources = [mp.Source(mp.GaussianSource(self.fcen, fwidth=self.df), component=mp.Hx,
                             center=mp.Vector3(0, 0, self.src_pos),
                             size=mp.Vector3(sx, sy, 0),
                             amp_func=pw_amp(self.k, mp.Vector3(0, 0, self.src_pos)))]

        refl_fr = mp.FluxRegion(center=mp.Vector3(0, 0, self.z_flux_refl), size=mp.Vector3(sx, sy, 0))
        trans_fr = mp.FluxRegion(center=mp.Vector3(0, 0, self.z_flux_trans), size=mp.Vector3(sx, sy, 0))

        return sources, refl_fr, trans_fr

    def empty_run(self, sx, sy, animate=False):

        sources, refl_fr, trans_fr = self.set_source(sx, sy)

        sim = mp.Simulation(cell_size=mp.Vector3(sx, sy, self.sz),
                            geometry=[],
                            sources=sources,
                            boundary_layers=self.pml_layers,
                            k_point=self.k,
                            resolution=self.resolution,
                            default_material=self.default_material)

        refl = sim.add_flux(self.fcen, self.df, self.nfreq, refl_fr)

        trans = sim.add_flux(self.fcen, self.df, self.nfreq, trans_fr)

        if animate:
            sim.run(mp.at_beginning(mp.output_epsilon),
                    mp.to_appended("ex", mp.at_every(0.6, mp.output_efield_z)),
                    until_after_sources=mp.stop_when_fields_decayed(25, mp.Ey, self.pt, 1e-3))
        else:
            sim.run(until_after_sources=mp.stop_when_fields_decayed(25, mp.Ey, self.pt, 1e-3))

        # for normalization run, save flux fields data for reflection plane
        self.store['straight_refl_data'] = sim.get_flux_data(refl)

        # save incident power for transmission plane
        self.store['flux_freqs'] = mp.get_flux_freqs(refl)
        self.store['straight_tran_flux'] = mp.get_fluxes(trans)
        self.store['straight_refl_flux'] = mp.get_fluxes(refl)

    def disk_run(self, sx, sy):

        sources, refl_fr, trans_fr = self.set_source(sx, sy)

        sim = mp.Simulation(cell_size=mp.Vector3(sx, sy, self.sz),
                            geometry=self.geometry,
                            sources=sources,
                            boundary_layers=self.pml_layers,
                            k_point=self.k,
                            resolution=self.resolution,
                            default_material=self.default_material)

        refl = sim.add_flux(self.fcen, self.df, self.nfreq, refl_fr)

        trans = sim.add_flux(self.fcen, self.df, self.nfreq, trans_fr)

        # for normal run, load negated fields to subtract incident from refl. fields
        #sim.load_minus_flux_data(refl, self.store['straight_refl_data'])

        sim.run(until_after_sources=mp.stop_when_fields_decayed(25, mp.Ey, self.pt, 1e-3))

        self.store['disk_refl_flux'] = mp.get_fluxes(refl)
        self.store['disk_tran_flux'] = mp.get_fluxes(trans)

    def save_obj(self, fname):
        file = open(fname + '.obj', 'wb')
        pickle.dump(self.store, file)


def pw_amp(k, x0):
    def _pw_amp(x):
        return cmath.exp(1j * 2 * math.pi * k.dot(x + x0))

    return _pw_amp
