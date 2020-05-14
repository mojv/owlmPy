import meep as mp
import math
import cmath
import numpy as np
import pickle


def run_nanodisk_simulation(name, empty, sx, sy, theta, resolution, heights_vector, materials_vector, geometries_vector,
        characteristic_vector, abs_layer, air_layer, lmin, lmax, coating_layer=None,
        magnetic_field=mp.Vector3(0, 0, 0)):

    store = {}

    um_scale = 1
    nfreq = 500

    sz = abs_layer + air_layer + np.sum(heights_vector) + abs_layer

    # The las item is the substrate and will be extended to the abs layer
    heights_vector[-1] = heights_vector[-1] + abs_layer
    materials_position = np.zeros(heights_vector.size)

    for i in range(heights_vector.size):
        materials_position[i] = 0.5 * sz - abs_layer - air_layer - np.sum(heights_vector[0:i]) - 0.5 * \
                                heights_vector[i]

    # -----------------------------------------------------------------------------------------------------
    # ------------------------------- Simulation Variables -------------------------------------------------
    # ----------------------------------------------------------------------------------------------
    z_flux_trans = -0.5 * sz + abs_layer
    z_flux_refl = 0.5 * sz - abs_layer
    pt = mp.Vector3(0, 0, z_flux_refl)  # Field Decay point`

    src_pos = 0.5 * sz - abs_layer - 0.1 * air_layer

    # -----------------------------------------------------------------------------------------------
    # ------------------------------- Geometry Setup -------------------------------------------------
    # ----------------------------------------------------------------------------------------------

    pml_layers = [mp.PML(thickness=abs_layer, direction=mp.Z, side=mp.High),
                  mp.PML(thickness=abs_layer, direction=mp.Z, side=mp.Low)]

    geometry = []

    if not empty:
        # -------------------------------- coating layer creation ---------------------------------------
        if coating_layer is not None:
            geometry.append(
                mp.Cylinder(
                    material=coating_layer['material'],
                    radius=characteristic_vector[0] + coating_layer['size'],
                    height=coating_layer['size'],
                    center=mp.Vector3(0, 0,
                                      materials_position[0] + 0.5 * heights_vector[0] + 0.5 * coating_layer['size'])))
            for i in range(materials_position.size):
                if geometries_vector[i] == 'cylinder':
                    geometry.append(
                        mp.Cylinder(material=coating_layer['material'],
                                    radius=characteristic_vector[i] + coating_layer['size'],
                                    height=heights_vector[i],
                                    center=mp.Vector3(0, 0, materials_position[i])))
                elif geometries_vector[i] == 'block':
                    geometry.append(mp.Block(
                        material=coating_layer['material'],
                        size=mp.Vector3(characteristic_vector[i][0], characteristic_vector[i][1],
                                        coating_layer['size']),
                        center=mp.Vector3(0, 0,
                                          materials_position[i] + 0.5 * heights_vector[i] + 0.5 * coating_layer[
                                              'size'])))

        # -------------------------------- geometric  layer creation ---------------------------------------
        for i in range(materials_position.size):
            mat = gyr(materials_vector[i], magnetic_field)
            if geometries_vector[i] == 'cylinder':
                geometry.append(
                    mp.Cylinder(material=mat, radius=characteristic_vector[i], height=heights_vector[i],
                                center=mp.Vector3(0, 0, materials_position[i])))
            elif geometries_vector[i] == 'block':
                geometry.append(
                    mp.Block(
                        material=mat,
                        size=mp.Vector3(characteristic_vector[i][0], characteristic_vector[i][1], heights_vector[i]),
                        center=mp.Vector3(0, 0, materials_position[i])))

    # -----------------------------------------------------------------------------------------------
    # ------------------------------- Source Setup -------------------------------------------------
    # ----------------------------------------------------------------------------------------------
    fmin = um_scale / lmax  # source min frequency
    fmax = um_scale / lmin  # source max frequency
    fcen = 0.5 * (fmin + fmax)
    df = fmax - fmin

    # CCW rotation angle (degrees) about Y-axis of PW current source; 0 degrees along -z axis
    theta = math.radians(theta)

    # k with correct length (plane of incidence: XZ)
    k = mp.Vector3(math.sin(theta), 0, math.cos(theta)).scale(fcen)

    if theta == 0:
        k = mp.Vector3(0, 0, 0)

    def pw_amp(k, x0):
        def _pw_amp(x):
            return cmath.exp(1j * 2 * math.pi * k.dot(x + x0))

        return _pw_amp

    # ------------------- Empty simulation ---------------------------------

    sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Hx,
                         center=mp.Vector3(0, 0, src_pos),
                         size=mp.Vector3(sx, sy, 0),
                         amp_func=pw_amp(k, mp.Vector3(0, 0, src_pos)))]

    refl_fr = mp.FluxRegion(center=mp.Vector3(0, 0, z_flux_refl), size=mp.Vector3(sx, sy, 0))
    trans_fr = mp.FluxRegion(center=mp.Vector3(0, 0, z_flux_trans), size=mp.Vector3(sx, sy, 0))

    sim = mp.Simulation(cell_size=mp.Vector3(sx, sy, sz),
                        geometry=geometry,
                        sources=sources,
                        boundary_layers=pml_layers,
                        k_point=k,
                        resolution=resolution)

    refl = sim.add_flux(fcen, df, nfreq, refl_fr)

    trans = sim.add_flux(fcen, df, nfreq, trans_fr)

    if not empty:
        sim.load_minus_flux('refl-flux', refl)

    sim.run(until_after_sources=mp.stop_when_fields_decayed(25, mp.Ey, pt, 1e-3))

    if empty:
        sim.save_flux('refl-flux', refl)

    sim.display_fluxes(refl)

    # save incident power for transmission plane
    store['flux_freqs'] = mp.get_flux_freqs(refl)
    store['tran_flux'] = mp.get_fluxes(trans)
    store['refl_flux'] = mp.get_fluxes(refl)

    if empty:
        save_obj(store, 'empty_' + name)
    else:
        save_obj(store, 'disk_' + name)


def save_obj(store, fname):
    file = open(fname + '.obj', 'wb')
    pickle.dump(store, file)


def gyr(material, bias):
    if bias.norm() != 0:
        material_suc = []
        for susceptibility in material.E_susceptibilities:
            if type(susceptibility).__name__ == 'DrudeSusceptibility':
                material_suc.append(
                    mp.GyrotropicDrudeSusceptibility(frequency=susceptibility.frequency, gamma=susceptibility.gamma,
                                                     sigma=susceptibility.sigma_diag.x, bias=bias))
            elif type(susceptibility).__name__ == 'LorentzianSusceptibility':
                material_suc.append(
                    mp.GyrotropicLorentzianSusceptibility(frequency=susceptibility.frequency,
                                                          gamma=susceptibility.gamma,
                                                          sigma=susceptibility.sigma_diag.x, bias=bias))

        return mp.Medium(epsilon=material.epsilon_diag.x, E_susceptibilities=material_suc,
                         valid_freq_range=material.valid_freq_range)

    return material

