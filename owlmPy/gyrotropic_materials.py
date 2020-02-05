import meep as mp


def gyrotropic_conversion (material, bias_vector):
    bias = mp.Vector3(bias_vector[0], bias_vector[1], bias_vector[2])
    material_suc = []
    for susceptibility in material.E_susceptibilities:
        if type(susceptibility).__name__ == 'DrudeSusceptibility':
            material_suc.append(
                mp.GyrotropicDrudeSusceptibility(frequency=susceptibility.frequency, gamma=susceptibility.gamma,
                                                 sigma=susceptibility.sigma_diag.x, bias=bias))
        elif type(susceptibility).__name__ == 'LorentzianSusceptibility':
            material_suc.append(
                mp.GyrotropicLorentzianSusceptibility(frequency=susceptibility.frequency, gamma=susceptibility.gamma,
                                                 sigma=susceptibility.sigma_diag.x, bias=bias))

    return mp.Medium(epsilon=material.epsilon_diag.x, E_susceptibilities=material_suc, valid_freq_range=material.valid_freq_range)