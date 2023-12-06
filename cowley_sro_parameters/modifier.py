from itertools import product
from typing import Mapping

import numpy as np
from ovito.data import DataCollection


def sro_modifier(type_map: Mapping[int, str] = None) -> callable:
    """
    Callable modifier that calculates Cowley SROs
    :param type_map: optional mapping that maps integer labels to other labels
                     if not specified, matrix elements will be stored as integers in the data collection
    :return: callable modifier
    """

    def wrapper(frame: int, data: DataCollection) -> None:
        """
        Wrapper function that acts as the SRO modifier
        :param frame: frame to evaluate SRO parameters (needed for OVITO custom modifier interface)
        :param data: data collection to modify
        :return: None
        """

        # get bond topology
        topology = np.array(data.particles.bonds.topology[...], dtype=int)

        # get types of each particle, and associated quantities
        types = np.array(data.particles['Particle Type'][...], dtype=int)
        unique_types = list(set(types))
        num_types = len(unique_types)

        # calculate concentrations
        concentrations = np.zeros(num_types)
        for i, type_ in enumerate(unique_types):
            concentrations[i] = np.mean(types == type_)

        # initialize an array that stores the number of (i, j) bonds we have
        # where i and j are particle types
        bond_types_count_array = np.zeros((num_types, num_types))

        for index, bond in enumerate(topology):
            # get particle indexes from bond
            first_atom, second_atom = bond

            # get types of those particle indexes
            first_type, second_type = types[first_atom], types[second_atom]

            # count that bond
            bond_types_count_array[first_type - 1, second_type - 1] += 1

        # normalize by the number of bonds we have, make array symmetric
        probability_array = bond_types_count_array / len(topology)
        probability_array += probability_array.T

        # calculate sro_array
        # outer gives array such that O[i, j] = c[i] * c[j]
        # https://doi.org/10.1016/j.actamat.2022.117621
        sro_array = 1.0 - probability_array / (2.0 * np.outer(concentrations, concentrations))

        # store matrix elements in data collection
        for i, j in product(unique_types, repeat=2):

            # if a type map is provided, store values with atom names
            # else, store with integer type labels
            sro = sro_array[i - 1, j - 1]
            if not type_map:
                key = f'sro_{i}{j}'
            else:
                key = f'sro_{type_map[i]}{type_map[j]}'
            data.attributes[key] = sro

    return wrapper
