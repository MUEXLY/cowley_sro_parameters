from typing import Mapping

import numpy as np
from ovito.data import DataCollection, NearestNeighborFinder


def nearest_neighbor_topology(num_neighbors: int) -> callable:
    """
    Callable modifier that stores topology from a set of nearest neighbors
    :param num_neighbors: number of nearest neighbors to find per atom
    :return: callable modifier
    """

    def wrapper(frame: int, data: DataCollection) -> None:
        """
        Wrapper function that acts as the topology creation modifier
        :param frame: frame to evaluate SRO parameters (needed for OVITO custom modifier interface)
        :param data: data collection to modify
        :return: None
        """

        finder = NearestNeighborFinder(num_neighbors, data)
        topology = set()

        for index in range(data.particles.count):

            for neigh in finder.find(index):
                pair = (index, neigh.index)
                if pair in topology or (pair[1], pair[0]) in topology:
                    continue
                topology.add(pair)

        bonds = data.particles_.create_bonds(count=len(topology))
        bonds.create_property('Topology', data=list(topology))
    
    return wrapper


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

        # if particle type names are available, use them
        particle_types = [t.name for t in data.particles.particle_types.types]
        type_ids = [data.particles.particle_types.type_by_name(t).id for t in particle_types]

        # if not available, just use numeric IDs
        if all(t == "" for t in particle_types):
            particle_types = np.unique(data.particles.particle_types[...])
            type_ids = particle_types.copy()

        type_map_ = {id_: t for id_, t in zip(type_ids, particle_types)}

        concentrations = {}
        for id_, t in zip(type_ids, particle_types):
            concentrations[t] = (data.particles.particle_types == id_).mean()

        # initialize an array that stores the number of (i, j) bonds we have
        # where i and j are particle types
        # bond_types_count_array = np.zeros((num_types, num_types))

        bond_count_map = {}
        for first_type in particle_types:
            for second_type in particle_types:
                bond_count_map[first_type, second_type] = 0

        for index, bond in enumerate(topology):
            # get particle indexes from bond
            first_atom, second_atom = bond

            # get types of those particle indexes
            # first_type, second_type = types[first_atom], types[second_atom]
            first_type = type_map_[data.particles.particle_types[first_atom]]
            second_type = type_map_[data.particles.particle_types[second_atom]]

            # count that bond
            bond_count_map[first_type, second_type] += 1

        # normalize by total number of bonds
        bond_count_map = {key: val / len(topology) for key, val in bond_count_map.items()}

        # symmetrize
        symmetrized_bond_count_map = {}
        for key, val in bond_count_map.items():
            reversed_key = (key[1], key[0])
            symmetrized_bond_count_map[key] = val + bond_count_map[reversed_key]

        sro_map = {}
        for key, val in symmetrized_bond_count_map.items():
            sro_map[key] = 1.0 - val / (2.0 * concentrations[key[0]] * concentrations[key[1]])

        for key, val in sro_map.items():
            i, j = key
            if type_map is not None:
                i = type_map[i]
                j = type_map[j]
            data.attributes[f"sro_{i}{j}"] = val

    return wrapper
