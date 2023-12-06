from itertools import combinations_with_replacement

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import ovito
from matplotlib.gridspec import GridSpec

from cowley_sro_parameters import sro_modifier


# type map constant info
TYPE_MAP = {1: 'Co', 2: 'Ni', 3: 'Cr', 4: 'Fe', 5: 'Mn'}
INVERSE_TYPE_MAP = {val: key for key, val in TYPE_MAP.items()}
NUM_TYPES = len(TYPE_MAP)

# cutoff info for SRO calculations
FIRST_CUTOFF = 0.0
SECOND_CUTOFF = 3.1


def main():
    # need to use Agg or else matplotlib conflicts with ovito
    mpl.use('Agg')

    # create ovito pipeline, add a bonds modifier to create needed topology
    pipeline = ovito.io.import_file('mc.dump')
    bonds_modifier = ovito.modifiers.CreateBondsModifier(lower_cutoff=FIRST_CUTOFF, cutoff=SECOND_CUTOFF)
    pipeline.modifiers.append(bonds_modifier)

    # create SRO modifier which calculates all SRO's and the Frobenius-normed SRO matrix at each timestep
    modifier = sro_modifier(type_map=TYPE_MAP)
    pipeline.modifiers.append(modifier)

    # initialize atom pairs
    pairs = list(combinations_with_replacement(TYPE_MAP.values(), 2))

    # initialize arrays to store values in
    frames = np.arange(pipeline.source.num_frames)
    timestep = np.zeros(frames.shape)
    sro_params = np.zeros((frames.shape[0], NUM_TYPES, NUM_TYPES))

    # store data attributes in initialized arrays
    for frame in frames:

        data = pipeline.compute(frame)
        timestep[frame] = data.attributes['Timestep']
        for e1, e2 in pairs:
            i, j = INVERSE_TYPE_MAP[e1], INVERSE_TYPE_MAP[e2]
            sro_params[frame, i - 1, j - 1] = data.attributes[f'sro_{e1}{e2}']

    # first, create gridspec for SRO plots
    fig = plt.figure()
    gs = GridSpec(NUM_TYPES, NUM_TYPES, figure=fig, wspace=0.3, hspace=0.3)

    # loop through the lower-diagonal of the SRO matrix
    for i in range(NUM_TYPES):
        for j in range(i + 1):
            t = TYPE_MAP[i + 1], TYPE_MAP[j + 1]
            ax = plt.subplot(gs[i, j])
            ax.set_xlim([-0.1, 5.1])
            ax.set_ylim([-1.1, 1.1])
            ax.set_xticks([0, 1, 2, 3, 4, 5])
            ax.set_yticks([-1, -0.5, 0, 0.5, 1])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid()
            # along the left side of the grid, label y-axis ticks
            if j == 0:
                ax.set_yticklabels([-1, '', 0, '', 1])

                # add y-axes label in the middle
                if i == (NUM_TYPES + 1) // 2 - 1:
                    ax.set_ylabel(r"$\alpha$-$\alpha'$ Cowley SRO parameter ($\chi_{\alpha\alpha'}$)")

            # along the bottom side of the grid, label x-axis ticks
            if i == NUM_TYPES - 1:
                ax.set_xticklabels([0, '', '', '', '', 5])

                # add x-axes label in the middle
                if j == (NUM_TYPES + 1) // 2 - 1:
                    ax.set_xlabel('MC-MD time (ns)')

            # along the diagonal, label atom type names
            if i == j:
                ax.text(2.5, 1.3, r"$\alpha = $" + t[1], ha='center', va='bottom')
                ax.text(5.4, 0.0, r"$\alpha' = $" + t[0], ha='left', va='center')
            ax.plot(1e-6 * timestep, sro_params[:, j, i], color='black')

    fig.tight_layout()

    fig.savefig('sro_params.svg', bbox_inches='tight')


if __name__ == '__main__':
    main()
