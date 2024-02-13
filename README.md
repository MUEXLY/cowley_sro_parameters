# Cowley short range order parameter modifier
=============================================

This repository contains an OVITO modifier that calculates the Cowley short range order parameters from a precomputed bond topology.

For use in a standalone Python script:

``pip install cowley_sro_parameters``

This code is intended to also interface with OVITO Pro, but this feature has not been tested. With OVITO Pro, you can install this modifier into your OVITO interface with:

``ovitos -m pip install --user cowley_sro_parameters``

An example of this repository used in a standalone script is in the `/example`_ directory. This package is a derivative of a larger work on vacancy concentration. If you find this package useful, please cite [our work on vacancy concentration](https://arxiv.org/abs/2402.07324) as well as the repository itself:

```bibtex
@misc{jeffries_sro_parameters,
    author={Jacob Jeffries},
    title={Cowley SRO Parameters Modifier},
    howpublished={\url{https://github.com/jwjeffr/cowley_sro_parameters}},
    year={2023}
}
```

# Acknowledgements

The  work  was  supported  by  the  grant  DE-SC0022980 funded by the U.S. Department of Energy,  Office of Science.