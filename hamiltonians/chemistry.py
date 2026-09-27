"""Identity-free Jordan--Wigner Hamiltonians for the paper's molecular systems.

The nine geometry/basis settings use neutral singlets and all basis orbitals.
Coordinates are in angstrom and energies in Hartree. FeMoco uses the bundled
Reiher CAS(54e,54o) active-space integrals.
"""

import math
from pathlib import Path

from . import jordan_wigner

FEMOCO_NAME = "femoco_reiher_54e_54o_jw"
FCIDUMP_PATH = Path(__file__).with_name("nitrogenase-54e-54o.fcidump")


# H2O: the two O-H bonds make angles +/-104.5/2 degrees with the z axis.
_water_x = 0.9576 * math.sin(math.radians(104.5 / 2.0))
_water_z = 0.9576 * math.cos(math.radians(104.5 / 2.0))
_WATER_GEOMETRY = (
    ("O", (0.0, 0.0, 0.0)),
    ("H", (_water_x, 0.0, _water_z)),
    ("H", (-_water_x, 0.0, _water_z)),
)

# NH3: three N-H bonds separated by 120 degrees around the z axis.
# cos(H-N-H) = (3 cos(polar_angle)^2 - 1) / 2.
_nh3_cos2 = (2.0 * math.cos(math.radians(106.7)) + 1.0) / 3.0
_nh3_radius = 1.012 * math.sqrt(1.0 - _nh3_cos2)
_nh3_height = 1.012 * math.sqrt(_nh3_cos2)
_AMMONIA_GEOMETRY = (("N", (0.0, 0.0, 0.0)),) + tuple(
    ("H", (
        _nh3_radius * math.cos(2.0 * math.pi * index / 3.0),
        _nh3_radius * math.sin(2.0 * math.pi * index / 3.0),
        _nh3_height,
    ))
    for index in range(3)
)

# CH4: tetrahedral vertices at distance 1.087 from the carbon.
_ch4_coordinate = 1.087 / math.sqrt(3.0)
_METHANE_GEOMETRY = (("C", (0.0, 0.0, 0.0)),) + tuple(
    ("H", tuple(sign * _ch4_coordinate for sign in signs))
    for signs in ((1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1))
)

# Each entry contains (geometry, basis).
MOLECULAR_HAMILTONIANS = {
    "h2_sto3g_jw": (
        (("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.735))), "sto-3g"
    ),
    "lih_sto3g_full_jw": (
        (("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.45))), "sto-3g"
    ),
    "beh2_sto3g_full_jw": (
        (("H", (0.0, 0.0, -1.3264)), ("Be", (0.0, 0.0, 0.0)),
         ("H", (0.0, 0.0, 1.3264))), "sto-3g"
    ),
    "h2o_sto3g_full_jw": (_WATER_GEOMETRY, "sto-3g"),
    "nh3_sto3g_full_jw": (_AMMONIA_GEOMETRY, "sto-3g"),
    "ch4_sto3g_full_jw": (_METHANE_GEOMETRY, "sto-3g"),
    "n2_sto3g_full_jw": (
        (("N", (0.0, 0.0, 0.0)), ("N", (0.0, 0.0, 1.0977))), "sto-3g"
    ),
    "h2o_ccpvdz_full_jw": (_WATER_GEOMETRY, "cc-pvdz"),
    "ch4_ccpvdz_full_jw": (_METHANE_GEOMETRY, "cc-pvdz"),
}


def generate(name, *, coefficient_tolerance=1e-12):
    """Return the selected molecular Hamiltonian as an identity-free LCP."""
    from pyscf import ao2mo, gto, scf

    if name == FEMOCO_NAME:
        from pyscf.tools import fcidump

        data = fcidump.read(str(FCIDUMP_PATH), verbose=False)
        if data["NORB"] != 54 or data["NELEC"] != 54:
            raise ValueError("Reiher FeMoco requires NORB=54 and NELEC=54")
        two_body = ao2mo.restore(1, data["H2"], data["NORB"])
        return jordan_wigner.from_spatial_integrals(
            data["H1"], two_body, coefficient_tolerance=coefficient_tolerance,
        )

    geometry, basis = MOLECULAR_HAMILTONIANS[name]
    molecule = gto.M(atom=geometry, basis=basis, unit="Angstrom", charge=0, spin=0, verbose=0)
    mean_field = scf.RHF(molecule).run()
    orbitals = mean_field.mo_coeff
    one_body = orbitals.T @ mean_field.get_hcore() @ orbitals
    two_body = ao2mo.restore(1, ao2mo.kernel(molecule, orbitals), orbitals.shape[1])
    target = jordan_wigner.from_spatial_integrals(
        one_body, two_body, coefficient_tolerance=coefficient_tolerance,
    )
    target.terms = dict(sorted(target.terms.items()))
    return target
