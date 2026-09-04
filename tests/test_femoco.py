from pathlib import Path

import numpy as np
import pytest
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.ops import InteractionOperator
from openfermion.transforms import jordan_wigner
from pyscf import ao2mo
from pyscf.tools import fcidump

from hamiltonians.femoco import (
    ReiherFeMocoHamiltonianPreset,
    _packed_integrals_to_pauli_terms,
)


_SMALL_FCIDUMP = """\
&FCI NORB=2,NELEC=2,MS2=0,
 ORBSYM=1,1,
 ISYM=1,
&END
 0.700000000000 1 1 1 1
 0.100000000000 2 1 1 1
 0.300000000000 2 1 2 1
 0.400000000000 2 2 1 1
-0.050000000000 2 2 2 1
 0.600000000000 2 2 2 2
-1.000000000000 1 1 0 0
 0.200000000000 2 1 0 0
-0.400000000000 2 2 0 0
 0.125000000000 0 0 0 0
"""


def _write_small_fcidump(path: Path) -> None:
    path.write_text(_SMALL_FCIDUMP, encoding="utf-8")


def _openfermion_reference(path: Path) -> dict[str, float]:
    data = fcidump.read(str(path), verbose=False)
    num_orbitals = int(data["NORB"])
    spatial_two_body = ao2mo.restore(1, data["H2"], num_orbitals)
    one_body, two_body = spinorb_from_spatial(
        data["H1"],
        spatial_two_body.transpose(0, 2, 3, 1),
    )
    qubit_operator = jordan_wigner(
        InteractionOperator(data["ECORE"], one_body, 0.5 * two_body)
    )
    reference: dict[str, float] = {}
    for indexed_paulis, coefficient in qubit_operator.terms.items():
        symbols = ["I"] * (2 * num_orbitals)
        for index, symbol in indexed_paulis:
            symbols[2 * num_orbitals - index - 1] = symbol
        reference["".join(symbols)] = float(complex(coefficient).real)
    return reference


def test_direct_fcidump_jordan_wigner_matches_openfermion(
    tmp_path: Path,
) -> None:
    path = tmp_path / "FCIDUMP"
    _write_small_fcidump(path)
    data = fcidump.read(str(path), verbose=False)
    direct = {
        pauli: coefficient
        for coefficient, pauli in _packed_integrals_to_pauli_terms(
            np.asarray(data["H1"], dtype=float),
            np.asarray(data["H2"], dtype=float),
            float(data["ECORE"]),
            1e-14,
        )
    }

    assert direct == pytest.approx(_openfermion_reference(path), abs=1e-12)


def test_reiher_femoco_preset_validates_active_space(tmp_path: Path) -> None:
    path = tmp_path / "not_femoco.FCIDUMP"
    _write_small_fcidump(path)

    with pytest.raises(ValueError, match="NORB=54"):
        ReiherFeMocoHamiltonianPreset(path).generate()


def test_reiher_femoco_missing_path_is_clear(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="FCIDUMP was not found"):
        ReiherFeMocoHamiltonianPreset(tmp_path / "missing").generate()
