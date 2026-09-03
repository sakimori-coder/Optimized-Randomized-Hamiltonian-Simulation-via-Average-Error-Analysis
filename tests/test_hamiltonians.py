import math

from hamiltonians.chemistry import H2_STO3G_JW, MOLECULAR_HAMILTONIANS
from hamiltonians.syk import SykHamiltonianPreset


def test_h2_preset_generates_identity_free_target() -> None:
    generated = H2_STO3G_JW.generate()
    target = generated.to_lcp(include_identity=False)

    assert generated.num_qubits == 4
    assert "IIII" not in target.terms
    assert H2_STO3G_JW.name in MOLECULAR_HAMILTONIANS


def test_syk_is_reproducible_and_has_one_term_per_quartet() -> None:
    preset = SykHamiltonianPreset(num_qubits=4, seed=13)
    first = preset.generate()
    second = preset.generate()

    assert first.terms == second.terms
    assert len(first.terms) == math.comb(8, 4)
    assert first.to_lcp().num_qubits == 4

