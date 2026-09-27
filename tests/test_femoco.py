import numpy as np
import pytest

from hamiltonians import chemistry
from operators import LCP


def test_femoco_reads_supplied_active_space_integrals(tmp_path, monkeypatch):
    path = tmp_path / "FCIDUMP"
    path.write_text(
        "&FCI NORB=54,NELEC=54,MS2=0,\n&END\n"
        " 1.0 1 1 0 0\n 5.0 0 0 0 0\n",
        encoding="utf-8",
    )
    target = LCP({"I" * 107 + "Z": -0.5, "I" * 106 + "ZI": -0.5})

    # Read and restore the bundled spatial integrals without running a full JW conversion.
    def convert(one_body, two_body, *, coefficient_tolerance):
        assert one_body.shape == (54, 54)
        assert one_body[0, 0] == 1.0 and np.count_nonzero(one_body) == 1
        assert two_body.shape == (54, 54, 54, 54)
        assert not np.any(two_body)
        assert coefficient_tolerance == 3e-12
        return target

    monkeypatch.setattr(chemistry, "FCIDUMP_PATH", path)
    monkeypatch.setattr(chemistry.jordan_wigner, "from_spatial_integrals", convert)
    monkeypatch.chdir(tmp_path)
    assert chemistry.generate(chemistry.FEMOCO_NAME, coefficient_tolerance=3e-12) is target


@pytest.mark.parametrize("header", ["NORB=2,NELEC=54", "NORB=54,NELEC=2"])
def test_femoco_requires_the_paper_active_space(tmp_path, monkeypatch, header):
    path = tmp_path / "not_femoco.FCIDUMP"
    path.write_text(f"&FCI {header},MS2=0,\n&END\n 0.0 0 0 0 0\n", encoding="utf-8")

    monkeypatch.setattr(chemistry, "FCIDUMP_PATH", path)
    with pytest.raises(ValueError, match="NORB=54.*NELEC=54"):
        chemistry.generate(chemistry.FEMOCO_NAME)


def test_femoco_missing_file_raises_file_not_found(tmp_path, monkeypatch):
    monkeypatch.setattr(chemistry, "FCIDUMP_PATH", tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        chemistry.generate(chemistry.FEMOCO_NAME)
