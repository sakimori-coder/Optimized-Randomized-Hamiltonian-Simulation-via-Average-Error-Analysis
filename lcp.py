r"""Linear combinations of Pauli strings.

The left-most character is the most-significant qubit.  Thus ``"XI"``
represents :math:`X \otimes I`.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations
from numbers import Number

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy import sparse


_PAULI_MATRICES: dict[str, NDArray[np.complex128]] = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}
_VALID_PAULIS = frozenset(_PAULI_MATRICES)


class LCP:
    """Linear combination of Pauli strings of the same length.

    Args:
        terms: Mapping from a Pauli string to its coefficient.  Coefficients,
            including zero and very small values, are stored unchanged.
        num_qubits: Number of qubits.  This is inferred from ``terms`` when
            possible and is only necessary when ``terms`` is empty.

    Example:
        >>> h = LCP({"ZI": 0.5, "IZ": 0.5, "XX": 1.0})
        >>> h.num_qubits
        2
        >>> h.coefficient_one_norm()
        2.0
    """

    def __init__(
        self,
        terms: Mapping[str, complex] | None = None,
        *,
        num_qubits: int | None = None,
    ) -> None:
        if num_qubits is not None and (
            not isinstance(num_qubits, int) or isinstance(num_qubits, bool) or num_qubits < 0
        ):
            raise ValueError("num_qubits must be a non-negative integer")

        terms = {} if terms is None else terms
        if not isinstance(terms, Mapping):
            raise TypeError("terms must be a mapping from Pauli strings to coefficients")

        inferred_num_qubits = num_qubits
        validated_terms: dict[str, complex] = {}
        for pauli_string, coefficient in terms.items():
            self._validate_pauli_string(pauli_string)
            if not isinstance(coefficient, Number):
                raise TypeError(f"coefficient of {pauli_string!r} must be a number")

            if inferred_num_qubits is None:
                inferred_num_qubits = len(pauli_string)
            if len(pauli_string) != inferred_num_qubits:
                raise ValueError("all Pauli strings must have the same length as num_qubits")

            # Do not simplify or discard coefficients here.
            validated_terms[pauli_string] = complex(coefficient)

        if inferred_num_qubits is None:
            raise ValueError("num_qubits is required when terms is empty")

        self._terms = validated_terms
        self._num_qubits = inferred_num_qubits

    @staticmethod
    def _validate_pauli_string(pauli_string: str) -> None:
        if not isinstance(pauli_string, str):
            raise TypeError("Pauli strings must be str instances")
        invalid_symbols = set(pauli_string) - _VALID_PAULIS
        if invalid_symbols:
            raise ValueError(f"invalid Pauli character(s): {sorted(invalid_symbols)}")

    @property
    def terms(self) -> dict[str, complex]:
        """Return a copy of the Pauli-string-to-coefficient mapping."""
        return dict(self._terms)

    @property
    def num_qubits(self) -> int:
        """Number of qubits on which the Hamiltonian acts."""
        return self._num_qubits

    def __sub__(self, other: LCP) -> LCP:
        """Subtract two operators acting on the same number of qubits."""
        if not isinstance(other, LCP):
            return NotImplemented
        if self.num_qubits != other.num_qubits:
            raise ValueError("LCP operands must act on the same number of qubits")

        result = self.terms
        for pauli_string, coefficient in other._terms.items():
            result[pauli_string] = result.get(pauli_string, 0j) - coefficient
        return LCP(result, num_qubits=self.num_qubits)

    def __mul__(self, scalar: complex) -> LCP:
        """Multiply every coefficient by a scalar."""
        if not isinstance(scalar, Number):
            return NotImplemented
        return LCP(
            {
                pauli_string: coefficient * scalar
                for pauli_string, coefficient in self._terms.items()
            },
            num_qubits=self.num_qubits,
        )

    def __rmul__(self, scalar: complex) -> LCP:
        """Multiply every coefficient by a scalar."""
        return self * scalar

    @staticmethod
    def _pauli_strings_commute(left: str, right: str) -> bool:
        """Return whether two equal-length Pauli strings commute."""
        anti_commuting_positions = sum(
            left_symbol != "I"
            and right_symbol != "I"
            and left_symbol != right_symbol
            for left_symbol, right_symbol in zip(left, right)
        )
        return anti_commuting_positions % 2 == 0

    def commutation_graph(self) -> nx.Graph:
        """Return the graph whose edges connect commuting Pauli strings.

        Every Pauli string is a node.  Its LCP coefficient is stored in the
        ``coefficient`` node attribute.  The graph has no self-loops.
        """
        graph = nx.Graph()
        graph.add_nodes_from(
            (pauli_string, {"coefficient": coefficient})
            for pauli_string, coefficient in self._terms.items()
        )

        for left, right in combinations(self._terms, 2):
            if self._pauli_strings_commute(left, right):
                graph.add_edge(left, right)

        return graph

    def to_matrix(self) -> NDArray[np.complex128]:
        """Convert the linear combination to a dense NumPy matrix."""
        dimension = 2**self.num_qubits
        result = np.zeros((dimension, dimension), dtype=complex)

        for pauli_string, coefficient in self._terms.items():
            term = np.array([[1]], dtype=complex)
            for symbol in pauli_string:
                term = np.kron(term, _PAULI_MATRICES[symbol])
            result += coefficient * term

        return result

    def to_csr(self) -> sparse.csr_matrix:
        """Convert the linear combination to a SciPy CSR sparse matrix."""
        dimension = 2**self.num_qubits
        result = sparse.csr_matrix((dimension, dimension), dtype=complex)

        for pauli_string, coefficient in self._terms.items():
            term = sparse.csr_matrix([[1]], dtype=complex)
            for symbol in pauli_string:
                term = sparse.kron(
                    term,
                    sparse.csr_matrix(_PAULI_MATRICES[symbol]),
                    format="csr",
                )
            result += coefficient * term

        return result.tocsr()

    def coefficient_one_norm(self) -> float:
        """Return the coefficient 1-norm, ``sum(abs(coefficient))``."""
        return float(sum(abs(coefficient) for coefficient in self._terms.values()))

    def operator_norm(self) -> float:
        """Return the spectral norm using the CSR representation.

        The spectral norm is the largest singular value of the operator.
        """
        matrix = self.to_csr()

        if matrix.nnz == 0:
            return 0.0
        if matrix.shape[0] <= 2:
            # ARPACK requires k < dimension - 1, so it cannot handle the
            # one-qubit (2 x 2) case with k=1.
            return float(np.linalg.norm(matrix.toarray(), ord=2))

        largest_singular_value = sparse.linalg.svds(
            matrix,
            k=1,
            which="LM",
            return_singular_vectors=False,
            tol=1e-7,
        )[0]
        return float(abs(largest_singular_value))
