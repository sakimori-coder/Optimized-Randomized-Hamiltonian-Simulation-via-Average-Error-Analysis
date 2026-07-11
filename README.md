## LCP: linear combinations of Pauli strings

```python
from lcp import LCP

# H = 0.5 Z⊗I + 0.5 I⊗Z + X⊗X
h = LCP({"ZI": 0.5, "IZ": 0.5, "XX": 1.0})

matrix = h.to_matrix()
csr_matrix = h.to_csr()
print(h.coefficient_one_norm())  # 2.0
print(h.operator_norm())

# Operator distance ||H1 - H2|| and scalar multiplication
distance = (h - 0.5 * h).operator_norm()

# Nodes are Pauli strings; edges connect commuting pairs.
graph = h.commutation_graph()

from commuting_groups import decompose_into_commuting_groups

# H = sum(groups), and every group consists of mutually commuting terms.
groups = decompose_into_commuting_groups(h)

from rotation_depth import minimum_pauli_rotation_depth

# Exact minimum under the free-Clifford, parallel-Rz cost model.
depth = minimum_pauli_rotation_depth(groups[0])
```

Pauli strings use `I`, `X`, `Y`, and `Z`; their left-most character is the
most-significant qubit. Zero coefficients and very small coefficients are kept.

## LCH: linear combinations of Pauli-string Hamiltonians

```python
from lch import LCH

h = LCH([(0.3, "X", 2), (0.7, "Z", 5)], num_qubits=1)

# LCH has an internal LCP representation:
print(type(h.lcp).__name__)  # LCP

print(h.coefficient_one_norm())  # 1.0
print(h.sampling_probabilities())  # [0.3 0.7]
print(h.evolution_costs)  # [2.0, 5.0]
print(h.to_matrix())
print(h.operator_norm())
```

`LCH` stores each term as a Pauli string and internally uses :class:`LCP` to
manage Hamiltonian coefficients. It can be materialized to
CSR/dense form as needed.

## qDRIFT step/cost estimate

```python
from qdrift_cost import qdrift_cost
from lch import LCH
h = LCH(
    [
        (0.3, "X", 2.0),
        (0.7, "Z", 5.0),
    ],
    num_qubits=1,
)

cost = qdrift_cost(h, time=1.2, epsilon=1e-3, strategy="hybrid")
print(cost.steps)
print(cost.per_step_cost)
print(cost.total_cost)
print(cost.sampling_probabilities)
```

`strategy="hybrid"` uses both
1) qDRIFT標準界（Λ-based）
2) 二次モーメント（BCH 2次項）を使う数値的境界

係数重みは `|c_j|` で、現在の実装では `||H_j||` は分布重みや
ステップ数見積もりには直接入れません（Pauli項前提のため）。
