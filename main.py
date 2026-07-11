"""Small script to benchmark qDRIFT cost from a random Pauli Hamiltonian."""

from __future__ import annotations

from argparse import ArgumentParser
from typing import Literal

import numpy as np

from lcp import LCP
from lch import LCH
from qdrift_cost import QDriftCost, qdrift_cost


def _random_pauli_string(num_qubits: int, rng: np.random.Generator) -> str:
    paulis = np.array(["I", "X", "Y", "Z"])
    return "".join(rng.choice(paulis, size=num_qubits).tolist())


def build_random_lcp(
    num_terms: int,
    num_qubits: int,
    *,
    seed: int | None = None,
) -> LCP:
    """Build a random LCP with unique Pauli strings and random real coefficients."""
    if not isinstance(num_terms, int) or num_terms < 0:
        raise ValueError("num_terms must be a non-negative integer")
    if not isinstance(num_qubits, int) or num_qubits < 0:
        raise ValueError("num_qubits must be a non-negative integer")
    if num_terms > 4**num_qubits:
        raise ValueError(
            "num_terms cannot exceed 4^num_qubits because Pauli strings must be unique"
        )

    rng = np.random.default_rng(seed)
    terms: dict[str, complex] = {}
    while len(terms) < num_terms:
        pauli_string = _random_pauli_string(num_qubits, rng)
        # Ensure unique Pauli strings since LCP/LCH stores one coefficient per term.
        if pauli_string in terms:
            continue
        terms[pauli_string] = complex(float(rng.normal()))
    return LCP(terms, num_qubits=num_qubits)


def build_lch_from_lcp_unit_cost(lcp_hamiltonian: LCP) -> LCH:
    """Convert LCP to LCH while assigning per-term evolution cost 1.0."""
    return LCH(
        [
            (coefficient, pauli_string, 1.0)
            for pauli_string, coefficient in lcp_hamiltonian.terms.items()
        ],
        num_qubits=lcp_hamiltonian.num_qubits,
    )


def run_qdrift_from_random_lcp(
    *,
    num_terms: int,
    num_qubits: int,
    time: float,
    epsilon: float,
    strategy: Literal["theory", "variance", "hybrid"] = "hybrid",
    seed: int | None = None,
) -> QDriftCost:
    """Generate random LCP -> LCH (cost=1 each) -> return qDRIFT cost."""
    random_lcp = build_random_lcp(num_terms=num_terms, num_qubits=num_qubits, seed=seed)
    random_lch = build_lch_from_lcp_unit_cost(random_lcp)
    return qdrift_cost(random_lch, time=time, epsilon=epsilon, strategy=strategy)


def main() -> None:
    parser = ArgumentParser(description="Estimate qDRIFT cost from random LCP/LCH")
    parser.add_argument("--num-terms", type=int, default=6, help="number of Pauli terms")
    parser.add_argument("--num-qubits", type=int, default=3, help="number of qubits")
    parser.add_argument("--time", type=float, default=1.0, help="evolution time")
    parser.add_argument("--epsilon", type=float, default=1e-3, help="qDRIFT error target")
    parser.add_argument(
        "--strategy",
        type=str,
        default="theory",
        choices=["theory", "variance", "hybrid"],
        help="qDRIFT step estimation strategy (theory is fast and recommended for large instances)",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--print-probabilities",
        action="store_true",
        help="print full sampling probability vector (large arrays may be noisy)",
    )
    args = parser.parse_args()

    if args.strategy != "theory" and (args.num_terms > 200 or args.num_qubits > 12):
        print(
            "Warning: the selected strategy can be expensive for large instances. "
            "Consider using --strategy theory for a fast estimate."
        )

    result = run_qdrift_from_random_lcp(
        num_terms=args.num_terms,
        num_qubits=args.num_qubits,
        time=args.time,
        epsilon=args.epsilon,
        strategy=args.strategy,
        seed=args.seed,
    )
    print(f"num_terms={args.num_terms}, num_qubits={args.num_qubits}")
    print(f"steps={result.steps}, lambda_sum={result.lambda_sum:.6f}, per_step_cost={result.per_step_cost:.6f}")
    print(f"theory_steps={result.theory_steps}, variance_steps={result.variance_steps}, total_cost={result.total_cost:.6f}")
    if args.print_probabilities:
        print(f"sampling probabilities={result.sampling_probabilities}")
    else:
        probs = result.sampling_probabilities
        if probs.size:
            top_k = min(10, probs.size)
            idx = probs.argsort()[::-1][:top_k]
            top_pairs = [(int(i), float(probs[i])) for i in idx]
            print(
                "top probabilities="
                + ", ".join(f"({i}, {p:.6e})" for i, p in top_pairs)
            )
            print(f"probability min={probs.min():.6e}, max={probs.max():.6e}, sum={probs.sum():.6f}")
        else:
            print("sampling probabilities=[]")


if __name__ == "__main__":
    main()
