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

cost = qdrift_cost(
    h,
    time=1.2,
    epsilon=1e-3,
    variance_method="sdp_bound",
)
print(cost.steps)
print(cost.normalized_variance_bound)
print(cost.variance_constant)
print(cost.per_step_cost)
print(cost.total_cost)
print(cost.sampling_probabilities)
```

分散評価法は`variance_method`で指定します。`exact`、
`contraction_bound`、`pauli_l1_bound`、`anticommuting_bound`、`sdp_bound`、
またはカスタム推定関数を使用できます。デフォルトの`contraction_bound`は
`B=1`を使う標準qDRIFTステップ数に対応し、行列を生成しません。

対象となる正規化分散

`||sum_j p_j (H/Lambda - H_j)^2||_op`

の数値評価または上限を`B_hat`とします。
`variance_constant = Lambda**2 * B_hat`であり、ステップ数は
`ceil(2 * variance_constant * abs(time)**2 / epsilon)`です。

係数重みは `|c_j|` で、現在の実装では `||H_j||` は分布重みや
ステップ数見積もりには直接入れません（Pauli項前提のため）。

## Pauli qDRIFTと可換グループqDRIFTの比較

```bash
python main.py \
  --num-terms 40 \
  --num-qubits 5 \
  --time 1.0 \
  --epsilon 0.01 \
  --seed 42 \
  --variance-method sdp_bound \
  --group-norm-method lp \
  --max-group-size 12 \
  --print-probabilities
```

可換グループ `G_g` ごとに作用素ノルムの推定上限 `h_g`を求め、
`H_g = G_g / h_g` として `H = sum_g h_g H_g` を構成します。
`--variance-method`は両方のqDRIFTに共通で、`exact`、
`contraction_bound`、`pauli_l1_bound`、`anticommuting_bound`、`sdp_bound`から
選べます。
`--group-norm-method`は
`coefficient_l1`、`lp`、`exact`から選べます。

単項版の1サンプルのコストはLCHに格納されたPauli回転深さ、
グループ版は可換項を並列実行したときの厳密な最小Pauli回転深さです。
表示される総コストは、ステップ数とサンプリング分布で平均した
1ステップ深さの積です。

`contraction_bound`/`pauli_l1_bound`/`anticommuting_bound`/`sdp_bound`と
`coefficient_l1`/`lp`の組み合わせは
`2^n`次元行列を生成しません。分散評価の`exact`と
グループノルムの`exact`は行列を生成するため、大きなqubit数には適しません。`--max-group-size`は
貪欲分割の1グループに含める最大項数です。

## 量子化学HamiltonianでのqDRIFT実験

次の分子Hamiltonianプリセットを使って、単項qDRIFTと可換グループqDRIFTを
比較できます。

- `h2_sto3g_jw`: H2、結合長0.735 Å、STO-3G、Jordan--Wigner
- `lih_sto3g_active_jw`: LiH、結合長1.45 Å、STO-3G、軌道0を凍結し
  空間軌道1--2を活性空間にしたJordan--Wigner Hamiltonian
- `h2o_sto3g_cas_4e_4o_jw`: H2O、O--H距離0.9576 Å、結合角104.5度、
  STO-3G、CAS(4e,4o)の8量子ビットJordan--Wigner Hamiltonian

```bash
uv run python quantum_chemistry_qdrift.py \
  --hamiltonian h2_sto3g_jw \
  --variance-method all \
  --group-norm-method lp \
  --time 1.0 \
  --epsilon 0.01 \
  --print-groups
```

Pauli係数はファイルに保存していません。実行時にプリセットの分子座標から
PySCFで分子積分を計算し、OpenFermionでJordan--Wigner変換して`LCH`を
生成します。同じ分子設定の計算結果は実行中のプロセス内でキャッシュされます。

任意の分子構造から直接`LCH`を生成することもできます。

```python
from quantum_chemistry_qdrift import generate_molecular_lch

h = generate_molecular_lch(
    [
        ("H", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, 0.90)),
    ],
    basis="sto-3g",
    multiplicity=1,
)
```

LiHを実行する場合はプリセット名を切り替えます。

```bash
uv run python quantum_chemistry_qdrift.py \
  --hamiltonian lih_sto3g_active_jw \
  --variance-method all \
  --group-norm-method lp \
  --time 1.0 \
  --epsilon 0.01 \
  --print-groups
```

8量子ビットH2Oを比較する場合は、例えば次を実行します。

```bash
uv run python quantum_chemistry_qdrift.py \
  --hamiltonian h2o_sto3g_cas_4e_4o_jw \
  --variance-method sdp_bound \
  --group-norm-method lp \
  --max-group-size 3 \
  --time 1.0 \
  --epsilon 0.01
```

このH2Oでは`sdp_bound`のdense SDPがH2や活性空間LiHより重く、手元の
環境では数十秒程度かかります。素早く粗い比較を行う場合は
`--variance-method contraction_bound`を指定します。グループ最大サイズは
総深さに影響し、この例では`3`にすると`sdp_bound`による総深さが単項版の
約69.9%になりました。

恒等Pauli項はグローバル位相だけを与えるため、デフォルトでは除外します。
qDRIFT分布へ含める場合は`--include-identity`を指定します。
新しい分子Hamiltonianは`quantum_chemistry_qdrift.py`の
`MOLECULAR_HAMILTONIANS`へ座標・基底・活性空間のプリセットを追加できます。

## qDRIFT分散項の行列フリー上界

```python
from qdrift_variance_estimator import (
    estimate_lch_centered_second_moment_norm,
)

contraction = estimate_lch_centered_second_moment_norm(
    hamiltonian, method="contraction_bound"
)
pauli_l1 = estimate_lch_centered_second_moment_norm(
    hamiltonian, method="pauli_l1_bound"
)
anticommuting = estimate_lch_centered_second_moment_norm(
    hamiltonian, method="anticommuting_bound"
)
moment_sdp = estimate_lch_centered_second_moment_norm(
    hamiltonian, method="sdp_bound"
)
```

いずれも実係数Pauli LCHに対する
`||sum_j p_j (H/Lambda - H_j)^2||_op` の上界です。`contraction_bound`
は項数に線形、`pauli_l1_bound`はPauli積をハッシュ集約して`O(n L^2)`、
`anticommuting_bound`はその非恒等項を貪欲に相互反可換集合へ分割します。
`sdp_bound`はPauli係数共分散
`C = Z (Diag(p) - p p.T) Z.T`を作り、Pauli積の一致・反可換性と
`<A_g**2> <= 1`を制約に持つlevel-1 Pauli moment SDPを解きます。
Pauli support数を`d`とすると前処理は`O(n d^2)`で、さらにdenseな
`d x d` SDPを解くため、他の行列フリー上界より重い方法です。ただし、
いずれの経路も`2^n`次元のHamiltonian行列は生成しません。
グループ化qDRIFTの`pauli_l1_bound`は、正規化グループを`A_g`として
`sum_g p_g A_g^2 - (sum_g p_g A_g)^2`全体をPauli展開します。
`anticommuting_bound`は同じ展開結果を相互反可換なPauli集合へ貪欲分割し、
各集合の係数ユークリッドノルムを使います。
`sdp_bound`はCVXPYで双対Pauli SOS問題を解き、数値Gram行列をPSDへ
射影した後、Pauli係数残差の1-normを加えて上方補正します。

## 可換Pauliハミルトニアンの作用素ノルム上界

```python
from commuting_operator_norm_estimator import (
    estimate_commuting_operator_norm_upper_bound,
)
from lcp import LCP

h = LCP({"ZII": 1, "IZI": 1, "IIZ": 1, "ZZZ": -1})

l1_bound = estimate_commuting_operator_norm_upper_bound(h)
lp_bound = estimate_commuting_operator_norm_upper_bound(
    h,
    method="lp",
)
sparse_norm = estimate_commuting_operator_norm_upper_bound(
    h,
    method="exact",
)
print(l1_bound)     # 4.0
print(lp_bound)     # 2.0
print(sparse_norm)  # approximately 2.0
```

`coefficient_l1`は三角不等式
`||H||_op <= sum_j abs(c_j)`を使います。計算量は`O(L)`で、`2^n`次元行列を
生成しません。

`lp`は可換Pauli積の依存関係
`prod_{j in S} P_j = (-1)^b I`をGF(2)消去で求め、同時固有値の符号に課される
parity制約を使って係数`l1`上限を締めます。各parity polytopeは2状態の
trellis-flow LPで表すため、関係の長さに対して線形サイズで、`2^n`次元行列を
生成しません。非恒等項数を`m`、依存関係数を`r`とすると、LP全体のサイズは
`O(m r)`（最悪`O(m^2)`）です。常に
`||H||_op <= lp_bound <= coefficient_l1_bound`ですが、複数の依存関係がある場合は
一般には厳密値とは限らず、選ばれた依存関係基底によって上限の強さも変わります。
返り値にはLPの主問題値ではなく、双対乗数から再構成した上限証明を使います。

`exact`は`2^n`次元のCSR疎行列を実際に構築し、
SciPyの反復SVDで最大特異値を計算します。時間とメモリはqubit数に対して指数的で、反復法による
数値推定値なので厳密な上限保証はありません。全methodで入力は有界な実係数の
可換LCPであることを仮定し、この推定器内では追加検証を行いません。

## 大qubit数でのqDRIFT比較

```bash
python main.py \
  --num-terms 60 \
  --num-qubits 100 \
  --time 1.0 \
  --epsilon 0.01 \
  --seed 42 \
  --variance-method sdp_bound \
  --group-norm-method lp \
  --max-group-size 12
```

この設定では、貪欲的な可換分割、LPによるグループノルム上限、
行列フリーな分散上限のすべてを`main.py`内の同じ比較経路で使います。
