## Hamiltonian生成

Hamiltonian生成機能は用途ごとに`hamiltonians/`へまとめています。

```python
from hamiltonians.chemistry import H2O_STO3G_CAS_4E_4O_JW
from hamiltonians.physics import (
    heisenberg_chain,
    schwinger_model,
    transverse_field_ising_chain,
)
from hamiltonians.random import build_random_lcp

h2o = H2O_STO3G_CAS_4E_4O_JW.to_lcp()
ising = transverse_field_ising_chain(8, coupling=1.0, transverse_field=0.5)
heisenberg = heisenberg_chain(8, coupling=1.0, periodic=True)
schwinger = schwinger_model(
    8,
    x=1.0,
    mu=0.5,
    background_field=0.0,
)
random_hamiltonian = build_random_lcp(200, 5, seed=42)
```

- `chemistry.py`: PySCF・OpenFermionによる分子Hamiltonianとプリセット
- `physics.py`: Ising鎖・Heisenberg鎖・Schwinger模型などの物理モデル
- `random.py`: ランダムPauli Hamiltonian
- `pauli.py`: `LCP`から単項qDRIFT用`LCH`への変換

`schwinger_model()`は、開境界でGauss則を使ってゲージリンクを消去した
無次元Schwinger模型を`LCP`として返します。`num_qubits`はstaggered fermion
の格子点数、`x`はホッピング係数、`mu`はstaggered mass、
`background_field`は左端から入る背景電場です。Hamiltonian中の恒等項は
グローバル位相しか変えないためデフォルトで除外し、必要なら
`include_identity=True`で残せます。この消去後Hamiltonianには長距離の
`ZZ`相互作用が含まれます。

## グルーピング

可換グルーピングに関係する実装は`grouping/`にまとめています。

- `grouping/commuting.py`: `greedy`と量子化学向け`chemistry`グルーピング
- `grouping/operator_norm.py`: 可換Pauli和の`coefficient_l1`、`frobenius`、`lp`、`exact`ノルム推定
- `grouping/commuting_decomposition.py`: 各グループを重み付け・正規化してLCHへ変換
- `grouping/rotation_depth.py`: 可換グループの最小Pauli回転深さ

外部からは`from grouping import ...`で必要な関数を読み込みます。

## Hamiltonianの基礎データ型

Hamiltonianを表す基礎データ型は`operators/`にまとめています。

```text
operators/
├── __init__.py
├── lcp.py
└── lch.py
```

利用側では個別モジュールではなく、`operators`パッケージから`LCP`と
`LCH`を読み込みます。

## LCP: linear combinations of Pauli strings

```python
from operators import LCP

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

from grouping.commuting import decompose_into_commuting_groups

# H = sum(groups), and every group consists of mutually commuting terms.
groups = decompose_into_commuting_groups(h)

from grouping import minimum_pauli_rotation_depth

# Exact minimum under the free-Clifford, parallel-Rz cost model.
depth = minimum_pauli_rotation_depth(groups[0])
```

Pauli strings use `I`, `X`, `Y`, and `Z`; their left-most character is the
most-significant qubit. LCP coefficients are always real and are stored as
`float`. Zero coefficients and very small coefficients are kept.

## LCH: linear combinations of LCP Hamiltonians

```python
from operators import LCH, LCP

h = LCH(
    [
        (0.3, LCP({"X": 1.0}), 2.0),
        (0.7, LCP({"Z": 1.0}), 5.0),
    ]
)

# The qDRIFT decomposition, without and with per-term costs:
print(h.terms)      # [(0.3, LCP(...)), (0.7, LCP(...))]
print(h.lcp_terms)  # [(0.3, LCP(...), 2.0), (0.7, LCP(...), 5.0)]

# Flatten H = sum_j c_j H_j into one Pauli LCP.
print(h.lcp.terms)  # {"X": 0.3, "Z": 0.7}
print(h.coefficient_one_norm())  # 1.0
print(h.sampling_probabilities())  # [0.3 0.7]
print(h.evolution_costs)  # [2.0, 5.0]
print(h.to_matrix())
print(h.operator_norm())

# 外側のqDRIFT項と、その内側のPauli項を読みやすく表示。
print(h)

# 大きなLCHでは表示数と精度を指定できる。Noneなら省略しない。
print(
    h.format_decomposition(
        precision=8,
        max_terms=None,
        max_pauli_terms=None,
    )
)
```

`LCH`は`H = sum_j c_j H_j`の順序付き分解を保持し、各`H_j`は実係数の
`LCP`です。`.terms`は`(c_j, H_j)`、`.lcp_terms`はコストを加えた
`(c_j, H_j, evolution_cost)`を返します。`.lcp`は内側のPauli係数を
集約した`sum_j c_j H_j`そのものです。異なる分解は異なるqDRIFT分布を
定めるため、LCH同士の加減算では外側の項を自動統合せず、`.lcp`を作る
段階でのみ同じPauli文字列の係数が加算されます。

`coefficient_one_norm()`と`sampling_probabilities()`は、平坦化したPauli
係数ではなく外側の係数`c_j`を使用します。Hamiltonian全体は必要に応じて
CSRまたはdense行列へ変換できます。

`format_decomposition()`は各外側項について`c_j`、qDRIFTのサンプリング
確率、発展コストを表示し、内側の各PauliについてLCP係数と
`c_j * (LCP係数)`を表示します。`print(h)`は巨大なHamiltonianで出力が
膨らみすぎないよう、外側・内側とも先頭20項まで表示します。

## qDRIFT step/cost estimate

```python
from qdrift_cost import qdrift_cost
from operators import LCH, LCP

h = LCH(
    [
        (0.3, LCP({"X": 1.0}), 2.0),
        (0.7, LCP({"Z": 1.0}), 5.0),
    ]
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

## 分散を使ったqDRIFTダイヤモンド距離上限

ダイヤモンド距離に関する処理は`diamond_distance/`にまとめています。

- `variance.py`: 正規化分散`B`の数値計算と各種上限
- `variance_bound.py`: 1つのqDRIFT分解に対する有限時間上限
- `comparison.py`: 複数のqDRIFT分解を同じ目的時間発展と比較

```python
from diamond_distance import qdrift_diamond_distance_bounds
from grouping import build_grouped_lch
from hamiltonians.pauli import build_lch_from_lcp_unit_cost
from operators import LCP

target = LCP({"ZI": 1.0, "IZ": 1.0, "XX": 1.0})
pauli_qdrift = build_lch_from_lcp_unit_cost(target)
grouped_qdrift = build_grouped_lch(
    target,
    group_norm_method="exact",
)

bounds = qdrift_diamond_distance_bounds(
    target,
    total_time=0.1,
    number_of_steps=10,
    qdrift_decompositions=[pauli_qdrift, grouped_qdrift],
    variance_method="exact",
)

for name, bound in zip(("pauli", "grouped"), bounds):
    print(name)
    print(bound.normalized_variance_bound)       # B
    print(bound.variance_constant)               # Lambda^2 B
    print(bound.step_time)                       # delta = t / N
    print(bound.step_second_order_bound)         # delta^2 Lambda^2 B
    print(bound.step_taylor_remainder_bound)
    print(bound.step_diamond_distance_upper_bound)
```

分散評価法は`variance_method`で指定します。`exact`、
`contraction_bound`、`pauli_l1_bound`、`anticommuting_bound`、`sdp_bound`、
またはカスタム推定関数を使用できます。デフォルトの`contraction_bound`は
`qdrift_cost()`で`B=1`を使う標準qDRIFTステップ数に対応し、行列を生成しません。
`qdrift_diamond_distance_bounds()`の既定値は数値計算の`exact`です。

対象となる正規化分散

`||sum_j p_j (H/Lambda - H_j)^2||_op`

の数値評価または上限を`B_hat`とします。
`variance_constant = Lambda**2 * B_hat`であり、ステップ数は
`ceil(2 * variance_constant * abs(time)**2 / epsilon)`です。

`qdrift_diamond_distance_bounds()`は、`total_time=t`を
`number_of_steps=N`分割した微小時間

```text
delta = t / N
```

における1ステップの時間発展チャネルについて

```text
d_diamond(E, U) = 0.5 * ||E - U||_diamond
```

の上限を返します。この正規化では値域が`[0, 1]`なので、状態のトレース距離や
そのHaar平均と同じ尺度で比較できます。二次項は

```text
step_second_order_bound
    = (Lambda * delta)^2 * B_hat
    = (Lambda * t)^2 * B_hat / N^2
```

です。`step_taylor_remainder_bound`は`1/N^3`で減少します。
`step_diamond_distance_upper_bound`は二次項とTaylor剰余を加え、1で切った
微小時間1ステップの上限です。全時間の合成誤差は返しません。単純な三角不等式で
評価するなら、この1ステップ上限を`N`倍します。

上限保証には`B_hat`自体が真の上限であることが必要です。`exact`は指数時間の
高精度な数値評価ですが、浮動小数点計算なので厳密な数値証明ではありません。
既存の`qdrift_cost()`のステップ数は従来の保守的な
`2 * Lambda**2 * B_hat * time**2 / epsilon`をそのまま使用します。

各`qdrift_decomposition`は同じ目的Hamiltonianの厳密な分解、すなわち
`qdrift_decomposition.lcp == target`であることを仮定します。一致しない場合には
一次の表現誤差`|delta| ||target - qdrift_decomposition.lcp||_op`が生じるため、
分散だけでは誤差を上限化できません。この前提は実験設定として明記し、コード内で
重複した一致検証は行いません。

係数重みは外側の`|c_j|`です。内側の`||H_j||`は自動的には分布重みへ
吸収されません。`contraction_bound`などを使う場合は各`H_j`がHermitian
contraction、すなわち`||H_j||_op <= 1`となる分解を渡します。例えば可換和
`G_g`とそのノルム上界`h_g`に対しては、外側の係数を`h_g`、内側のLCPを
`H_g = G_g / h_g`とします。

### LCPから可換グループLCHを作る

```python
from grouping import build_grouped_lch

grouped_hamiltonian = build_grouped_lch(
    hamiltonian_lcp,
    grouping_method="greedy",
    group_norm_method="lp",
    max_group_size=12,
)
print(grouped_hamiltonian.format_decomposition())
```

この関数はまず`H = sum_g G_g`となる可換Pauliグループを作り、各グループの
作用素ノルム推定値を`h_g`として
`H = sum_g h_g (G_g / h_g)`というLCHを返します。したがって、LCHの外側係数が
グループ重み`h_g`、内側LCPが正規化Hamiltonianです。
`grouping_method`には`greedy`または`chemistry`、`group_norm_method`には
`coefficient_l1`、`frobenius`、`lp`、`exact`を指定できます。
`frobenius`では
`h_g = ||G_g||_F = sqrt(2**n * sum_P |a_P|**2)`を行列化せずに計算するため、
qDRIFT確率は`p_g`が各グループのFrobeniusノルムに比例します。

## Pauli qDRIFTと可換グループqDRIFTの比較

```bash
python main.py \
  --num-terms 40 \
  --num-qubits 5 \
  --time 1.0 \
  --epsilon 0.01 \
  --seed 42 \
  --variance-method sdp_bound \
  --grouping-method greedy \
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
`coefficient_l1`、`frobenius`、`lp`、`exact`から選べます。
`--grouping-method`は一般Hamiltonian向けの`greedy`と、Jordan--Wigner変換した
分子Hamiltonian向けの`chemistry`から選べます。

単項版では各LCPが1本の単位Pauliを表し、その1サンプルのコストはLCHに
格納されたPauli回転深さ、
グループ版は可換項を並列実行したときの厳密な最小Pauli回転深さです。
表示される総コストは、ステップ数とサンプリング分布で平均した
1ステップ深さの積です。

`contraction_bound`/`pauli_l1_bound`/`anticommuting_bound`/`sdp_bound`と
`coefficient_l1`/`lp`の組み合わせは
`2^n`次元行列を生成しません。分散評価の`exact`と
グループノルムの`exact`は行列を生成するため、大きなqubit数には適しません。`--max-group-size`は
貪欲分割の1グループに含める最大項数です。

## 量子化学HamiltonianでのqDRIFT実験

次の分子Hamiltonianプリセットから`LCP`を生成し、単項qDRIFTと
可換グループqDRIFTを
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
  --grouping-method chemistry \
  --group-norm-method lp \
  --time 1.0 \
  --epsilon 0.01 \
  --print-groups
```

量子化学CLIでは`chemistry` groupingがデフォルトです。Pauli文字列`P`に対して

```text
S(P) = {q : P[q] is X or Y}
```

を非対角サポートとし、最初に`(S(P), #Y mod 2)`が等しい項をまとめます。
これにより、Z-only項、同じ軌道対のhopping・controlled-hopping項、同じ
4軌道集合のdouble-excitation項がそれぞれ可換seed groupになります。
4軌道seedは互いに素な軌道集合から先にBaranyai型に束ね、最後に係数1-normの
大きいfragmentからfully-commuting判定によるsorted insertionでマージします。
Z-only項は常に1グループに保ちます。`--max-group-size`は非対角seedだけに適用
されるため、Z-onlyグループの項数はこの上限を超えることがあります。

LCPから直接呼ぶこともできます。

```python
from grouping.commuting import chemistry_commuting_groups

groups = chemistry_commuting_groups(
    molecular_lcp,
    max_group_size=12,
)
```

この方法は可換性と量子化学由来の構造を利用する決定的heuristicですが、
qDRIFTコストを最小化する保証はありません。従来方式と比較するときは
`--grouping-method greedy`を指定します。

Pauli係数はファイルに保存していません。実行時にプリセットの分子座標から
PySCFで分子積分を計算し、OpenFermionでJordan--Wigner変換して`LCP`を
生成します。同じ分子設定の計算結果は実行中のプロセス内でキャッシュされます。

任意の分子構造から直接`LCP`を生成することもできます。

```python
from hamiltonians.chemistry import generate_molecular_lcp

h = generate_molecular_lcp(
    [
        ("H", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, 0.90)),
    ],
    basis="sto-3g",
    multiplicity=1,
)
```

等間隔の水素鎖は、活性軌道を切らずSTO-3Gの全軌道を使用します。

```python
from hamiltonians.chemistry import hydrogen_chain_preset

h20 = hydrogen_chain_preset(20, spacing=1.0).to_lcp()
```

この設定のH20はJordan--Wigner変換前に20空間軌道、変換後に40量子ビットです。

LiHを実行する場合はプリセット名を切り替えます。

```bash
uv run python quantum_chemistry_qdrift.py \
  --hamiltonian lih_sto3g_active_jw \
  --variance-method all \
  --grouping-method chemistry \
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
  --grouping-method chemistry \
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
新しい分子Hamiltonianは`hamiltonians/chemistry.py`の
`MOLECULAR_HAMILTONIANS`へ座標・基底・活性空間のプリセットを追加できます。

## qDRIFT分散項の行列フリー上界

```python
from diamond_distance.variance import (
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

いずれも実係数の外側係数と実係数LCPサンプルからなるLCHに対する
`||sum_j p_j (H/Lambda - H_j)^2||_op` の上界です。`contraction_bound`
は入力サイズに線形、`pauli_l1_bound`は全サンプル中のPauli積を
ハッシュ集約し、
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

## 理想的な時間発展

```python
import numpy as np

from average_trace_distance.ideal_time_evolution import ideal_time_evolution
from operators import LCP

hamiltonian = LCP({"X": 0.7, "Z": -0.2})
initial_state = np.array([1.0, 0.0])
ideal_density = ideal_time_evolution(
    hamiltonian,
    time=1.0,
    initial_state=initial_state,
)
```

`ideal_time_evolution`はPauli Hamiltonianの状態ベクトルへの作用を
Qulacsの`GeneralQuantumOperator`で計算し、それをSciPyの`LinearOperator`
として`expm_multiply`へ渡して
`exp(-1j * time * H) @ initial_state`を計算します。戻り値はその純粋状態の
密度演算子を因子行列`F`による`F F^\dagger`として保持する
rank-1の`DensityOperator`です。`DensityOperator`は状態ベクトルの列
`[|psi_1>, ..., |psi_m>]`と確率分布`[p_1, ..., p_m]`を受け取り、
`sum_i p_i |psi_i><psi_i|`を表します。Hamiltonianや密度演算子の
`2**n x 2**n`行列は構築しませんが、状態ベクトル自体が長さ`2**n`なので、
時間とメモリは量子ビット数に対して指数的です。入力状態は規格化されている
必要があります。

## 1ステップqDRIFTチャネル

```python
from average_trace_distance.density_operator import trace_distance
from operators import LCH, LCP
from average_trace_distance.qdrift_channel import qdrift_channel_output

hamiltonian = LCH(
    [
        (0.7, LCP({"X": 1.0}), 1.0),
        (-0.2, LCP({"Z": 1.0}), 1.0),
    ]
)
qdrift_density = qdrift_channel_output(
    hamiltonian,
    time=1.0,
    initial_state=initial_state,
)

distance = trace_distance(qdrift_density, ideal_density)
```

`H = sum_j c_j H_j`（各`H_j`はLCP）、
`lambda = sum_j abs(c_j)`に対して、この関数は
`p_j = abs(c_j) / lambda`で枝を選び、
`exp(-1j * lambda * time * sign(c_j) * H_j)`を作用させる1ステップの
平均qDRIFTチャネルを計算します。各`H_j`内のPauli項は互いに可換である
ことを仮定し、
`exp(-1j * theta * P) = cos(theta) I - 1j sin(theta) P`
を順番に作用させる厳密なPauli回転列として実装します。回転列はQulacsの
`QuantumCircuit`として事前構築され、状態更新はQulacs上で行われます。
可換性の検査は行いません。

出力の`DensityOperator`は状態列`[psi_1, ..., psi_J]`と確率分布
`[p_1, ..., p_J]`を保持します。トレース距離を計算するときに
`F[:, j] = sqrt(p_j) * psi_j`を作るため、密度行列そのものは保持しません。
保存量は`O(J * 2**n)`、rankは高々`min(J, 2**n)`です。各`H_j`には複数の
Pauli項を含められるため、同じ関数で正規化した可換グループのqDRIFT
チャネルも表せます。

これは複数の独立なqDRIFTステップを合成した出力ではありません。`N`ステップ
では各ステップ時間を`time/N`としてチャネルを`N`回合成しますが、その密度
演算子のrankは一般に`J`以下には留まりません。

## Haar平均トレース距離

平均トレース距離に関する処理は`average_trace_distance/`にまとめています。

- `comparison.py`: 複数のqDRIFT分解を共通のHaar初期状態で比較
- `density_operator.py`: 状態アンサンブルによる密度演算子とトレース距離
- `ideal_time_evolution.py`: 理想時間発展を行列フリーに計算
- `monte_carlo.py`: Monte Carlo推定結果とHaar状態生成
- `qdrift_channel.py`: qDRIFTの1ステップ出力密度演算子
- `statistics.py`: 標本平均・標本標準偏差・標準誤差
- `variance_bound.py`: `tau(V)`と`tau(V^2)`による解析上限

数値実験の実行ファイルはプロジェクト直下に置いています。

- `average_trace_distance_comparison.py`: ランダムHamiltonianでの比較
- `molecular_average_trace_distance.py`: 量子化学HamiltonianでのMonte Carlo・解析上限比較

### 有限Haarサンプルによる推定

```python
from average_trace_distance.comparison import (
    estimate_qdrift_haar_average_trace_distances,
)

estimates = estimate_qdrift_haar_average_trace_distances(
    target_hamiltonian,
    total_time=1.0,
    number_of_steps=10,
    qdrift_decompositions=[pauli_qdrift, grouped_qdrift],
    num_initial_states=100,
    trace_distance_method="low_rank",
    seed=42,
)

for estimate in estimates:
    print(estimate.step_time)
    print(estimate.mean, estimate.standard_error)
```

`target_hamiltonian`は目的Hamiltonianを表す`LCP`、
`qdrift_decompositions`は比較したいqDRIFT分解を表す`LCH`の列です。
各LCHは

```text
qdrift_decomposition.lcp == target_hamiltonian
```

を満たすものと仮定します。この条件は実験設定として明示し、実行時には
検査しません。戻り値は入力したLCHと同じ順序です。

評価するのは全時間`t`を`N`分割した1ステップ

```text
step_time = total_time / number_of_steps
```

における誤差です。各Haar初期状態について、理想発展後の純粋状態と、
qDRIFTの全branchを混合した密度演算子とのトレース距離
`0.5 * ||rho_qdrift - rho_ideal||_1`を計算します。同じHaar状態と理想発展を
すべてのLCHで共有するため、結果は対応のある比較になります。
`number_of_steps`回分の誤差を累積した値ではありません。

### 単項qDRIFTとグループqDRIFTの比較

```python
from average_trace_distance_comparison import (
    compare_average_trace_distances,
)

comparison = compare_average_trace_distances(
    hamiltonian,
    total_time=1.0,
    number_of_steps=10,
    num_initial_states=100,
    group_norm_method="frobenius",
    max_group_size=12,
    trace_distance_method="low_rank",
    seed=42,
)
print(comparison.pauli.mean, comparison.pauli.standard_error)
print(comparison.grouped.mean, comparison.grouped.standard_error)
```

この補助関数は入力Hamiltonianから単項qDRIFTと可換グループqDRIFTのLCHを
構築し、上の複数LCH用関数へ渡します。各結果には個々のトレース距離
`values`、標本平均`mean`、標本標準偏差`sample_standard_deviation`、
標準誤差`standard_error`が含まれます。

Haarランダム状態の生成とqDRIFTのPauli回転回路はQulacsで実行します。
回転後の状態ベクトルをNumPy配列として取り出し、それ以降の低ランク密度
演算子をNumPyで保持します。理想的な非可換
Hamiltonianでは、Qulacsで`H @ state`を計算しながらSciPyの
`expm_multiply`を使うため、Trotter誤差は入りません。

既定の`low_rank`は因子行列をQR分解し、差の非零固有値を小さい縮約行列から
求めます。片方がrank 1なら負または正の固有値が高々1個しかないため、
SciPyの`eigh`で必要な最小または最大固有値だけを求めます。両方が混合状態の
場合だけ縮約行列の全固有値を求めます。
`trace_distance_method="dense"`では`FF^\dagger-GG^\dagger`をdense行列として
構築します。片方が純粋状態ならSciPyの`eigh`で必要な最小または最大固有値
だけを求め、両方が混合状態の場合だけNumPyの`eigvalsh`で全固有値を求めます。
因子数がHilbert空間次元以上の場合に適しています。

### τ(V)とτ(V²)によるHaar平均上限

```python
from average_trace_distance.variance_bound import (
    qdrift_haar_average_trace_distance_bounds,
)

bounds = qdrift_haar_average_trace_distance_bounds(
    target_hamiltonian,
    total_time=1.0,
    number_of_steps=10,
    qdrift_decompositions=[pauli_qdrift, grouped_qdrift],
)

for bound in bounds:
    print(bound.tau_v, bound.tau_v_squared)
    print(bound.normalized_average_variance_bound)
    print(bound.step_second_order_bound)
    print(bound.step_taylor_remainder_bound)
    print(bound.step_average_trace_distance_upper_bound)
```

`H = Lambda * sum_j p_j A_j`と
`V = sum_j p_j (A_j - H/Lambda)**2`に対して、正規化トレース
`tau(X) = Tr(X) / 2**n`を使います。`V = sum_P v_P P`をPauli辞書へ集約すると

```text
tau(V)   = v_I
tau(V^2) = sum_P |v_P|^2
```

なので、Hilbert空間行列を作らず`O(L^2)`時間で計算できます。共有式に対応する
二次上限は

```text
step_second_order_bound
    = (Lambda * step_time)^2
      * (tau(V) + sqrt(tau(V^2))) / 2
```

です。`normalized_average_variance_bound`は
`(tau(V) + sqrt(tau(V^2))) / 2`そのものです。
`step_average_trace_distance_upper_bound`は、各sampleのPauli係数1-normを
`r_j`として、さらに

```text
(2/3) * |Lambda * step_time|^3
      * (sum_j p_j r_j^3 + ||target_hamiltonian / Lambda||_Pauli,1^3)
```

というTaylor剰余上限を加え、トレース距離の自明な上限1で切った値です。
これも`step_time = total_time / number_of_steps`に対する1ステップ上限であり、
`number_of_steps`倍した全時間上限ではありません。

ランダムHamiltonianでコマンドライン実行する場合は次のようにします。

```bash
uv run python average_trace_distance_comparison.py \
  --num-qubits 5 \
  --num-terms 20 \
  --time 1.0 \
  --number-of-steps 10 \
  --num-initial-states 100 \
  --grouping-method greedy \
  --group-norm-method lp \
  --operator-variance-method exact \
  --max-group-size 12 \
  --seed 42 \
  --print-groups
```

`--print-groups`を付けると、グループ化qDRIFTのLCH分解を
`LCH.format_decomposition()`の形式で表示します。
出力にはMonte Carlo平均とHaar平均上限に加え、同じ`step_time`に対する
正規化ダイヤモンド距離の二次項・Taylor剰余・1ステップ上限も表示されます。

### 量子化学Hamiltonianの平均トレース距離比較

`molecular_average_trace_distance.py`は、量子化学プリセットから
Hamiltonianを動的生成し、共通のHaarランダム初期状態に対する単項qDRIFTと
グループ化qDRIFTを比較します。Monte Carlo平均・標準誤差に加えて、
`tau(V)`、`tau(V^2)`、`C`、二次項、Taylor剰余込み上限を表示します。
目的LCPと単項・グループ化LCHは一度だけ構築され、両方の計算で共有されます。

H2Oを500状態で比較する例は次の通りです。

```bash
uv run python molecular_average_trace_distance.py \
  --hamiltonian h2o_sto3g_cas_4e_4o_jw \
  --time 1.0 \
  --number-of-steps 10 \
  --mode both \
  --num-initial-states 500 \
  --grouping-method chemistry \
  --group-norm-method lp \
  --max-group-size 12 \
  --trace-distance-method low_rank \
  --seed 42
```

`--mode monte_carlo`では有限Haar標本による推定のみ、`--mode bound`では
`tau(V)`と`tau(V^2)`による解析上限のみを計算します。大きな分子で
状態ベクトル計算を行いたくない場合は`--mode bound`を使用します。

H2、LiH、H2Oは`--hamiltonian`で選択でき、水素鎖には
`--hydrogen-chain NUM_ATOMS --spacing DISTANCE`を使います。ただし状態ベクトル
の長さは`2**num_qubits`なので、大きな水素鎖のMonte Carlo計算は指数的に
重くなります。ここでも比較対象は`time / number_of_steps`の1ステップです。

## 可換Pauliハミルトニアンの作用素ノルム上界

```python
from grouping import (
    estimate_commuting_operator_norm_upper_bound,
)
from operators import LCP

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
