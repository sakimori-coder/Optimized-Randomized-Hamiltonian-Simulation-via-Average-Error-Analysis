# qDRIFTの改善率の数値計算

化学HamiltonianとSYK模型について、単項Pauli qDRIFTと、可換グループを
HSノルム重みでサンプルするqDRIFTの改善率 `I_M`・`I_r`・`I_sig` を計算します。
数式と実装の対応は [NUMERICS.md](NUMERICS.md) にまとめています。

## 実行

Python 3.10.11以上とLinuxを使います。依存パッケージをインストールします。

```bash
uv sync --frozen
```

化学系は分子IDを指定して実行します。引数を省略すると下表の全10系を順に処理します。

```bash
uv run --frozen python chemistry_improvement_factors.py h2_sto3g_jw lih_sto3g_full_jw
```

| 分子ID | 量子ビット数 | 基底・軌道 |
|---|---:|---|
| `h2_sto3g_jw` | 4 | STO-3G |
| `lih_sto3g_full_jw` | 12 | STO-3G |
| `beh2_sto3g_full_jw` | 14 | STO-3G |
| `h2o_sto3g_full_jw` | 14 | STO-3G |
| `nh3_sto3g_full_jw` | 16 | STO-3G |
| `ch4_sto3g_full_jw` | 18 | STO-3G |
| `n2_sto3g_full_jw` | 20 | STO-3G |
| `h2o_ccpvdz_full_jw` | 48 | cc-pVDZ |
| `ch4_ccpvdz_full_jw` | 68 | cc-pVDZ |
| `femoco_reiher_54e_54o_jw` | 108 | CAS(54e,54o)の供給積分 |

`I_M` は全系、`I_r` と `I_sig` は16量子ビット以下で計算します。
FeMocoの入力は同梱の `hamiltonians/nitrogenase-54e-54o.fcidump` です。

SYK模型は量子ビット数を指定します。各サイズで50インスタンスを生成し、
「量子ビット数 × realization」を1つの仕事として並列計算します。
引数を省略すると5〜50量子ビットを処理します。

```bash
uv run --frozen python syk_improvement_factors.py 5 10 15
```

`I_M` は全サイズ、`I_r` と `I_sig` は15量子ビット以下で計算します。
結合スケールは `J=1`、結合定数の分散は `3! J² / (2n)³` です。

## 計算条件と結果

実行条件は各スクリプト冒頭の定数で設定します。
両スクリプトの既定値は100ステップ、100初期状態、各方式・各初期状態1000軌跡です。
発展時間はインスタンスごとに `t = 1 / sum_P |a_P|` とします。

化学系は分子を順に処理し、各初期状態の軌跡を同じノード内で並列計算します。
化学系の `NUM_WORKERS` は `len(os.sched_getaffinity(0))` で取得した利用可能な論理CPU数です。

SYKは指定した全量子ビット数とrealizationの組を、1つのプロセスプールへ投入します。
既定では46サイズ × 50 realization = 2300個の仕事です。
空いたworkerが次の組を処理するため、量子ビット数ごとの待ち合わせはありません。
各workerがHamiltonian生成・グループ化・改善率計算までを担当し、
仕事の内部では `estimate_i_r_and_i_sig(..., num_workers=1)` で逐次計算します。

SYKの `NUM_WORKERS` は同時に処理する仕事数で、化学系と同じく
`len(os.sched_getaffinity(0))` で取得した利用可能な論理CPU数を使います。

両スクリプトとも起動時に `QULACS_NUM_THREADS=1` を設定し、各workerのQulacsを1スレッドで実行します。

結果と計算条件を次のCSVに1インスタンスずつ保存します。実行ごとに上書きします。

- `results/chemistry_improvement_factors.csv`
- `results/syk_improvement_factors.csv`

SYKのCSVは親プロセスが指定した量子ビット数の順、realization番号順に書き込みます。
状態ベクトルを計算しないサイズでは、`I_r` と `I_sig` は空欄です。
改善率はすべて単項版の値をグループ版の値で割った比で、1より大きいと改善を表します。
SYKの曲線をまとめる場合は、各量子ビット数についてインスタンスごとの改善率の中央値を取ります。

化学系のseedは42です。SYKでは基準seed 42から結合定数用とサンプリング用の
別々のseedを作り、両方をCSVに記録します。
SYKではseedとサンプル数を固定すると、仕事の並列数や量子ビット数の指定順を変えても
同じ乱数標本で計算します。化学系は軌跡worker数も固定します。

## Pythonから利用する

Hamiltonian生成は `LCP`、グループ化は `LCH` を返します。
`LCP` はPauli係数の辞書、`LCH` は `H = sum_j H_j` の各 `H_j` を保持します。

```python
import os
os.environ["QULACS_NUM_THREADS"] = "1"

from hamiltonians import chemistry, syk
from grouping import build_fermionic_lch
from improvement_factors import calculate_i_m, estimate_i_r_and_i_sig

H = chemistry.generate("h2_sto3g_jw")
# SYKの場合: H = syk.generate(num_qubits=5, seed=42)
H_grouped = build_fermionic_lch(H)
t = 1 / sum(abs(a) for a in H.terms.values())

I_M = calculate_i_m(H, H_grouped)
I_r, I_sig = estimate_i_r_and_i_sig(
    H, H_grouped, total_time=t, number_of_steps=100,
    num_initial_states=20, num_trajectories=200, seed=42, num_workers=8,
)
```

`calculate_i_m` は係数だけから計算します。
`estimate_i_r_and_i_sig` は各Haar初期状態の理想発展を1回計算し、
同じ入力・理想状態を使って単項版とグループ版のqDRIFT軌跡をサンプルします。
`num_workers=1` なら逐次実行です。

個別の状態発展も、Qulacsの `QuantumState` を渡して計算できます。
両関数とも入力をコピーして使い、発展後の新しい状態を返します。

```python
from qulacs import QuantumState
from ideal_time_evolution import ideal_time_evolved_state
from qdrift_trajectory import sample_qdrift_state

initial = QuantumState(H.num_qubits)
initial.set_Haar_random_state(42)
ideal = ideal_time_evolved_state(H, time=t, initial_state=initial)
sampled = sample_qdrift_state(H_grouped, t, 100, initial, rng=42)
```

## ファイル構成

| ファイル | 役割 |
|---|---|
| [chemistry_improvement_factors.py](chemistry_improvement_factors.py) | 化学系の実験とCSV出力 |
| [syk_improvement_factors.py](syk_improvement_factors.py) | SYK模型の実験とCSV出力 |
| [improvement_factors.py](improvement_factors.py) | `I_M`・`I_r`・`I_sig` の計算 |
| [grouping.py](grouping.py) | フェルミオン系の可換グループ分割 |
| [qdrift_trajectory.py](qdrift_trajectory.py) | qDRIFT軌跡のサンプルと状態発展 |
| [ideal_time_evolution.py](ideal_time_evolution.py) | 理想時間発展 |
| `hamiltonians/` | 化学系・SYKの生成と共通Jordan–Wigner変換 |
| `operators/` | `LCP`・`LCH` と正規化HSノルム |
| `tests/` | 密行列計算などによる数値検証 |

## 検証

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 QULACS_NUM_THREADS=1 \
  uv run --frozen --extra test python -m pytest -q
```

小規模系でJordan–Wigner変換、グループの保存性・可換性・独立性、
理想時間発展、qDRIFT軌跡、3つの改善率を検証します。
OpenFermionは化学Hamiltonianの照合テストで使います。
