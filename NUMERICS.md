# 論文の式と数値計算

単項Pauli qDRIFTと、可換グループをHSノルム重みでサンプルするqDRIFTを比較します。
対応する原稿は作業ディレクトリの `../thesis/main.tex` です。
以下では改稿で変わる式番号の代わりに、そのTeXラベルを記します。

## 読む順序

| 計算 | 原稿のラベル | 実装の入口 |
|---|---|---|
| 化学系の実験 | `tab:appendix-molecular-settings` | [chemistry_improvement_factors.py](chemistry_improvement_factors.py): `main` |
| SYK模型の実験 | `eq:appendix-syk-hamiltonian` | [syk_improvement_factors.py](syk_improvement_factors.py): `main` |
| 化学Hamiltonianの生成 | `tab:appendix-molecular-settings` | [hamiltonians/chemistry.py](hamiltonians/chemistry.py): `generate` |
| SYK Hamiltonianの生成 | `eq:appendix-syk-hamiltonian` | [hamiltonians/syk.py](hamiltonians/syk.py): `generate` |
| 共通Jordan–Wigner変換 | 分子の第二量子化Hamiltonian、Majorana積 | [hamiltonians/jordan_wigner.py](hamiltonians/jordan_wigner.py): `majorana_product`、`from_spatial_integrals` |
| 可換かつ二進線形独立なグループへの分割 | `app:grouping` | [grouping.py](grouping.py): `build_fermionic_lch` |
| グループ重みと確率 | `eq:grouped-pauli-hs-distribution` | [operators/lch.py](operators/lch.py): `hs_norms`、`sampling_probabilities` |
| 改善率 `I_M` | `eq:predicted-improvement-factor` | [improvement_factors.py](improvement_factors.py): `calculate_i_m` |
| 改善率 `I_r`・`I_sig` | `eq:actual-improvement-factors` | [improvement_factors.py](improvement_factors.py): `estimate_i_r_and_i_sig` |
| qDRIFTの状態発展 | `eq:random-product-formula` | [qdrift_trajectory.py](qdrift_trajectory.py): `sample_qdrift_state` |
| 理想時間発展 | `eq:random-product-formula` | [ideal_time_evolution.py](ideal_time_evolution.py): `ideal_time_evolved_state` |

`grouping.py` は `chemistry.generate()` と `syk.generate()` が返すJWハミルトニアンを対象にします。
4サイトにX/Yを持つ項は、論文の `A_even, B_even, A_odd, B_odd` を書いた `_PACKET` で分けます。
Z列は生成側で決まる第1・第2端点間と第3・第4端点間のJWパリティ列を前提にします。
パケットを係数の絶対値和の降順に並べ、端点が重ならないものを同じグループに詰めます。
残りの項はX/Yの位置とYの個数の偶奇で分け、二進消去で独立性を確認しながら詰めます。
探索候補数はパケット側が各サイズ16グループ、残りの項が8グループです。

## 記号とデータの表現

| 論文の記号 | コードの名前 |
|---|---|
| `N`: 時間ステップ数 | `number_of_steps` |
| `S`: Haar入力状態数 | `num_initial_states` |
| `R`: 各入力・各方式の軌跡数 | `num_trajectories` |

`LCP` は実係数Pauli和 `sum_P a_P P` を保持します。
`LCP.hs_norm()` は正規化HSノルム `sqrt(Tr(H**2) / 2**n)` を返します。
`LCH` は論文と同じく `H = sum_j H_j` の分解です。
`terms[j]` は元のPauli係数を含む `H_j` そのもの（`LCP`）で、
`lcp` は全項の和 `H` です。例えば

```python
H = LCH([LCP({"XI": 3.0, "IZ": -4.0}), LCP({"ZZ": 2.0})])
```

は `H_0 = 3 XI - 4 IZ`, `H_1 = 2 ZZ` を保持します。
各項を事前に正規化したり、外側の係数と演算子に分けたりしません。
単項版も `H_j = a_P P` を保持します。

実験で使うHSノルム重みと確率は、この分解から計算します。

```text
h_j = sqrt(tau(H_j**2)) = sqrt(sum_(P in G_j) a_P**2)
Lambda = sum_j h_j
p_j = h_j / Lambda
B_j = H_j / p_j
```

`hs_norms()`、`lambda_sum()`、`sampling_probabilities()` が
それぞれ `h_j`、`Lambda`、`p_j` に対応します。
ここでのHSノルム重みは `sqrt(Tr(H_j**2) / 2**n)` であり、
通常の非正規化Frobeniusノルムとは `sqrt(2**n)` の因子が異なります。
作用素ノルムの上界であるという仮定は置きません。

分散演算子 `M_p = sum_j p_j (B_j - H)**2` は、論文の `D_p` と同じです。
`tau(M_p) = Tr(D_p) / d`, `d = 2**n` と対応します。
Pauli直交性から、密行列を作らずに

```text
tau(M_p) = sum_j tau(H_j**2) / p_j - tau(H**2)
         = Lambda**2 - sum_P a_P**2
```

を求めます。最後の等号では上の重みを使い、`a_P` は全項を合算した `H` のPauli係数です。
比較する改善率はすべて `Pauliの値 / groupedの値` です。
`I_M` は両方式で同じ `t, N` を使うときの解析上界の比で、
サンプルから求める `I_r` や `I_sig` と必ず一致する量ではありません。

## 時間発展と平均の取り方

両方式で共通の `t = 1 / sum_P |a_P|` と `delta = t / N` を使います。
各ステップで独立にグループ `j` を確率 `p_j` で選び、
`exp(-i delta H_j / p_j)` を入力状態に作用させます。
群内のPauli項は可換なので、Pauli回転の積による評価に群内Trotter誤差はありません。
Qulacsの各Pauli回転角は `-2 * delta * a_P / p_j` です。
Pauli文字列の左端を状態ベクトルの最上位ビットとします。
SYKのMajoranaモードは文字列の左から番号を付けます。Qulacsのqubit番号とは
逆順ですが、これは固定したqubitの置換であり、係数ノルムやHaar平均を変えません。
SYKの各Majorana四重積は共通の `majorana_product` で係数 `±1` のPauli項へ
変換し、乱数結合 `J_abcd` を掛けます。X・Z成分を整数のビットで表し、
反交換による位相を追跡します。SYKの結合定数には打ち切りを入れません。

固定した入力状態 `psi` と理想状態 `phi = exp(-i t H) psi` に対して、

```python
fidelities = [abs(vdot(phi, chi_b))**2 for chi_b in trajectories]
signals = [vdot(psi, chi_b) for chi_b in trajectories]
infidelity = 1 - mean(fidelities)
signal_error = abs(vdot(psi, phi) - mean(signals))
```

を計算し、その後で入力状態に関する平均を取ります。
ノード内で軌道をプロセス並列化した場合も、複素信号の部分和と軌道数を集めてから
絶対値を取ります。各軌道で求めた絶対誤差の平均では、論文とは異なる量になります。

有限軌道数のinfidelity推定量は不偏ですが、信号誤差推定量には一般に上向きの
バイアスがあります。
軌道数と入力状態数を増やした収束確認は、小規模系の実装検証とは別に必要です。

実験の前提は次のとおりです。

- 分子の同一Pauli項を合算後、`|a_P| <= 1e-12` を除き、恒等項も除きます。
  ノルム・確率・理想発展・qDRIFTはすべて同じ恒等項除去後のHamiltonianを使います。
- Haar状態は全 `2**n` 次元から生成します。固定電子数の部分空間には限定しません。
  両方式で同じ入力・理想状態を使い、軌道は方式ごとに独立に生成します。
- 深さ1の根拠はグループ内の可換性と二進線形独立性です。
  コードはその発展を計算しますが、深さ1のClifford回路の合成や配線は行いません。
  論文の資源比較は `R_z` 層数です。
- 理想発展は `expm_multiply`、軌道は倍精度の状態ベクトルで評価します。
  測定ショットの雑音は加えません。

FeMocoの入力は同梱の `hamiltonians/nitrogenase-54e-54o.fcidump` です。
SHA-256: `0ed675b5bc6e002549d61589ca17a5baa705f5dba51ea52da40690e543106d7f`。
通常の分子はPySCFのRHF軌道から1電子積分と2電子積分を求めます。
FeMocoはFCIDUMPから与えられた積分を読み込みます。どちらも実数の空間軌道積分
`h[p,q]`、`g[p,q,r,s] = (pq|rs)` を同じ `from_spatial_integrals` に渡します。

共通変換では、スピンを足し合わせた密度演算子を

\[
E_{pq}=\sum_\sigma a^\dagger_{p\sigma}a_{q\sigma},\qquad
P_{pq}=\frac{E_{pq}+E_{qp}}{2}-\delta_{pq}I
\]

と定義します。`P_pq` 自体もSYKと同じMajorana積の関数で求めます。
`E_pq E_rs` の反交換関係と積分の8重対称性から、恒等項を除いて

\[
H=\sum_{pq}\widetilde h_{pq}P_{pq}
+\frac12\sum_{pqrs}(pq|rs)P_{pq}P_{rs},\qquad
\widetilde h_{pq}=h_{pq}+\sum_r(pq|rr)-\frac12\sum_r(pr|rq)
\]

と書けます。`p >= q` の空間軌道対だけを列挙し、非対角対には重み2を付けます。
異なる対は順序交換した項もまとめ、同じ対の積には係数 `1/2` を残します。
反可換なPauli積は順序交換で打ち消されます。全Pauli係数を合算した後だけ、
上記の `1e-12` 条件を適用します。途中の積分やMajorana係数には打ち切りを入れません。

## 数値検証

[README.md](README.md) のコマンドで `tests/` を実行します。
[tests/test_jordan_wigner.py](tests/test_jordan_wigner.py) は占有数基底の演算子との照合、
[tests/test_grouping.py](tests/test_grouping.py) は係数の保存とグループの可換性・独立性を検証します。
[tests/test_hamiltonians.py](tests/test_hamiltonians.py) は分子の設定・Hartree–Fockエネルギー・
OpenFermionとの係数照合、およびSYKと密なMajorana行列の照合を行います。

[tests/test_ideal_time_evolution.py](tests/test_ideal_time_evolution.py) と
[tests/test_qdrift_trajectory.py](tests/test_qdrift_trajectory.py) は、2量子ビットの
密行列指数関数と状態発展を比較します。恒等項の位相、Y、負の係数、入力状態の保存も確認します。
[tests/test_improvement_factors.py](tests/test_improvement_factors.py) は、密行列から得た分散と、
同じ乱数で生成した密行列の軌跡から、3つの改善率を独立に照合します。
1プロセス・複数プロセス・軌跡数よりworker数が多い場合を含みます。
実験スクリプトでは、化学系の逐次処理・軌跡並列、SYKの量子ビット数とrealizationを
またぐ並列実行、CSV出力を検証します。SYKでは仕事の並列数と量子ビット数の指定順を
変えても、seedとサンプリング結果が変わらないことを確認します。

## 実行条件

実行方法は [README.md](README.md) を参照してください。
現在の両スクリプトの既定値は `N=100, S=100, R=1000`、基準seed 42です。
原稿に記載されている `S=20, R=200` で計算する場合は、各スクリプト冒頭の
`NUM_INITIAL_STATES` と `NUM_TRAJECTORIES` を変更します。

化学系は全10系、SYKは5〜50量子ビットで各50インスタンスを処理します。
`I_M` は全サイズ、状態ベクトルによる `I_r`・`I_sig` は化学系16量子ビット以下、
SYK15量子ビット以下で計算します。SYKでは各インスタンスの比を計算してから、
量子ビット数ごとに中央値を取ります。

化学系はインスタンスを順番に処理し、各初期状態の軌跡を `num_workers` 組に分けて
同じノード内で並列計算します。部分和を集めてから軌跡平均を求めます。

SYKは「量子ビット数 × realization」を1つの仕事として、全サイズを共通の
プロセスプールへ投入します。`NUM_WORKERS` は同時に処理する仕事数です。
各仕事では初期状態と軌跡を逐次計算します。量子ビット数ごとの待ち合わせはありません。
CSVは指定した量子ビット数の順、realization番号順に親プロセスだけが書き込みます。

初期状態のseedは `seed + state_index`、軌跡用のseedは `np.random.default_rng(seed)` で生成します。
SYKではHamiltonian用とサンプリング用のseedを分けます。
SYKの仕事内の軌跡worker数は1で固定されているため、仕事の並列数や実行順を変えても
乱数標本は変わりません。化学系で同じ標本を再現するには、seed・サンプル数・
軌跡worker数を固定します。
結果と実行条件はCSVに記録します。`results/` はGit管理から除外されています。
