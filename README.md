# qDRIFT metric comparison

このプロジェクトは、SYK模型または分子Hamiltonianについて次の2方式を比較する
ためだけのコードです。

- 単項Pauli qDRIFT
- `chemistry_depth1`グループ化＋Frobenius重み付きqDRIFT

グループ`g`の重みとサンプリング確率は

```text
h_g = sqrt(sum_(P in g) a_P^2)
p_g = h_g / sum_g h_g
```

です。出力する3指標は

```text
tau(M_p) ratio              = tau(M_p)_pauli / tau(M_p)_grouped
average infidelity ratio    = infidelity_pauli / infidelity_grouped
average QPE signal ratio    = QPE_error_pauli / QPE_error_grouped
```

で、比が1より大きいほどグループ化による改善が大きいことを表します。

## 分子Hamiltonian

```bash
uv run python qdrift_metric_comparison.py \
  --hamiltonian h2o_sto3g_cas_4e_4o_jw \
  --time 0.3 \
  --number-of-steps 20 \
  --num-initial-states 50 \
  --num-trajectories 500 \
  --max-group-size 8 \
  --num-workers 8 \
  --trajectory-chunks-per-state 8 \
  --seed 42
```

利用可能な分子名は次で確認できます。

```bash
uv run python qdrift_metric_comparison.py --help
```

## SYK模型

```bash
uv run python qdrift_metric_comparison.py \
  --syk-qubits 10 \
  --syk-coupling-scale 1.0 \
  --syk-seed 42 \
  --time 0.3 \
  --number-of-steps 20 \
  --num-initial-states 20 \
  --num-trajectories 200 \
  --max-group-size 10 \
  --num-workers 8 \
  --seed 42
```

## 指標

`H = sum_j K_j`、`p_j`をqDRIFT確率、`B_j = K_j/p_j`として

```text
M_p = sum_j p_j (B_j - H)^2
```

です。`tau(M_p)`はPauli係数走査により`O(L)`で厳密に計算します。

入力`|psi>`、理想状態`|phi> = exp(-itH)|psi>`、ランダムqDRIFT軌道状態
`|chi_s>`に対して、状態と軌道をサンプルして

```text
average infidelity
  = E_psi [1 - E_s |<phi|chi_s>|^2]

average QPE signal error
  = E_psi |<psi|phi> - E_s <psi|chi_s>|
```

を計算します。QPE信号は複素軌道平均を取った後に絶対値を取ります。

## 並列化

Qulacs、OpenMP、BLASはworkerごとに1スレッドへ固定しています。Haar入力状態と
理想発展をプロセス並列化し、その後、各入力状態のqDRIFT軌道をチャンク分割して
プロセス並列化します。標準worker数は8です。逐次実行は
`--num-workers 1`で選択できます。

状態ベクトル計算量は概ね`O(K S R m_bar 2^n)`なので、小規模Hamiltonian向け
です。`tau(M_p)`だけは大規模系でも計算できますが、このCLIは指定された3指標を
常にまとめて計算します。

## PBSクラスタ（OpenPBS 20）

`pbs/submit.sh`はHaar入力状態`K`個をPBS Job Arrayへ分割します。1つのarray
subjobが1ノードを使い、そのノード内で入力状態とqDRIFT軌道をプロセス並列化
します。QPE信号については、1つの入力状態に属する全軌道を必ず同じshardで
処理してから絶対値を取ります。

96コア、約370 GBのノードを4ノード使う例は次です。既定メモリ要求はOS等の
余裕を残した350 GBです。

```bash
cd /path/to/my-project

NUM_NODES=4 \
NCPUS=96 \
MEMORY=350gb \
WALLTIME=12:00:00 \
HAMILTONIAN=h2o_sto3g_cas_4e_4o_jw \
TOTAL_TIME=0.3 \
NUMBER_OF_STEPS=20 \
NUM_INITIAL_STATES=200 \
NUM_TRAJECTORIES=500 \
MAX_GROUP_SIZE=8 \
SEED=42 \
./pbs/submit.sh
```

- `NUM_NODES`: 入力状態の総分割数かつarray subjob数（1 subjob = 1ノード）
- `NUM_WORKERS`: 各ノード内のプロセス数（既定は`NCPUS`と同じ96）
- `TRAJECTORY_CHUNKS_PER_STATE`: 1入力状態・1方式あたりの軌道チャンク数

利用可能ノード数が変わった場合は、たとえば
`NUM_NODES=2 ./pbs/submit.sh`または`NUM_NODES=8 ./pbs/submit.sh`のように提出時の
値だけ変更します。同じ`SEED`と`TRAJECTORY_CHUNKS_PER_STATE`なら、ノード数を
変更してもグローバル入力状態番号から同じ乱数を生成します。

OpenPBS 20.0.xには、新しい版の`-J 1-N%同時実行上限`がありません。そのため、
このPBS 20用スクリプトでは`NUM_NODES`とshard数を同じにしています。各subjobは
独立なので、PBSが4ノードを同時に確保できない場合でも、空いたノードから順に
実行されます。

SYK模型は次のように提出します。

```bash
SOURCE_KIND=syk \
SYK_QUBITS=12 \
SYK_COUPLING_SCALE=1.0 \
SYK_SEED=42 \
NUM_NODES=4 \
./pbs/submit.sh
```

クラスタ固有の`module load`等が必要なら、その処理を書いたファイルを
`PBS_ENV_SCRIPT=/absolute/path/to/setup.sh`で指定します。queueとaccountも
`PBS_QUEUE`、`PBS_ACCOUNT`で指定できます。

提出後に表示されるjob IDは`qstat -t JOB_ID`で確認します。すべてのsubjobが
正常終了した後、結果を検証して統合します。

```bash
uv run --frozen python qdrift_metric_comparison.py \
  --merge-partials /absolute/path/to/result-directory
```

統合処理は、全shardと全グローバル入力状態番号が重複なく揃っていること、実験
条件と`tau(M_p)`が全ノードで一致することを検証してから、3つの
Pauli/grouped比と全入力状態に対する標準誤差を出力します。

この並列化は独立な状態・軌道サンプルをノードへ分配するもので、1本の状態ベクトル
を複数ノードへ分散する実装ではありません。概算の主要メモリ下限は、1ノードの
入力状態数を`K_local`、worker数を`W`として
`16 * 2^n * (2 K_local + W)` byte程度です。大きな`n`では
`NUM_NODES`を増やして`K_local`を減らし、必要なら`NUM_WORKERS`も減らして
ください。72 qubit級の状態ベクトル計算は、この方式では実行できません。

## SYK 5–50 qubit・50 disorder realization sweep

`syk_scaling_experiment.py`は、各qubit数について独立なSYK Hamiltonianを50個
生成し、改善率をHamiltonianごとに

```text
factor = Pauli qDRIFTの値 / grouped qDRIFTの値
```

として計算します。既定の実験範囲は次のとおりです。

- 5–15 qubit: `tau(M_p)`、平均infidelity、平均QPE信号誤差
- 16–50 qubit: `tau(M_p)`のみ（状態ベクトルやqDRIFT軌道は生成しない）
- 各qubit数: 50 disorder realization
- `max_group_size`: 各qubit数と同じ値

ローカルで小さなpilot実験を行う例です。

```bash
uv run --frozen python syk_scaling_experiment.py run \
  --output-dir results/syk_pilot \
  --min-qubits 5 \
  --max-qubits 8 \
  --num-realizations 3 \
  --statevector-max-qubits 8 \
  --time 0.3 \
  --number-of-steps 20 \
  --num-initial-states 20 \
  --num-trajectories 200 \
  --num-workers 8 \
  --trajectory-chunks-per-state 8

uv run --frozen python syk_scaling_experiment.py aggregate \
  --output-dir results/syk_pilot
```

PBSクラスタで要求された全範囲を実行する場合は次です。

```bash
uv sync --frozen

NUM_NODES=8 \
NCPUS=96 \
MEMORY=350gb \
NUM_WORKERS=96 \
TAU_WORKERS=8 \
WALLTIME=48:00:00 \
TOTAL_TIME=0.3 \
NUMBER_OF_STEPS=20 \
NUM_INITIAL_STATES=20 \
NUM_TRAJECTORIES=200 \
./pbs/submit_syk_scaling.sh
```

`NUM_NODES`個のPBS array workerへ全2,300 realizationタスクを均等に分けます。
15 qubit以下では`NUM_WORKERS`を1 realization内の状態・軌道並列化に使い、
16 qubit以上では`TAU_WORKERS`個のdisorder realizationをノード内で並列化します。
50 qubitでは1 Hamiltonianが`C(100,4) = 3,921,225` Pauli項を持つため、最初は
`TAU_WORKERS=4`または`8`を推奨します。

各realizationは次のように個別保存されます。

```text
RESULT_DIR/
  configuration.json
  realizations/q005_r000.json
  realizations/q005_r001.json
  ...
```

walltimeで終了した場合、最初の提出時に表示された同じ`RESULT_DIR`を指定して
再投入できます。完成済みJSONは設定とseedを検証してskipします。ノード数は再投入
時に変更できます。

```bash
RESULT_DIR=/absolute/path/to/previous/result \
NUM_NODES=4 \
./pbs/submit_syk_scaling.sh
```

全array jobの終了後に集約します。

```bash
uv run --frozen python syk_scaling_experiment.py aggregate \
  --output-dir /absolute/path/to/result
```

生成物は以下です。

- `syk_realizations.csv`: 全realizationのPauli値、grouped値、改善率、MC標準誤差
- `syk_summary.csv`: qubit数・指標ごとの平均、標準誤差、中央値、四分位、10–90%点
- `syk_improvement_factors.png` / `.pdf`: 改善率の中央値と10–90%帯

グラフの縦軸は`Pauli / grouped`の対数軸で、1より大きいほどグループ化による改善が
大きいことを表します。既定の`time=0.3`、`R=20`、`K=20`、`S=200`は計算確認用
です。最終的な数値実験では、先にpilotでMC標準誤差を確認してから`K`と`S`を増やして
ください。分母が0になった非有限factorはraw CSVには残しますが、集約統計とグラフ
からは除外し、`finite_count`列で使用されたrealization数を確認できます。

実装確認時の参考値として、50 qubit・1 realizationの`tau(M_p)`計算は約119秒、
最大RSS約2.2 GBでした（実行環境に依存します）。15 qubitの状態ベクトル側は最小
条件`K=1, S=2, R=1`でも約54秒だったため、全条件を投入する前に5–15 qubitの
pilotでwalltimeを測ることを推奨します。
