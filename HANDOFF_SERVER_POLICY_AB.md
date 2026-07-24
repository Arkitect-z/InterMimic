# Theia S1 Raw-vs-Full policy 实验服务器 HANDOFF

更新时间：2026-07-24（Asia/Shanghai）

## 0. 服务器 agent 的任务边界

本仓库已经提供数据转换、冻结/验证、单 run 训练与恢复、严格评测、统计汇总和
LaTeX 表格导出。服务器 agent 只需要：

1. 填入服务器真实路径和 Conda 环境；
2. 根据 GPU 型号/显存实测选择共同的目标环境数；
3. 编写或优化 **8 个独立进程**的 GPU 调度脚本；
4. 依次通过 CPU preflight 和 GPU smoke gate 后启动正式 run；
5. 运行现成聚合器并回传 `results.md` 和完整结果目录。

不能由调度脚本改变的实验契约：

- conditions：`raw`、`full`；
- training seeds：`0 1 2 3`；
- Stage A / B：20,000 / 2,000 epochs；
- 每条 reference 的正式评测 trial 数：`K=10`；
- Raw/Full 同一 training seed 使用同一个 evaluation seed；
- 主结果只使用 epoch 22,000 final checkpoint；
- 训练从随机权重开始，不加载本机或单序列 policy；
- 两组必须使用同一个 frozen `policy_ab_manifest.json`；
- 两组共享物体轨迹、接触标签、contact reward/termination 和全部 PPO 配置；
  唯一实验自变量是 Raw 与 tactile-refined 人体/手部参考运动；
- 主表只报告 InterMimic 的 Succ.、Duration、\(E_h\)、\(E_o\)。

`run_theia_server.sh` 是旧的单 condition 工具，**不能用于这次论文 A/B**。

## 1. 两个 Git 仓库都必须同步

InterMimic 是父 Theia 仓库中的独立 Git 仓库。本流程跨两个仓库：

```text
Theia/toolkit/scripts/theia2intermimic.py
Theia/thirdparty/InterMimic/
```

只同步 InterMimic 会缺少 Raw/Full converter。服务器开始前记录：

```bash
git -C /path/to/Theia rev-parse HEAD
git -C /path/to/Theia/thirdparty/InterMimic rev-parse HEAD
git -C /path/to/Theia status --short
git -C /path/to/Theia/thirdparty/InterMimic status --short
```

正式运行时源码应 clean；若因紧急修复无法 clean，必须保存两个仓库的
`git diff --binary`，并保证修复对 Raw/Full 和全部 seeds 同时生效。不能在一半
run 完成后只更新另一半。

## 2. 科学问题与唯一自变量

比较：

- `raw`：`smplx_humanoid_motion.npy`
- `full`：Setting-1 measured-tactile full refinement，
  `smplx_humanoid_motion_refined.npy`

两组共享 sequence IDs、帧索引、object trajectory、measured-contact schedule、
对象资产、场景 Z shift、训练/评测配置和预算。唯一变化是 humanoid/hand
reference kinematics。这个实验支持“完整 refinement pipeline 提高 downstream
policy learnability”，不能单独归因于 tactile，也不能称为通用 task success。

## 3. 必备环境与路径

下面用占位符：

```bash
THEIA_ROOT=/server/path/Theia
INTERMIMIC_ROOT=$THEIA_ROOT/thirdparty/InterMimic
SOURCE_ROOT=/server/path/to/S1_sequence_directories
OBJECTS_ROOT=$THEIA_ROOT/data/objects
POLICY_DATA_ROOT=/server/experiments/theia_policy_ab_data
EXPERIMENT_ROOT=/server/experiments/theia_policy_ab_runs
CONDA_ENV=intermimic
```

先确认：

```bash
conda run --no-capture-output -n "$CONDA_ENV" python - <<'PY'
from isaacgym import gymapi
import h5py
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.is_available(), torch.cuda.device_count())
PY
nvidia-smi
```

训练需要 CUDA 和 Isaac Gym；转换/统计需要 `torch, h5py, scipy, trimesh, numpy`。

## 4. 冻结候选 S1 清单

不要在结果产生后按 policy 表现删序列，也不要把本地文档中的 94 或论文旧稿的
120 当作服务器事实。依据仓库 `docs/dataset_id.md`，正式 S1 只包含同高度
`L11/L22/L33/L44/L55` 和空间变体 `V01/V02/V03`。先按服务器实存目录生成并
冻结排序后的规范候选清单：

```bash
python - "$SOURCE_ROOT" /server/manifests/s1_policy_candidates.txt <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2])
pattern = re.compile(r"S1L([1-5])\1P\d{2}T\d{4}V0[1-3]")
paths = sorted(
    path.resolve()
    for path in root.iterdir()
    if path.is_dir() and pattern.fullmatch(path.name)
)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("".join(f"{path}\n" for path in paths))
print(f"Frozen {len(paths)} canonical S1 candidates in {output}")
PY
N_CANDIDATES=$(wc -l < /server/manifests/s1_policy_candidates.txt)
test "$N_CANDIDATES" -gt 0
```

另行审计未进入清单的 `S1*` 目录，确认它们确实是 V04、非规范/旧命名或不属于
论文 S1 协议，保存这份审计记录。若规范序列的附加 action/task 元数据不完整，
可准备一份在 rollout 前冻结的 CSV/JSON：

```text
reference_id,height,variation,action,task
...
```

`height` 只能与 ID 中同高 level 一致，`variation` 只能是 V1--V3。不能用
metadata 把非规范 level/variation 改名后纳入，也不能运行后猜测姿态桶。

## 5. 一次性生成 paired 数据

必须在 `intermimic` 环境中运行：

```bash
cd "$INTERMIMIC_ROOT"
conda run --no-capture-output -n "$CONDA_ENV" \
  python isaacgym/scripts/prepare_theia_policy_ab.py \
  --source-root "$SOURCE_ROOT" \
  --sequence-list /server/manifests/s1_policy_candidates.txt \
  --objects-dir "$OBJECTS_ROOT" \
  --output-root "$POLICY_DATA_ROOT" \
  --expected-count "$N_CANDIDATES"
```

目录名缺 L/V 时追加：

```bash
--reference-metadata /server/manifests/s1_policy_metadata.csv
```

转换器会：

- 强制显式选择 Raw 或 refined，不 fallback；
- 保留真实 `P<number>` subject；
- 先分别计算两组所需 ground shift，再使用两者较大的共同 shift 重写；
- 强制 Raw/Full shape、frame indices、object/contact source 一致；
- 强制 tensor 的 `318:386`（object pose + contact）逐位相同；
- 只把完整成功的 pair 放入 `eligible/raw` 和 `eligible/full`；
- 输出数据、converter 和 object asset SHA-256。

主要产物：

```text
policy_ab_manifest.json
eligible_pairs.csv
excluded_pairs.csv
eligible/raw/*.pt
eligible/full/*.pt
data_hashes_raw.txt
data_hashes_full.txt
asset_hashes.txt
metadata/
conversion_logs/
```

若存在技术排除，首次命令会在写完清单后非零退出。先人工检查
`excluded_pairs.csv`。只有缺文件、损坏、缺资产、非双手双物体或 schema
不支持等预注册技术原因可接受；不能按 motion quality 或 rollout 成功率排除。
接受后不必重做转换，在下一节显式设置 `ACCEPT_EXCLUSIONS=1`。若要从头重转，
使用新的空 `POLICY_DATA_ROOT`，不要把新旧 staging 混合。

## 6. 解析正式 env/minibatch

每个 environment 固定绑定 `env_id % N` 对应的 reference。为了每条 reference
训练副本完全相等，同时保持原 recipe 的每 epoch 四个 PPO minibatches：

```bash
N=$(python - "$POLICY_DATA_ROOT/policy_ab_manifest.json" <<'PY'
import json, sys
print(len(json.load(open(sys.argv[1]))["references"]))
PY
)
TARGET_ENVS=2048
REPLICAS=$((TARGET_ENVS / N))
test "$REPLICAS" -ge 1
NUM_ENVS=$((N * REPLICAS))
MINIBATCH_SIZE=$((NUM_ENVS * 8))
echo "N=$N replicas=$REPLICAS envs=$NUM_ENVS minibatch=$MINIBATCH_SIZE"
```

因为 horizon 为 32，PPO batch 是 `32*NUM_ENVS`；上式令 minibatch 为其
四分之一。例如 N=94 时是 1974 env / 15792 minibatch。

服务器 agent 可根据 24 GiB 显存修改 `TARGET_ENVS`，但最终必须满足：

- `NUM_ENVS=N*k` 且 `k>=1`；
- `(32*NUM_ENVS) % MINIBATCH_SIZE == 0`；
- `MINIBATCH_SIZE % 4 == 0`（当前 `seq_len=4`）；
- Raw/Full 和 4 seeds 使用相同的解析值；
- smoke 与正式 run manifest 记录实际值。

不要追求 `nvidia-smi` 显示 100% memory。应在完成至少 200 个 epoch 的 warmup
后选择不会 OOM 的最大 `N*k`，为 PhysX、checkpoint 保存和临时 tensor 保留约
1.5--2 GiB。建议以 `N*floor(target/N)` 从低到高试探；所有 8 个正式 worker
冻结为同一个最终值。训练代码默认关闭逐 step scalar diagnostics，并把资源监控
交给外部调度器，避免用日志换取显存/FPS。

## 7. CPU preflight（第一道硬门）

```bash
cd "$INTERMIMIC_ROOT"
ACCEPT_EXCLUSIONS=0 \
NUM_ENVS="$NUM_ENVS" \
MINIBATCH_SIZE="$MINIBATCH_SIZE" \
conda run --no-capture-output -n "$CONDA_ENV" \
  bash isaacgym/scripts/preflight_theia_policy_ab.sh "$POLICY_DATA_ROOT"
```

若已人工接受预注册技术排除，改为 `ACCEPT_EXCLUSIONS=1`。必须看到：

```text
CPU preflight passed
PRECHECK_READY.json
```

该门会复算 pair/data/asset hash、检查 `[T,594]`、finite/quaternion/contact、
双手接触、足部碰撞和 Raw/Full 逐列配对。失败时不能关闭 validator。
`policy_ab_validation.json` 还会记录每条 pair 的
`mean_body_position_delta_cm` 和 `contact_hand_position_delta_cm`，并在整个
Raw/Full 人体参考集合逐位相同时直接失败。这两个字段只用于训练前确认 tactile
refinement 确实进入了 policy 数据，不参与筛选序列或事后调参。

## 8. GPU smoke（第二道硬门）

先在不会与正式结果混用的目录运行：

```bash
CUDA_VISIBLE_DEVICES=<one_gpu> \
NUM_ENVS="$NUM_ENVS" \
MINIBATCH_SIZE="$MINIBATCH_SIZE" \
ACCEPT_EXCLUSIONS="${ACCEPT_EXCLUSIONS:-0}" \
CONDA_ENV="$CONDA_ENV" \
bash isaacgym/scripts/smoke_theia_policy_ab.sh \
  "$POLICY_DATA_ROOT" \
  "$EXPERIMENT_ROOT/_smoke"
```

默认对 Raw/Full 各执行：

- 50 epoch fresh；
- 从完整 checkpoint 恢复后再跑 50 epoch；
- 1 epoch Stage-B full-sequence fine-tune；
- 全部 references 的 K=10 正式 evaluator 完整性与显存测试。

它会检查 checkpoint epoch、NaN/Inf、正 FPS、完整评测 cohort，以及 Raw/Full
FPS 比不低于 0.70。必须得到：

```text
_smoke/SMOKE_READY.json
```

若服务器 agent 为节省首次管线调试时间，可先用
以下独立目录验证代码连通性：

```bash
CUDA_VISIBLE_DEVICES=<one_gpu> \
NUM_ENVS="$NUM_ENVS" \
MINIBATCH_SIZE="$MINIBATCH_SIZE" \
ACCEPT_EXCLUSIONS="${ACCEPT_EXCLUSIONS:-0}" \
CONDA_ENV="$CONDA_ENV" \
SMOKE_BOOTSTRAP_EPOCHS=2 \
SMOKE_FIRST_CHUNK_EPOCHS=1 \
SMOKE_EVAL_K=1 \
bash isaacgym/scripts/smoke_theia_policy_ab.sh \
  "$POLICY_DATA_ROOT" "$EXPERIMENT_ROOT/_smoke_quick"
```

快速 smoke 与正式 smoke **绝不能复用输出目录**。脚本会冻结
`smoke_spec.txt` 并拒绝混用不同参数。正式 8-run 启动前，仍需在全新的
`"$EXPERIMENT_ROOT/_smoke"` 完成默认 100-epoch、K=10 smoke。

## 9. 单 run 正式入口

接口：

```bash
NUM_ENVS="$NUM_ENVS" \
MINIBATCH_SIZE="$MINIBATCH_SIZE" \
CONDA_ENV="$CONDA_ENV" \
bash isaacgym/scripts/run_theia_policy_seed.sh \
  CONDITION SEED CONDITION_DATA_DIR \
  "$POLICY_DATA_ROOT/policy_ab_manifest.json" \
  "$EXPERIMENT_ROOT"
```

示例：

```bash
NUM_ENVS="$NUM_ENVS" \
MINIBATCH_SIZE="$MINIBATCH_SIZE" \
CONDA_ENV="$CONDA_ENV" \
bash isaacgym/scripts/run_theia_policy_seed.sh \
  raw 0 "$POLICY_DATA_ROOT/eligible/raw" \
  "$POLICY_DATA_ROOT/policy_ab_manifest.json" "$EXPERIMENT_ROOT"
```

该入口自动：

1. 复算 condition 数据 hash 并对照 frozen manifest；
2. 冻结 run spec；
3. 从随机权重完成 20k Hybrid/RSI；
4. 完整恢复 optimizer/normalizer/epoch/frame 后完成 2k Start/full-sequence；
5. 保留固定 milestone；
6. 对 final checkpoint 执行每 reference 恰好 K=10 的完整 cohort；
7. 输出 reference-level 四项 InterMimic 指标。

重复同一命令会读取 checkpoint 真实 epoch，只补剩余预算。若 dataset、
condition、seed、预算、配置或源码 fingerprint 改变，会拒绝把不兼容状态混入
同一 run root。入口默认还会硬性拒绝 seeds 0--3、20k+2k epochs、K=10、
30 FPS 和 `EVAL_SEED=10000+training_seed` 之外的参数；仅隔离的代码微测可显式
设置 `ALLOW_NONFORMAL_PROTOCOL=1`，这类目录禁止进入正式聚合。

评测先写入 `.attempt.*` 临时目录，所有 CSV/JSON 和哈希验证成功后才发布到
`evaluation/final`，且 `validation.json` 最后落盘。中断的 attempt 会保留用于
排障，不会被当成有效结果；不要手工复制或编辑正式评测文件。

## 10. 多 GPU 调度脚本的要求

8×RTX 4090 的首选拓扑是 8 卡同波，每张卡运行一个完整数据集 policy：

```text
raw/seed_0   full/seed_0
raw/seed_1   full/seed_1
raw/seed_2   full/seed_2
raw/seed_3   full/seed_3
```

这里的并行单位是 `condition × seed`，不是 sequence。禁止把不同 S1 序列分给
不同 GPU 独立训练；那会得到 8 个只覆盖子集的 policy，而不是一个覆盖全部 S1
的 policy。若主机 CPU/RAM 或 PCIe 使 8 路吞吐下降，再改为 4 卡两波或更少
GPU 多波。

调度器必须：

- 为每个 worker 设置清晰的 `CUDA_VISIBLE_DEVICES`；
- 不在 worker 间共享 checkpoint/output path；
- 传播相同 `NUM_ENVS`、`MINIBATCH_SIZE`、epoch 和 K；
- 捕获每个 worker exit code；任一失败则总 job 非零；
- 允许重新运行失败 worker，不能删除已完成的其他 run；
- 监控主机 RAM、GPU memory、温度、利用率和每 run FPS；
- 8 路并发 FPS 若低于单路 smoke 的 70%，改为分波，不改变实验预算；
- 不使用 DDP 把一个 policy 跨多卡训练，除非另行完整验证；本实验的自然并行
  单位是 condition × seed。

服务器调度脚本本身也应保存到实验目录并记录 SHA-256。

## 11. Checkpoint 与评测产物

每个 run：

```text
<condition>/seed_<n>/
  run_spec.txt
  data_manifest.json
  policy_seed.log
  bootstrap/theia_smplx/nn/
    mimic.pth
    mimic_epoch_00002000.pth
    mimic_epoch_00005000.pth
    mimic_epoch_00010000.pth
    mimic_epoch_00015000.pth
    mimic_epoch_00020000.pth
  finetune/theia_smplx/nn/
    mimic.pth
    mimic_epoch_00022000.pth
  evaluation/final/
    manifest.txt
    eval.log
    episodes.csv
    episode_summary.json
    termination_causes.csv
    per_reference.csv
    validation.json
    summary.json
```

训练期每 250 epoch 覆盖保存 `mimic.pth` 供断点恢复，并保留上述固定
milestone；正式配置关闭了约 134 MB 的 reward-best 反复写盘。主表禁止选择
reward-best/test-best。

若需要 learning curve，可用同一个正式 evaluator 对固定 milestone 运行，并把
结果写入 `evaluation/milestone_<epoch>/`；不能根据中间 test 结果选择主
checkpoint。主审稿人要求的最低 Raw-vs-Full 结果只依赖 final 22k。

## 12. 一键聚合并直接出表

8 个 final evaluation 全部成功后：

```bash
cd "$INTERMIMIC_ROOT"
conda run --no-capture-output -n "$CONDA_ENV" \
  python isaacgym/scripts/aggregate_theia_policy_ab.py \
  --experiment-root "$EXPERIMENT_ROOT" \
  --pair-manifest "$POLICY_DATA_ROOT/policy_ab_manifest.json"
```

聚合器严格要求 `{raw,full} × seeds {0,1,2,3} × same references × K=10`，
并交叉核验 8 个 `run_spec.txt` 的训练预算、env/minibatch、Git/source hashes、
评测 pipeline 及 episodes/per-reference/summary artifact hashes。随后使用
reference 等权、4-seed sample std，并做 10,000 次固定 RNG 的 crossed paired
bootstrap。输出：

```text
results/results.md
results/per_reference_all.csv
results/per_seed.csv
results/paired_results.csv
results/bootstrap_results.json
results/summary.json
results/main_table.tex
results/spatial_table.tex
```

`results.md` 是服务器 agent 最先回传给作者的简明结果；两份 `.tex` 可直接
复制/`\input` 到论文。不要手工改数字。

## 13. 正式指标

- **Succ. (%)**：每条 reference 的 10 个 completed rollout 至少一个成功，
  再对 references 等权平均。
- **Duration (s)**：每条 reference 最长 trial 的 steps / 30，再等权平均。
- **\(E_h\) (cm)**：同一 best trial 的 21-key-body tracking error。
- **\(E_o\) (cm)**：同一 best trial 的双物体 surface-point tracking error。

同 duration 时按较小 \(E_h+E_o\)，再按较小 trial ID 破同分。四项均为
InterMimic 风格 reference-imitation 指标。`semantic_success`、reach/contact、
stable grasp、wrong contact 和终态误差只作内部诊断，不进入论文主表。

正式 evaluator 使用 `theia_policy_eval.yaml`，其 Start/full-sequence 和
ET/IET 开关与 Stage-B 保持一致。项目有意关闭 GT contact-miss hard termination，
允许可行但与 GT 接触时序不同的策略；论文 methods/caption 必须披露这一
termination 变体，不能声称逐项复现了 InterMimic 未公开的官方 rollout budget。

Raw 也使用与 Full 完全相同的 measured contact schedule 和 object trajectory；
因此论文中的准确条件名应是“Raw kinematics + shared object/contact
supervision”与“Tactile-refined kinematics + shared object/contact
supervision”。这种配对设计用于把成功率差异归因于人体/手部 refinement，
不能在看到结果后为 Raw/Full 分别调整接触权重或终止条件。

## 14. 最终 Go / No-Go

以下全部存在且 valid 才可正式启动：

- 父 Theia 和 InterMimic 两个仓库的所需提交；
- frozen candidate list 和可审计的 L/V metadata；
- `policy_ab_manifest.json`；
- `PRECHECK_READY.json`；
- `SMOKE_READY.json`；
- Raw/Full 相同 N、env、minibatch、seeds、epochs 和 K；
- 每张计划使用的 GPU 已通过 CUDA/Isaac 启动；
- 调度器有独立目录、退出码传播和恢复逻辑。

代码静态协议测试：

```bash
python isaacgym/scripts/test_theia_training_protocol.py
python isaacgym/scripts/test_theia_policy_results.py
```

当前本机已经用一条真实 339-frame S1 sequence 完成 paired conversion、CPU
preflight，以及 Raw/Full 各 `1 epoch fresh + 1 epoch full-state resume +
1 epoch Stage-B + K=1 evaluation` 的微型 GPU 闭环；正式单
condition/seed wrapper 的从零、恢复、目录冻结、评测和幂等重跑也已实测。
合成结果测试另行覆盖了 K-trial 汇总、配对 bootstrap 和表格生成。服务器全部
数据仍未知，因此服务器 `PRECHECK_READY.json` 和默认 100-epoch
`SMOKE_READY.json` 仍是正式训练的最后两道必需门，不能由本机单序列微测替代。
