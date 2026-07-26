# Theia S1 Raw-vs-Refined policy 实验服务器 HANDOFF

更新时间：2026-07-25（Asia/Shanghai）

## 0. 最重要的实验契约

`single_reference_raw_vs_refined_v2`（2000 epochs）是本项目唯一受支持的
正式 policy 方法。服务器尚未产生 v1/1100 正式结果，因此不存在旧结果迁移或
混用需求。旧的 universal-policy、20k+2k、四 seed 和 v1 脚本/文档只可用于
历史审计，不能启动 rebuttal 实验。

本次 rebuttal 不训练一个覆盖全部 S1 的通用 policy，也不做多 training-seed
重复。实验单位是单条 reference：

```text
每条 S1 reference
  ├── Raw MoCap：从随机权重训练 1 次
  └── Measured-tactile refinement：从随机权重训练 1 次
```

正式参数已固定：

- 每个 `reference × condition` 只训练一次；
- training seed 固定为 `0`，它只是固定随机初始化，不表示重复训练；
- 单阶段 Hybrid/RSI 训练 2000 epochs；
- Raw/Refined 使用相同 seed、PPO、环境数、物理参数、物体轨迹和接触标签；
- 每条 policy 用 `K=10` 个并行 rollout 评测；这是一次并行评测，不是训练十次；
- 主指标为 InterMimic 风格的 RefSucc@10、Duration、\(E_h\)、\(E_o\)；
- 额外保留 episode completion，用于描述 rollout 稳定性；
- 统计时以 reference 为单位做 paired bootstrap，不把帧或 K 个 rollout 当作
  独立训练样本，也不声称估计了 training-seed 方差。

内部目录仍使用条件名 `full`，其论文含义是
`Measured-tactile refinement`。不要再调度 `seed_1/2/3`，也不要使用旧的
20k+2k 全数据集方案。

## 1. 服务器 agent 的工作边界

仓库已提供：

- Raw/Refined 成对转换、hash 冻结和 CPU 校验；
- Theia、InterMimic、ProtoMotions 三仓库版本硬检查；
- 单 reference 的 Raw/Refined 配对训练与断点续训；
- 接收 reference list 的单 GPU worker；
- K=10 完整 cohort 评测；
- 逐 reference、姿态分组和论文表格聚合。

服务器 agent 只需：

1. 填写服务器路径和 Conda 环境；
2. 在 24 GiB RTX 4090 上烟测并选择共同的 `NUM_ENVS`；
3. 把全部 reference IDs 不重不漏地分成集群/GPU 列表；
4. 每张 GPU 启动一个 list worker；
5. 汇集各集群的 `references/` 目录并运行聚合器。

服务器 agent 可以优化 GPU 调度、失败重启和外部资源监控，但不能更改
Raw/Refined 之间的训练预算、seed、配置或评测 K。

## 2. 三仓库版本管理（第一道硬门）

正式流程跨越：

```text
Theia/toolkit/scripts/theia2intermimic.py
Theia/thirdparty/InterMimic/
Theia/thirdparty/ProtoMotions/
```

这两个第三方目录不是 Theia 的 Git submodule，必须分别 clone/fetch/checkout，
不能只更新父 Theia。唯一正式版本清单是：

```text
InterMimic/THEIA_POLICY_SERVER_VERSION.json
```

它固定：

```text
Theia
  remote: Arkitect-z/Theia
  commit: 611e75243247c67f96e977b99345c74cbba7806c

InterMimic
  remote: Arkitect-z/InterMimic
  tag: theia-policy-rebuttal-v3-2000

ProtoMotions
  remote: NVlabs/ProtoMotions
  commit: 4a905b998101333a2fb91f2de8e2cab4bd0db68e
```

父 Theia 提供 converter；它依赖 ProtoMotions 的
`poselib.poselib.skeleton.skeleton3d.SkeletonMotion`。服务器 agent 应先确认
三个工作区没有需要保留的 tracked 修改，再执行：

```bash
git -C "$THEIA_ROOT" fetch origin
git -C "$THEIA_ROOT" checkout --detach \
  611e75243247c67f96e977b99345c74cbba7806c

git -C "$THEIA_ROOT/thirdparty/ProtoMotions" fetch origin
git -C "$THEIA_ROOT/thirdparty/ProtoMotions" checkout --detach \
  4a905b998101333a2fb91f2de8e2cab4bd0db68e

git -C "$INTERMIMIC_ROOT" fetch origin main --tags
git -C "$INTERMIMIC_ROOT" checkout --detach \
  theia-policy-rebuttal-v3-2000
```

然后由 agent 自主运行唯一总检查器：

```bash
cd "$INTERMIMIC_ROOT"
python isaacgym/scripts/check_theia_server_versions.py \
  --output-json /server/manifests/theia_policy_repository_versions.json
```

只有输出 `valid: true` 才能继续。检查器会：

- 接受 HTTPS 或 SSH clone，但 GitHub owner/repository 必须匹配；
- 强制 Theia 与 ProtoMotions 为精确 commit；
- 强制 InterMimic HEAD 精确等于正式 tag；
- 拒绝三个仓库的 tracked dirty 修改；
- 复算 converter 与 `SkeletonMotion` 源码 SHA-256；
- 输出三个仓库的 HEAD、remote、tag commit 和协议参数。

未跟踪的 SMPL-X 模型文件允许存在，因为它们不改变代码版本。GPU 调度脚本、
reference lists 和实验输出应写在仓库外，避免污染 tracked 工作区。converter、
CPU preflight、单 reference 入口和 list worker 都会再次自动运行同一版本硬门；
不能只靠人工记录绕过。

## 3. 路径

```bash
THEIA_ROOT=/server/path/Theia
INTERMIMIC_ROOT="$THEIA_ROOT/thirdparty/InterMimic"
SOURCE_ROOT=/server/path/to/S1_sequence_directories
OBJECTS_ROOT="$THEIA_ROOT/data/objects"
POLICY_DATA_ROOT=/server/experiments/theia_policy_ab_data
EXPERIMENT_ROOT=/server/experiments/theia_policy_ab_runs
CONDA_ENV=intermimic
```

CUDA/Isaac Gym 检查：

```bash
conda run --no-capture-output -n "$CONDA_ENV" python - <<'PY'
from isaacgym import gymapi
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.is_available(), torch.cuda.device_count())
PY
nvidia-smi
```

转换还依赖 `torch, numpy, scipy, h5py, trimesh`；转换本身不要求 CUDA。

## 4. 冻结全部 S1 候选清单

正式 S1 使用同高度 `L11/L22/L33/L44/L55` 和 `V01/V02/V03`。必须在查看
policy 结果之前冻结全量清单：

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
if not paths:
    raise SystemExit("No canonical S1 sequence was found")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("".join(f"{path}\n" for path in paths))
print(f"Frozen {len(paths)} S1 references in {output}")
PY
```

审计未进入清单的 `S1*` 目录并保存原因。不能根据 policy 成功率事后删除难
序列。若需要 action/task 字段，在 rollout 之前冻结 metadata CSV；它只用于
分组，不参与训练。

## 5. 一次性生成成对数据

```bash
cd "$INTERMIMIC_ROOT"
N_CANDIDATES=$(grep -cv '^[[:space:]]*$' \
  /server/manifests/s1_policy_candidates.txt)

conda run --no-capture-output -n "$CONDA_ENV" \
  python isaacgym/scripts/prepare_theia_policy_ab.py \
  --source-root "$SOURCE_ROOT" \
  --sequence-list /server/manifests/s1_policy_candidates.txt \
  --objects-dir "$OBJECTS_ROOT" \
  --output-root "$POLICY_DATA_ROOT" \
  --expected-count "$N_CANDIDATES"
```

如需 metadata，追加：

```bash
--reference-metadata /server/manifests/s1_policy_metadata.csv
```

正式转换会：

- 显式生成 Raw 和 Refined，不允许 fallback；
- 复核三仓库正式版本并把完整回执写入 manifest；
- 使用 Raw/Refined 两者所需值中较大的共同 ground shift；
- 强制 frame indices、object trajectory 和 contact schedule 成对一致；
- 强制 tensor 的 object/contact 列 `318:386` 逐位相同；
- 保存 converter、数据、物体资产和依赖 SHA-256。

若 `excluded_pairs.csv` 非空，先判断是否真的是文件损坏、缺资产或 schema
错误。目标是所有规范且可转换的序列都运行；不能以 motion 难度或结果差为排除
理由。

## 6. CPU preflight

这里的 `NUM_ENVS` 只需不小于全数据集 reference 数，以便一次校验所有文件；
它不要求能被 reference 数整除，因为正式训练每个 job 只有一个 reference。

```bash
cd "$INTERMIMIC_ROOT"
NUM_ENVS=2048 \
MINIBATCH_SIZE=16384 \
CONDA_ENV="$CONDA_ENV" \
bash isaacgym/scripts/preflight_theia_policy_ab.sh "$POLICY_DATA_ROOT"
```

若人工确认了预注册技术排除，才可加 `ACCEPT_EXCLUSIONS=1`。必须生成：

```text
PRECHECK_READY.json
policy_ab_validation.json
raw_dataset_validation.json
full_dataset_validation.json
repository_versions.json
```

该门还会确认 Raw/Refined 的人体参考确实不同、object/contact supervision
相同、数据 finite、四元数有效和手—物体标签一致。Isaac FK propagation
只能作为诊断记录，不能据此排除一个 finite、schema 合法且仍可能训练成功的
reference。

## 7. GPU smoke 与 4090 环境数

不要直接用正式目录调试。先选一个预先固定的 reference：

```bash
SMOKE_REF=$(head -n 1 /server/manifests/s1_policy_candidates.txt)
SMOKE_REF=$(basename "$SMOKE_REF")

CUDA_VISIBLE_DEVICES=0 \
CONDA_ENV="$CONDA_ENV" \
ALLOW_PROTOCOL_OVERRIDE=1 \
TRAIN_EPOCHS=2 \
K=1 \
NUM_ENVS=32 \
MINIBATCH_SIZE=256 \
TARGET_ENVS=32 \
bash isaacgym/scripts/run_theia_policy_reference.sh \
  "$SMOKE_REF" "$POLICY_DATA_ROOT" \
  "$EXPERIMENT_ROOT/_smoke_quick"
```

然后在另一个 smoke 目录测试正式显存规模。8×4090 24 GiB 的起点是：

```text
NUM_ENVS=2048
MINIBATCH_SIZE=16384
```

若 OOM，依次尝试 1792、1536、1280、1024；默认 minibatch 可始终取
`NUM_ENVS*8`。要求：

```text
(NUM_ENVS * 32) % MINIBATCH_SIZE == 0
MINIBATCH_SIZE % 4 == 0
```

至少跑 50--100 epochs 后再确认显存/FPS。为 PhysX、checkpoint 和临时 tensor
保留约 1.5--2 GiB，不以 `nvidia-smi` 必须占满 24 GiB 为目标。正式 Raw 与
Refined、所有 GPU/集群使用同一个最终环境数。

快速 smoke 的 `pair_spec` 与正式 2000-epoch 协议不同，聚合器会拒绝它进入
论文结果。

## 8. 按集群/GPU 切分 reference lists

列表每行可写 reference ID 或原始目录路径，允许空行和 `#` 注释。全部列表的
并集必须等于 frozen eligible IDs，且彼此不重叠。可由服务器 agent 根据各集群
空闲 GPU 数和预计序列长度做负载均衡。

示例：

```text
/server/manifests/cluster_a_gpu0.txt
/server/manifests/cluster_a_gpu1.txt
...
/server/manifests/cluster_b_gpu0.txt
```

启动前用仓库脚本对照 frozen manifest 检查所有集群列表的并集：

```bash
python isaacgym/scripts/validate_theia_reference_lists.py \
  --pair-manifest "$POLICY_DATA_ROOT/policy_ab_manifest.json" \
  --output-json /server/manifests/s1_policy_shards.json \
  /server/manifests/cluster_a_gpu*.txt \
  /server/manifests/cluster_b_gpu*.txt
```

只有 `valid: true` 才能启动。它会拒绝 missing、extra、单列表重复和跨集群
duplicate，并冻结每个 list 的 SHA-256。worker 还会在启动时再次拒绝本列表的
重复 ID 和 manifest 外 ID。

## 9. 正式单 GPU worker

每张 GPU 对自己的列表顺序运行。每个 reference 内部先训练 Raw 一次，再训练
Refined 一次：

```bash
CUDA_VISIBLE_DEVICES=0 \
CONDA_ENV="$CONDA_ENV" \
NUM_ENVS=2048 \
MINIBATCH_SIZE=16384 \
SHARD_NAME=cluster_a_gpu0 \
bash isaacgym/scripts/run_theia_policy_reference_list.sh \
  /server/manifests/cluster_a_gpu0.txt \
  "$POLICY_DATA_ROOT" \
  "$EXPERIMENT_ROOT"
```

8 张 GPU 启动 8 个这样的进程，每个进程使用不同 list 和
`CUDA_VISIBLE_DEVICES`。不同集群可以同时启动。不要在同一 GPU 上叠加多个
Isaac Gym worker；一个 2048-env worker 已用于充分占用该卡。

正式命令不要设置以下变量：

```text
ALLOW_PROTOCOL_OVERRIDE
TRAIN_EPOCHS
TRAINING_SEED
EVAL_SEED
K
```

脚本默认并硬检查：

```text
TRAIN_EPOCHS=2000
TRAINING_SEED=0
EVAL_SEED=10000
K=10
TORCH_DETERMINISTIC=0
```

`TORCH_DETERMINISTIC=0` 与此前成功 launcher 一致且更快；Raw/Refined 仍共享
固定 seed 和相同训练配置，但 GPU PhysX 不保证逐步 bitwise 确定。重复相同
worker 命令会恢复模型、优化器、normalizer、epoch/frame 和持久 PSI curriculum；
恢复后的新 episode 不承诺复现中断前的瞬时模拟/RNG 状态。已经完成的训练和
已 hash 验证的评测会跳过。

PSI 训练只接受本版本生成、包含 schema 化 curriculum 状态的 checkpoint。
旧 checkpoint 的 `env_state` 为 `None`，不能作为完整断点续训来源；脚本会拒绝
在原目录静默从零覆盖。需要使用旧权重做非正式实验时，应另建目录并显式采用
policy-only warm start，不能混入正式结果。

## 10. 多集群结果汇集

最好让所有集群写同一个可靠共享文件系统。如果各集群使用独立存储，最终必须把：

```text
EXPERIMENT_ROOT/references/<reference_id>/
EXPERIMENT_ROOT/shards/
EXPERIMENT_ROOT/repository_versions.json
```

汇入一个中央 `EXPERIMENT_ROOT`。reference lists 不重叠，因此不应出现同名
reference；若出现冲突，先比较 `pair_spec.txt` 和 hash，不能静默覆盖。
所有集群的 `repository_versions.json` 必须指向同一 release/tag commit；绝对
路径以及 HTTPS/SSH remote 表示可以不同。中央目录保留任一份 `valid: true`
的正式回执，聚合器会再用每条结果的 `pair_spec.txt`/`run_spec.txt` 交叉校验
InterMimic commit 和 clean diff hash。

每条完成的 reference 都必须有：

```text
references/<id>/PAIR_READY.json
references/<id>/runs/raw/seed_0/evaluation/final/validation.json
references/<id>/runs/full/seed_0/evaluation/final/validation.json
```

## 11. 一键聚合和论文产物

全部结果汇集后：

```bash
cd "$INTERMIMIC_ROOT"
python isaacgym/scripts/aggregate_theia_policy_references.py \
  --experiment-root "$EXPERIMENT_ROOT" \
  --pair-manifest "$POLICY_DATA_ROOT/policy_ab_manifest.json" \
  --output-dir "$EXPERIMENT_ROOT/results"
```

默认要求 manifest 中每条 eligible reference 的 Raw/Refined 正式结果都存在；
缺一条即失败。若只想检查某个集群的中间进度，可加该集群的
`--reference-list`，但中间结果不能作为全 S1 论文表。
聚合器还会拒绝缺失/不一致的三仓库版本回执、非正式 tag、dirty worktree
生成的 `pair_spec.txt` 或 `run_spec.txt`。

产物：

```text
results/per_reference_paired.csv  # 每条序列的 Raw/Refined 配对结果
results/paired_results.csv        # 全 S1 均值、paired CI、差值
results/by_pose.csv               # height/variation 分组的同一组标准指标
results/summary.json              # 机器可读统计与协议
results/results.md                # 可直接审阅
results/main_table.tex            # 论文主表
```

主表只放 InterMimic 常用四项：

```text
RefSucc@10 ↑ | Duration ↑ | E_h ↓ | E_o ↓
```

`Episode completion` 放在补充结果中。它是 K 个 rollout 的完成比例，可说明
policy 稳定性，但不替代 InterMimic 的 reference coverage。

## 12. 与先前成功单序列代码的差异审计

已验证的早期基线提交是：

```text
3bc54a5 Add dual-object SMPLX interaction policy with residual control
```

其保存 checkpoint 位于 epoch 1100。当前逐 reference 配方继续使用其核心，
并把正式预算提高到 2000 epochs，为更困难的未见序列保留额外收敛空间：

仓库内可确认的事实是：`3bc54a5` 跟踪的
`theia_data/sub1_CupBlue+KettleGreen_S1L33P01T0508V01.pt`
SHA-256 为
`d52cf6b0d4d672321c81a1d99892778fef3fdb09202176fb128bf0e1ecbd73c9`，
当前文件则为
`8b2bc64a7e991573e9198b0969a440889005db2e47165fe1a681cabdcbf9c790`。
epoch-1100 checkpoint 的 SHA-256 是
`31df8385c8473f27147ebd89d1be6d9facaea4da768ee55f7cc51cfe2449fa8a`，
但 checkpoint 内没有数据或配置 hash，仓库证据不能证明它使用了当前 Refined
文件。因此 epoch 1100 只能作为早期单序列配方成功的证据，不能作为当前数据
逐字节同源的证据。正式实验统一训练 2000 epochs，而不是按序列难度动态加时；
Raw 是否能在相同预算内学会正是本次 comparison 的测量对象。更长预算也可能
让部分 Raw policy 追上，因此它提高困难序列的绝对成功率，但不保证扩大
Raw/Refined 差值。

| 项目 | 早期成功配方 | 当前正式逐 reference 配方 |
|---|---|---|
| 网络 | `[1024,1024,512]` | 不变 |
| PPO LR / clip | `2e-5 / 0.2` | 不变 |
| horizon / mini epochs | `32 / 6` | 不变 |
| 初始化 | Hybrid，配置 rollout 100；本序列有效 156 | 不变 |
| 训练预算 | 成功 checkpoint epoch 1100 | 固定 2000 |
| contact reward | legacy multiplicative | 保留 |
| wrist/object phase reset | 已硬编码使用 | 保留并配置化 |

保留的必要 bug 修复：

- dual-object 和多 reference 的正确 actor/data 绑定；
- action 以 `t+1` reference 为 PD target，保留旧版成功配方的 residual
  correction range；action 仍裁剪到 `[-1,1]`，关节物理范围由 Isaac 执行；
- 右腕 DOF 解析、四元数角速度；RSI reset 始终恢复参考速度，
  `initVel` 只控制参考数据首帧速度的构造；
- reset 后 observation 刷新和 batched reset；
- Raw/Refined 共同 ground alignment；
- 物体密度字典带默认值，不要求精确真实密度；
- 手与目标物体的正确配对；
- 去掉会虚增失败轨迹 reward 的 object reward floor；
- evaluator 的 active-env cohort、恰好 K trials 和 episode-level error 聚合；
- schema、finite、资产与序列绑定错误的 data fail-fast；FK propagation
  超阈值仅诊断，不阻断训练；

为降低失败风险和开销，当前明确不采用：

- 因错过 GT contact timing 而 hard terminate；
- approximate wrong-contact penalty；
- exact actor-pair contact 参与训练；
- terminal semantic bonus；
- reward-best checkpoint 筛选；
- 逐 step diagnostics、逐 step tensor `.item()` 和 trajectory dump；
- 20k+2k 通用 policy、多 training seeds。

训练仍保留弱 contact shaping、wrist tracking 和 object contact-phase tracking。
这不是额外的 contact ablation，而是早期成功配方已有控制目标的修正版。Raw 与
Refined 使用完全相同的 contact schedule 和权重；Refined 的潜在优势只能来自
更物理可实现的人体/手部参考几何，而不是给 Raw 人为增加惩罚。无法诚实保证
Refined 一定显著更高；若结果不显著，不能通过修改 Raw 专属配置制造差距。

## 13. 训练期记录与速度

正式训练关闭：

- `enableTrainingDiagnostics`；
- `enableStepDiagnostics`；
- exact-contact evaluation during training；
- reward-best 额外 checkpoint；

保留：

- 标准 `train.log` 和 TensorBoard scalar；
- 每 50 epochs 一次非阻塞资源 telemetry；
- Physical Buffer/PSI 状态替换（`physicalBufferSize: 3`），其课程状态随
  full-state checkpoint 保存和恢复；
- 每 200 epochs 覆盖写一个 rolling checkpoint，并保留同周期永久 milestone；
- epoch 2000 final checkpoint；
- 一次 K=10 评测的 CSV/JSON/hash。

这些保留项用于断点恢复和 rebuttal 证据链，开销远小于 PhysX/RL rollout。
每个 run 不保存图像、视频或逐步轨迹。可视化应只对少量选定 policy 单独 play，
不要放在批量训练 worker 中。
