# Theia 全数据集服务器训练 HANDOFF

> **范围提示（2026-07-24）：**本文档保留为单 condition 生产训练的历史
> handoff。论文的 paired S1 Raw-vs-Full 实验禁止使用
> `run_theia_server.sh`，因为它不是逐 reference、Raw/Refined 各训练一次的
> list-sharded \(K=10\) 协议。正式论文实验请只遵循
> `HANDOFF_SERVER_POLICY_AB.md` 中唯一受支持的 v2/3000 方法；本文其余命令
> 均不具有正式实验资格。为防止误启动，`run_theia_server.sh` 默认拒绝运行；
> 只有非正式历史诊断才可显式设置 `ALLOW_LEGACY_UNIVERSAL=1`。

更新时间：2026-07-24（Asia/Shanghai）

目标：让服务器端 agent 在完全不了解此前对话的情况下，直接接管全部未训练
Theia 序列的检查、训练、断点恢复、评测和结果汇报。

## 0. 必须先读的结论

1. 服务器上的所有序列都视为从未训练过。正式训练必须从随机 policy 权重
   开始，不加载本机单序列 checkpoint。
2. 生产方案是“原始 InterMimic 训练目标 + 已确认的 correctness bug 修复”。
   不把本机未完成从零验证的 soft-contact、terminal pose bonus 或精确
   actor-pair contact 训练加入服务器默认方案。
3. 此历史方案的入口如下（不得用于正式 Raw-vs-Refined 实验）：

   ```bash
   THEIA_DATA_DIR=/absolute/path/to/server/theia_pt \
   bash isaacgym/scripts/run_theia_server.sh
   ```

4. 同一命令可安全用于断点恢复。脚本读取 checkpoint 的真实 epoch，只补跑
   剩余预算，并恢复 model、optimizer、normalizer、epoch、frame 和 best state。
5. 新物体不需要填写真实密度。Cup/Kettle 保留已有近似值，其他物体统一使用
   `1000 kg/m³`。
6. 不允许只按 PPO reward 宣布成功。最终必须读取逐 episode 和逐序列评测；
   默认门槛是总体 semantic success `>=95%`、最差序列 `>=50%`。
7. 未见服务器数据前，不能诚实保证某个数值成功率。当前代码保证的是：
   CUDA、数据、资产、骨架、序列绑定或 checkpoint 不自洽时 fail fast，而不是
   静默训练错误 policy。

## 1. 极重要：代码版本

本次修复之前的基线：

```text
branch: main
previous base commit: 3bc54a5325e0ed263682294a1a0dcce548caf8d0
```

服务器端不能只 checkout 上述旧基线；这样会缺失一键脚本、全量配置、评测器
和预检器。必须拉取同时包含本文档和下列文件的后续提交。

服务器 agent 的第一步必须确认下列文件已被 Git 跟踪，并核对第 15 节的
SHA-256：

```text
isaacgym/scripts/run_theia_server.sh
isaacgym/scripts/train_theia_full.sh
isaacgym/scripts/eval_theia.sh
isaacgym/scripts/validate_theia_dataset.py
isaacgym/src/intermimic/data/cfg/theia_full_train.yaml
isaacgym/src/intermimic/data/cfg/theia_full_finetune.yaml
isaacgym/src/intermimic/data/cfg/theia_eval.yaml
```

```bash
git ls-files \
  HANDOFF_SERVER_THEIA.md \
  isaacgym/scripts/run_theia_server.sh \
  isaacgym/src/intermimic/data/cfg/theia_full_train.yaml
```

不要用旧基线提交覆盖修复。服务器训练产生的 checkpoint、日志和评测目录不在
源码提交内，也不应被 `git clean` 当作无价值文件删除。

## 2. 已确认的最终训练设计

### 2.1 Stage A：从零 bootstrap

- 随机 policy 权重，不使用本机 checkpoint。
- `stateInit: Hybrid`
- Hybrid/RSI 分段训练。
- 默认目标：20,000 epochs。
- RSI 会提高接触转换区间的采样权重。
- rollout 会自动延长到覆盖最晚的接触 onset。
- PPO 使用原始 InterMimic 从零参数：
  - learning rate `2e-5`
  - horizon `32`
  - minibatch `16384`
  - mini epochs `6`
  - PPO clip `0.2`
  - critic coefficient `5`

### 2.2 Stage B：完整序列 finetune

- 从 Stage A checkpoint 完整恢复训练状态。
- `stateInit: Start`
- 自动将 rollout length 设置为数据集中最长序列。
- 每条较短序列仍在自身最终帧结束。
- 默认再训练 2,000 epochs。

### 2.3 生产接触设计

保留：

- `contactRewardMode: legacy_multiplicative`
- 参考 contact 作为原始 InterMimic shaping。
- 左手只匹配 obj1，右手只匹配 obj2。
- 很弱的正确 pair grasp bonus。
- wrist/object trajectory early termination，用于避免大量无效 rollout。

关闭：

- GT contact miss hard termination。
- GPU 距离代理 wrong-contact penalty。
- terminal world-frame object pose bonus。
- 精确 PhysX actor-pair contact 训练。

原因：本机 A/B 是从强单序列 checkpoint 微调，不能证明 soft-contact 更适合
完全未训练的新序列。`legacy_multiplicative + correctness fixes` 是当前风险
最低的服务器默认。

精确 actor-pair contact 仅用于最终 CPU PhysX 诊断，不进入训练 reward。

## 3. 已修复的关键问题

服务器 agent 不应回退这些修复：

- 右腕 DOF 名称和索引错误。
- 四元数角速度计算错误和非物理尖峰。
- `initVel: false`。
- residual PD 改为围绕 `t+1` reference，并按 body/wrist/finger 限幅后再按
  XML joint limit clamp。
- reset 后 observation 陈旧。
- object reward floor 导致物体未跟随仍可获得高 reward。
- 物体 density/mass 未正确进入 Isaac runtime。
- 非手部 contact 标签被误当作 forbidden contact。
- motion、object pair、support table 在多序列环境中错配。
- 每个环境固定绑定正确序列和对应资产。
- 完整序列被固定 339 帧截断。
- FK 过去只检查第一条 motion；现在所有 motion 并行、各自均匀采样 8 个时刻，
  总计只需 8 个 PhysX steps。
- 评测器 `games_num × 10` 提前退出；现在每个初始 env 恰好记录一次，并要求
  `Episodes == NUM_ENVS`。
- 误差过去覆盖同一个 `[sequence,timestep]`；现在按 episode 累计并输出
  mean、median、P95 和 failed mean。
- checkpoint 过去只恢复 policy 或吞掉异常；现在正式 resume 恢复完整状态并
  fail loudly。
- 训练前 5 epochs 曾跳过 frame/log/budget；已经删除该 continue。

## 4. 服务器数据契约

一键启动前，服务器数据必须满足：

- 每个文件是二维 PyTorch tensor，shape 为 `[T, 594]`。
- 文件名以 `sub<number>_` 开始。
- 文件名中恰有一个包含 `+` 的 token，例如：

  ```text
  sub12_CupBlue+KettleGreen_sequence001.pt
  ```

- `+` 左侧是 obj1，必须由左手操作；右侧是 obj2，必须由右手操作。
- 当前任务结构要求每条序列左右手都至少有正接触帧。
- human contact 中非手部 link 必须是中性标签 `0`。
- obj1 contact 列必须和左手 contact 一致，obj2 必须和右手一致。
- 所有 quaternion 必须有限且近似单位四元数。
- 所有数据必须无 NaN/Inf。
- 足部碰撞几何最低点必须不明显穿地。
- 每个对象必须存在：

  ```text
  isaacgym/src/intermimic/data/assets/objects/objects/<Object>/<Object>.obj
  isaacgym/src/intermimic/data/assets/objects/<Object>.urdf
  ```

- 所有 motion 必须与当前 `smplx/theia.xml` articulation 兼容。运行时 FK
  硬检查阈值是最大位置误差 `1.5 cm`、最大旋转误差 `10°`。

如果服务器数据包含以下情况，必须停止并报告，不能仅关闭 validator：

- 单手任务或左右手与 obj1/obj2 语义相反。
- 同一序列超过两个对象。
- 同一只手需要同时承担两个对象但 PT schema 无法表达。
- 不同人物需要不同骨架/shape，而不是共同的 `theia.xml`。
- 序列数大于单卡能同时容纳的环境数。

## 5. 一键启动

### 5.1 数据位于仓库外

```bash
cd /path/to/InterMimic

THEIA_DATA_DIR=/absolute/path/to/server/theia_pt \
bash isaacgym/scripts/run_theia_server.sh
```

### 5.2 数据位于仓库内 `theia_data/`

```bash
cd /path/to/InterMimic
bash isaacgym/scripts/run_theia_server.sh
```

脚本会自动：

1. 尝试进入 `intermimic` Conda 环境。
2. 验证 CUDA、PyTorch CUDA 和 Isaac Gym。
3. 统计全部 `.pt` 序列。
4. 在约 2048 env 的目标下选择按序列完全平衡的环境数。
5. 运行静态数据/资产 preflight。
6. 创建 GPU PhysX 环境并检查所有 motion 的 FK。
7. 训练 Stage A。
8. 完整恢复到 Stage B。
9. 运行平衡的 GPU 全长评测。
10. 写入 manifest、logs、checkpoints、CSV 和 JSON。
11. 按总体和最差序列门槛返回 0 或非 0。

## 6. 常用覆盖参数

所有参数都通过环境变量传入：

| 变量 | 默认值 | 含义 |
|---|---:|---|
| `THEIA_DATA_DIR` | `theia_data/` | 服务器 PT 数据目录 |
| `CONDA_ENV` | `intermimic` | Conda 环境名 |
| `TARGET_ENVS` | `2048` | 自动环境数的目标上限附近值 |
| `NUM_ENVS` | 自动 | 强制训练环境数 |
| `BOOTSTRAP_EPOCHS` | `20000` | Stage A 绝对目标 epoch |
| `FINETUNE_EPOCHS` | `2000` | Stage B 追加 epoch |
| `OUTPUT_ROOT` | `checkpoints/theia_server_full` | 所有正式产物目录 |
| `SEED` | `42` | 训练随机种子 |
| `EVAL_REPEATS` | `4` | 每条序列目标评测次数 |
| `EVAL_TARGET_ENVS` | `TARGET_ENVS` | 评测环境容量 |
| `EVAL_ENVS` | 自动 | 强制评测环境数 |
| `MIN_SUCCESS_RATE` | `0.95` | 总体 semantic success 门槛 |
| `MIN_SEQUENCE_SUCCESS_RATE` | `0.50` | 最差序列门槛 |

显存不足、且序列数不超过 1024 时：

```bash
TARGET_ENVS=1024 \
THEIA_DATA_DIR=/absolute/path/to/server/theia_pt \
bash isaacgym/scripts/run_theia_server.sh
```

不要把 `NUM_ENVS` 设为小于 `max(512, sequence_count)`。当前实现要求每个
motion 至少有一个固定绑定环境。

## 7. 断点恢复

作业、SSH 或网络中断后，使用完全相同的命令：

```bash
THEIA_DATA_DIR=/absolute/path/to/server/theia_pt \
bash isaacgym/scripts/run_theia_server.sh
```

脚本会：

- 优先选择 epoch 更高的 stage checkpoint。
- 从 checkpoint 读取绝对 epoch。
- 只运行 `target_epoch - current_epoch`。
- 完整恢复 optimizer、normalizers、frame、reward-best state。
- 已完成 stage 会显示 `[SKIP]`。
- 使用 `flock` 防止同一 `OUTPUT_ROOT` 启动两个并发作业。

不要手工使用 policy-only warm start 代替 resume。

## 8. 未达成功门槛时

脚本不会删除任何训练产物，但最终返回非零。

优先查看：

```text
checkpoints/theia_server_full/evaluation/epoch_*/summary.json
checkpoints/theia_server_full/evaluation/epoch_*/episodes.csv
```

如果总体已经较高，但少数序列失败：

1. 查 `summary.json -> sequences` 的最差序列。
2. 查对应 `episodes.csv` 的 completion、final position、final rotation、
   contact 和 wrong-contact diagnostics。
3. 先确认失败不是 PT、对象资产或左右手语义错误。
4. 数据正确时，增加全长 finetune 预算并运行同一脚本：

   ```bash
   FINETUNE_EPOCHS=4000 \
   THEIA_DATA_DIR=/absolute/path/to/server/theia_pt \
   bash isaacgym/scripts/run_theia_server.sh
   ```

`FINETUNE_EPOCHS=4000` 表示 Stage B 总追加预算提高到 4000，不是从头再跑
4000；脚本只补差额。

不要因为一次失败就切换 soft contact、加入 wrong-contact penalty 或关闭全部
early termination。任何新的 reward 设计必须有同数据、同 seed、从零训练的
受控 A/B 证据。

## 9. 输出目录

默认：

```text
checkpoints/theia_server_full/
  .run.lock
  server_run.log
  data_manifest.json
  bootstrap/
    data_manifest.json
    run_manifest.txt
    train.log
    theia_smplx/
      nn/
        mimic.pth
        mimic_best.pth
      summaries/
  finetune/
    data_manifest.json
    run_manifest.txt
    train.log
    theia_smplx/
      nn/
        mimic.pth
        mimic_best.pth
      summaries/
  evaluation/
    epoch_<N>_<timestamp>/
      manifest.txt
      eval.log
      summary.json
      episodes.csv
```

正式候选默认使用：

```text
checkpoints/theia_server_full/finetune/theia_smplx/nn/mimic.pth
```

不要仅因为 `mimic_best.pth` 的 PPO reward 较高就将其作为最终 policy。

## 10. 成功指标定义

当前 semantic success 要求：

- 完整运行到该序列自身最终帧。
- 两个目标对象都曾出现正确 hand-object 接触。
- 如果该序列 GT 确实包含连续 10 帧双手同时接触，则还要求同时稳定抓持。
- 最终每个对象位置误差 `<=5 cm`。
- 最终每个对象旋转误差 `<=20°`。

wrong-contact steps 会报告，但默认不是 universal failure condition，因为未知
任务可能存在不同于 GT、但物体任务确实完成的有效策略。

GPU 快速评测 contact source 是：

```text
net_force_plus_distance_proxy
```

正式物理诊断可对最终候选额外运行 CPU actor-pair contact：

```bash
conda activate intermimic

MOTION_FILE=/absolute/path/to/server/theia_pt \
STRICT_CONTACTS=1 \
NUM_ENVS=<至少为序列数> \
EVAL_OUTPUT_DIR=checkpoints/theia_server_full/evaluation_cpu_exact \
bash isaacgym/scripts/eval_theia.sh \
  checkpoints/theia_server_full/finetune/theia_smplx/nn/mimic.pth
```

CPU PhysX 与 GPU PhysX 是不同 backend，结果必须分开报告。

## 11. Policy 可视化

服务器有显示或 VirtualGL 时，可只选择一条序列进行 GUI play。不要把整个
大数据集同时开 GUI。

```bash
conda activate intermimic
cd /path/to/InterMimic
export PYTHONPATH="$PWD/isaacgym/src:$PWD:${PYTHONPATH:-}"

VIZ_DATA="$(mktemp -d /tmp/theia-viz.XXXXXX)"
ln -s /absolute/path/to/one_sequence.pt "$VIZ_DATA/"

python -m intermimic.run \
  --task InterMimic \
  --cfg_env isaacgym/src/intermimic/data/cfg/theia_eval.yaml \
  --cfg_train isaacgym/src/intermimic/data/cfg/train/rlg/theia.yaml \
  --test \
  --num_envs 4 \
  --motion_file "$VIZ_DATA" \
  --checkpoint \
    checkpoints/theia_server_full/finetune/theia_smplx/nn/mimic.pth
```

无显示服务器以 `episodes.csv`、`summary.json` 和必要时保存的视频作为验收
依据。

## 12. 本机已经完成的验证

本机硬件/环境：

```text
GPU: NVIDIA GeForce RTX 4080 SUPER, 16 GB
driver: 580.159.03
PyTorch: 2.4.1+cu124
torch CUDA runtime: 12.4
Isaac Gym GPU PhysX: enabled
```

已通过：

- CUDA tensor 计算和 Isaac Gym import。
- GPU PhysX pipeline。
- 512 env，从随机权重完成 1 epoch bootstrap 并保存 checkpoint。
- 512 env，从 bootstrap 完整恢复 optimizer，epoch `1 -> 2`、frame
  `16384 -> 32768`。
- Start/full-sequence finetune，自动 rollout length `339`。
- 使用两个只读硬链接模拟两条 motion：两序列资产绑定、并行 FK、PPO 训练均
  通过，约 2653 FPS。
- 全 motion 并行 FK：
  - max position error 约 `12.9 mm`
  - max rotation error 约 `8.81°`
- 16 episode 最新 policy GPU 回归：
  - completion `16/16`
  - semantic success `16/16`
- 一键脚本缩短版完整链路：
  - `BOOTSTRAP_EPOCHS=1`
  - `FINETUNE_EPOCHS=1`
  - 自动训练、完整恢复、最终评测全部通过。
- 使用同一缩短命令第二次启动：
  - bootstrap 显示 `[SKIP]`
  - finetune 显示 `[SKIP]`
  - 重新生成独立 timestamp 评测目录。

缩短版从随机权重只训练 2 epochs，最终成功率为 0 是预期现象；该测试只证明
一键控制流和恢复机制，不是 policy 性能证据。

## 13. 单序列回归和接触 A/B 结论

本机只有一条 339 帧 CupBlue+KettleGreen 序列。

选出的单序列严格 CPU actor-pair checkpoint：

```text
checkpoints/theia_local_verified/theia_smplx/nn/
  mimic_semantic_97_66_cpu_exact.pth
SHA-256:
b8ab8d47195dabb4996735db27eee12d938acacf2a69ff0424bc60f415ec7cd3
```

128 episode CPU PhysX 结果：

```text
completion: 127/128
semantic success: 125/128 = 97.66%
true intended dual contact: 128/128
wrong-contact steps: 0
```

这证明本机修复后的物理和控制链路有效，但不能证明对服务器新序列的零样本
泛化。因此此 checkpoint 只用于本机 regression/visualization，不用于服务器
正式初始化。

接触 A/B 使用强 checkpoint 微调并同时改变多个变量：

```text
soft final:          semantic 123/128
soft reward-best:    semantic 123/128
legacy reward-best:  semantic 125/128
```

因此没有证据支持在服务器从零训练中用 soft 替换原始 multiplicative contact。

## 14. 故障处理顺序

### CUDA 不可见

先运行：

```bash
nvidia-smi
ls -l /dev/nvidiactl /dev/nvidia0
```

如果是 Docker，必须使用 GPU passthrough，例如 `--gpus all`。不要通过重装
仓库 Python 包掩盖缺少 `/dev/nvidia*` 的问题。一键 launcher 也会在创建环境
前强制检查 CUDA。

### 数据 preflight 失败

读取 `data_manifest.json -> errors`，修数据、文件名或资产。不要直接关闭：

- quaternion 检查
- contact mapping 检查
- bimanual contact 检查
- foot collision 检查
- object asset 检查

物体 density 缺失已经不再是错误。

### FK 失败

报告 motion id、frame、body、max position/rotation error。优先确认 converter、
人物 skeleton 和 joint order。不要简单提高阈值超过 `1.5 cm / 10°`。

### CUDA OOM

先降低 `TARGET_ENVS`，但不能小于序列数和 PPO 最低 512。若序列数本身超过
单卡容量，当前固定资产绑定架构无法真正一键全量训练；必须报告并设计按
object-pair 共享环境或多卡方案，不能静默只训练子集。

### 最终成功率不足

先区分：

- completion failure
- final position failure
- final rotation failure
- contact requirement failure
- 单个坏序列/坏资产

数据正确后先增加 `FINETUNE_EPOCHS`。不要先改网络容量或奖励。

## 15. 关键文件 SHA-256

服务器收到相同工作树时应匹配：

```text
ac85099860a9884d539c48d2ad1cfdc9a20e51b44502bc242fdd267413bd2670  isaacgym/scripts/run_theia_server.sh
2bebe91acce8fca022e5a8892e74f4da1c32ca25051d7f33dd36891b9589422c  isaacgym/scripts/train_theia_full.sh
d92f6ee77ed4c92a5f6315ac9ac9baaa99f0ce5ac4937fab5d1aeaf3196e235e  isaacgym/scripts/eval_theia.sh
2bfd34090497d188eada260e9a7a260555eb1c30359d3a9f307fd79537f24f91  isaacgym/scripts/validate_theia_dataset.py
456078567250e16db0e6ef420938740c9f2970e69e01d3188dbea0d89fa55746  isaacgym/src/intermimic/data/cfg/theia_full_train.yaml
0f11421e93d0ba98c48036169ef37b9390dc30266e635e9f004ba8d5e21f6fe5  isaacgym/src/intermimic/data/cfg/theia_full_finetune.yaml
7b755d3fa6b0bd2339f12f3e09f551b0d8fdb520354cdaf9abcf1cc43bd176c0  isaacgym/src/intermimic/data/cfg/theia_eval.yaml
fb463f9f75a4d02b6159b1bbcac0add5a0ca243dedf08a08de193d1fed80b12d  isaacgym/src/intermimic/data/cfg/train/rlg/theia.yaml
564c97701f84e7d58a765499c1fbbcb5690652800bdf330bb0a32b2f23a98f94  isaacgym/src/intermimic/env/tasks/intermimic.py
7793c05ad749fde3eae5d9c2eb6c8ba9dd7a8850c5f4f70a2e99ca39aadbf12b  isaacgym/src/intermimic/learning/intermimic_agent.py
77ff2204bab8589a86922c617264a4cfc12f6199ec96f91a5b6e014149f732c9  isaacgym/src/intermimic/learning/intermimic_players.py
```

核对命令：

```bash
sha256sum \
  isaacgym/scripts/run_theia_server.sh \
  isaacgym/scripts/train_theia_full.sh \
  isaacgym/scripts/eval_theia.sh \
  isaacgym/scripts/validate_theia_dataset.py \
  isaacgym/src/intermimic/data/cfg/theia_full_train.yaml \
  isaacgym/src/intermimic/data/cfg/theia_full_finetune.yaml \
  isaacgym/src/intermimic/data/cfg/theia_eval.yaml \
  isaacgym/src/intermimic/data/cfg/train/rlg/theia.yaml \
  isaacgym/src/intermimic/env/tasks/intermimic.py \
  isaacgym/src/intermimic/learning/intermimic_agent.py \
  isaacgym/src/intermimic/learning/intermimic_players.py
```

## 16. 服务器 agent 的首次执行清单

按顺序执行：

1. 阅读本文全文。
2. 确认没有已有训练进程，或先识别其 `OUTPUT_ROOT` 和 checkpoint。
3. 确认当前提交包含 HANDOFF、一键脚本和全部关键修复，不回退旧基线。
4. 核对第 15 节关键文件 hash。
5. 运行 `nvidia-smi`。
6. 确认服务器数据目录和对象资产目录。
7. 统计序列数、长度 min/max、对象数和人物数。
8. 直接运行一键命令；不要先自行改变 reward/config。
9. 持续监控：
   - `server_run.log`
   - GPU utilization/memory
   - reward 和 reset rate
   - checkpoint epoch/frame 是否增长
10. 中断后运行同一命令恢复。
11. 训练结束后读取 `summary.json` 和 `episodes.csv`。
12. 汇报总体成功率、最差序列、失败类型和最终 checkpoint SHA-256。
13. 只有达到门槛后，才可称为服务器全数据集候选 policy。

## 17. 完成定义

服务器端任务只有同时满足以下条件才算完成：

- 全部预期序列进入 manifest，没有静默 subset。
- data/FK/CUDA 检查全部通过。
- Stage A 和 Stage B checkpoint 都存在。
- Stage B checkpoint epoch 达到目标。
- 最终评测 `actual_episodes == expected_episodes`。
- 总体 semantic success 达到设定门槛。
- 最差序列达到设定门槛。
- 保存 checkpoint、代码/config/data hashes、训练日志和逐 episode CSV/JSON。
- 最终汇报明确区分 GPU proxy contact 与 CPU actor-pair contact。
