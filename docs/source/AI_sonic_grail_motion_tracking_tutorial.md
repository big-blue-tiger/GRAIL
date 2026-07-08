# SONIC 与 GRAIL Motion Tracking 教学笔记

这份笔记面向已经熟悉 IsaacLab 的读者，所以不会重复解释
`ManagerBasedRLEnv`、observation/reward/action manager 的基本机制，而是直接看
GRAIL 怎样把 retarget 后的 4D HOI 数据接到 `imports/SONIC` 里，并在 SONIC 的预训练
whole-body controller 上扩展出两类 task-general tracker：

- **物体感知控制器**：object-aware latent adaptor，用于 pick-up 和 whole-body manipulation。
- **地形/场景感知控制器**：scene-aware tracker，用于 stairs、curbs、slopes、sitting 等场景交互。

论文里的关键思想是：GRAIL 不为每条轨迹单独训练一个控制器，而是把同一任务族的
4D HOI 轨迹合成一个 motion pool，在 SONIC 这个预训练全身控制器上做互补适配。
物体线保留 SONIC 的身体运动先验，只训练一个“外挂”的物体感知 adaptor；地形线则把
controller 和 height-map encoder 一起微调，让机器人能根据局部几何调整全身运动。

## 1. `imports/SONIC` 项目框架

`imports/SONIC` 是 GRAIL vendored 进来的 SONIC/GEAR whole-body control 工程。对
GRAIL motion tracking 最重要的是下面几块：

| 路径 | 作用 |
| --- | --- |
| `gear_sonic/` | 训练主栈：Hydra 配置、IsaacLab env、MDP terms、PPO trainer、policy module、motion library。GRAIL 的 task-general tracking 基本都在这里跑。 |
| `gear_sonic/config/exp/manager/universal_token/` | 训练实验入口。GRAIL 相关配置分在 `hoi/` 和 `scene/`。 |
| `gear_sonic/envs/manager_env/` | IsaacLab manager env 的封装：机器人、场景对象、observation/reward/action/termination term。 |
| `gear_sonic/envs/wrapper/manager_env_wrapper.py` | 环境外层 wrapper。物体线的 66 维 meta-action 在这里被拆成 SONIC latent residual 和手部 primitive，再合成为 43 DOF joint target。 |
| `gear_sonic/trl/modules/` | 网络模块。`UniversalTokenModule` 是 SONIC token encoder/FSQ/decoder 主体；`BaseModuleAux` 是物体 adaptor 的 MLP。 |
| `gear_sonic/utils/motion_lib/` | motion library 加载器。它把 GRAIL retarget 输出的 `robot/`、`objects/`、BPS、contact label 等变成训练时的 reference。 |
| `gear_sonic_deploy/` | C++/ONNX/TensorRT 部署栈，用于真实机器人或 MuJoCo sim2sim 部署。理解训练逻辑时可以先放一边。 |
| `decoupled_wbc/` | 早期/部署侧 decoupled whole-body control、遥操作和数据采集工具。和 GRAIL 训练配置不是主线，但能看到 teleop/data collection 的接口。 |
| `models/` | 下载后的 checkpoint bundle，例如 `sonic_manipulation_base/`、`pnp_table/`、`pnp_ground/`、`terrain_stairs/`。 |

训练入口通常是：

```bash
cd imports/SONIC
python -u gear_sonic/train_agent_trl.py +exp=manager/universal_token/hoi/pnp_table ...
python -u gear_sonic/train_agent_trl.py +exp=manager/universal_token/scene/terrain_tracking ...
```

`train_agent_trl.py` 会先创建 manager env，reset 一次拿到各 observation group 的真实
shape，再把这些维度回填到 `env.config["obs"]` 和
`env.config["robot"]["algo_obs_dim_dict"]`。如果配置里设置了
`manager_env.config.meta_action_dim`，训练脚本会把 policy 的动作维度改成这个值，而不是
IsaacLab 环境原生 action space。这就是物体线能让 actor 输出 66 维 meta-action 的原因。

## 2. GRAIL 到 SONIC 的数据流

从 GRAIL 论文和代码看，motion tracking 前的数据链是：

1. GRAIL 先生成或收集 3D 资产、场景和视频先验。
2. `grail.pipelines.recon_4dhoi` 从生成视频中恢复 metric 4D human-object interaction。
3. `grail.retargeting.retarget` 把 SMPL-X/SOMA 风格人体运动 retarget 到 Unitree G1。
4. `grail.retargeting.process` 进一步生成训练友好的 hand action、contact points、meta/table 信息。
5. `grail.retargeting.compute_bps` 为每个 object USD 生成 BPS shape encoding。
6. SONIC 的 `MotionLibBase` / `MotionLibRobot` 在训练时加载这些数据。

训练期最常见的数据目录是：

```text
data/motion_lib/<name>/
├── robot/<seq>.pkl        # G1 29-DOF body trajectory, optional hand_action_* / hand_dof_pos
├── objects/<seq>.pkl      # object root_pos/root_quat/contact_points
├── object_usd/<seq>.usd   # IsaacLab 中 spawn 的对象或场景 USD
├── meta/<seq>.pkl         # table pose / scene scale 等可选信息
└── bps/<seq>.npy          # object shape BPS, 物体线使用
```

在 `MotionLibBase` 里，`motion_file` 指向 `robot/`，`object_motion_file` 指向
`objects/`，`bps_dir` 指向 BPS 目录。加载后会形成几类核心张量：

- reference robot body motion：root/body/joint position、orientation、velocity。
- reference object motion：`object_root_pos`、`object_root_quat`，以及多未来帧版本。
- contact label 和 contact center：由 `contact_points_left_hand/right_hand` 插值得到。
- hand action：`hand_action_left/right`，约定 `-1.0 = open`，`+1.0 = closed`。
- object BPS：按 motion key 查表，作为静态 shape descriptor。

IsaacLab 场景对象则由 `ModularTrackingEnvCfg` 创建。对普通 pick-up/manipulation，它会从
`manager_env.config.object_usd_path` 加载动态 rigid object；对 terrain/sitting，它也把
terrain/chair 作为 `Object` USD 放进场景，只是通常是静态或 kinematic object。这样一来，
同一套 command/observation/reward 可以同时访问“参考轨迹中的 object state”和“仿真中当前
object state”。

## 3. SONIC 原始控制器：universal token motion tracking

SONIC 的核心网络在 `gear_sonic/trl/modules/universal_token_modules.py`。它把不同来源的
motion target 编成统一 token，再用 decoder 产生 joint-level action：

```text
tokenizer observation
  -> encoder, e.g. G1/SMPL/teleop/SOMA
  -> FSQ quantizer
  -> decoder with proprioception
  -> action mean
```

GRAIL 主要用的是 G1 encoder 路线。基础 SONIC 目标是 29 DOF G1 body control：腿、腰、手臂和
手腕，不包括 dexterous fingers。它本身擅长大规模人体动作跟踪，提供稳定的 locomotion 和全身平衡
先验。GRAIL 的两个 tracker 都是在这个基础上做扩展：

- 物体线不直接改 SONIC 主体，而是在 latent/token 空间加 residual，并外挂手部动作。
- 地形线把 height map 作为额外 tokenizer input，让 SONIC 自身学会 scene-conditioned tracking。

## 4. 物体感知控制器：Object-Aware Latent Adaptor

### 4.1 对应论文思路

论文 Sec. 3.3 / Appendix B.1 里，object-aware tracker 的形式是：

```text
(Δz_t, a_hand_t) = π_phi(s_t, o_t)
a_body_t = G(z_t + λ Δz_t)
```

其中：

- `z_t` 是 SONIC encoder/FSQ 得到的运动 token。
- `Δz_t` 是 adaptor 输出的 64 维 latent residual。
- `λ = 0.1`，residual 在进入 FSQ 前缩放。
- `a_hand_t` 是 2 维左右手 open/close primitive，每只手映射到 7 个 finger DOF。
- SONIC 的 encoder、quantizer、decoder 冻结，只训练 adaptor `π_phi`。

这样做的意图很明确：不要让 manipulation 训练破坏 SONIC 已经学好的 locomotion/balance
prior，只在 token 上做小幅修正，同时补上 SONIC 基础身体模型没有的手指闭合能力。

### 4.2 代码入口

物体线的 release config 在：

```text
imports/SONIC/gear_sonic/config/exp/manager/universal_token/hoi/
├── pnp_table.yaml
├── pnp_ground.yaml
├── advanced_manip_table.yaml
└── advanced_manip_ground.yaml
```

它们共享的关键配置是：

- `robot.type: g1_43dof`：29 body DOF + 14 hand DOF。
- `actor_critic: hoi_staged_mlp_aux`：actor 是 3 层 MLP，hidden dims `[512, 256, 128]`，SiLU。
- `meta_action_dim: 66`：前 64 维是 token residual，后 2 维是左右手 primitive。
- `tokenizer_action_dim: 64`，`hand_action_dim: 2`。
- `use_latent_residual: true`，`latent_residual_mode: pre_quantization`，`latent_residual_scale: 0.1`。
- `action_transform_module_cfg/checkpoint: models/sonic_manipulation_base/...`：加载冻结的 SONIC base controller。
- observation 使用 `hoi_manip/grab_43dof_policy_mf_delta_bps`。

注意一个 release 代码细节：论文概念上 adaptor 输出 hand primitive；当前配置里
`pnp_table.yaml` 设置了 `use_motion_hand_actions: true`，wrapper 会用 motion data 里的
`hand_action_left/right` 覆盖 actor 输出的最后 2 维。`pnp_ground` 和 advanced manipulation
配置则是让策略输出 primitive。阅读结果或改配置时要看这个开关。

### 4.3 Observation：物体感知来自哪里

`grab_43dof_policy_mf_delta_bps.yaml` 把物体感知拆成几类：

- proprioception：joint pos/vel、base lin/ang vel、previous action。
- motion target：当前 command、多未来帧 command、motion anchor pose/orientation。
- current object state：object position + 6D orientation in robot body frame。
- object future delta：reference future object pose 减去 simulated current object pose。
- hand-object state：hand-to-object transform、fingertip contact forces。
- table state：table position/orientation in body frame。
- shape：BPS object encoding，默认 10 维。

最关键的是 **object delta observation**。它不是只告诉策略“参考 object 在哪里”，而是告诉策略：
“从当前仿真里的物体状态出发，未来若干帧应该往哪里走/怎么转”。对应实现：

- `object_pos_delta_multi_future`：把 simulated current object position 和 reference future object
  position 都转到 robot body frame，再做 `ref_future - sim_current`。
- `object_ori_delta_multi_future_6d`：做 `q_current^-1 * q_ref_future`，再转 6D rotation。

这使得策略能闭环修正 object drift，而不是开环照抄 reference。

### 4.4 Action：66 维 meta-action 怎样变成 43 DOF

动作转换发生在 `ManagerEnvWrapper.step()`：

1. actor 输出 `meta_actions`，shape 是 `(num_envs, 66)`。
2. wrapper 拆分：
   - `meta_actions[:, :64]` -> `tokenizer_meta_actions`
   - `meta_actions[:, 64:]` -> `hand_actions_raw`
3. 如果启用 `use_motion_hand_actions`，用 motion lib 的 `hand_action_left/right` 覆盖手部输出。
4. 如果启用 `use_finger_primitive`，把每个 primitive 映射成 7 个 finger joint target：
   - `discrete` 模式下，action >= 0 表示 closed，action < 0 表示 open。
   - 左右手各 7 DOF，总计 14 DOF。
5. 对身体部分，wrapper 调用冻结的 `action_transform_module`：
   - 它先用 SONIC encoder 编出 latent。
   - 加上 `0.1 * residual`。
   - 在 `pre_quantization` 模式下先相加再 FSQ quantize。
   - decoder 输出 29 DOF body action。
6. wrapper 把 29 DOF body action 和 14 DOF hand action 按 joint index 合并成 43 DOF env action。

这就是“基于 SONIC 外挂手部”的代码实现：核心 SONIC decoder 仍产生 29 DOF body target，手指由
GRAIL 的 wrapper/adaptor 在外层补齐。

### 4.5 Reward 与 termination

物体线 reward 由几组项构成：

- motion tracking reward：root/body/joint pose、orientation、velocity 等跟踪项。
- wrist 权重增强：object-aware config 会提高 wrist link tracking 权重，鼓励手部对准。
- object tracking reward：比较 simulated object pose 和 reference object pose。
- grasp reward：基于 finger-object contact sensor，接触数达到阈值后饱和。
- grasp finger direction：鼓励 thumb/index/middle 从相对方向形成 pinch。
- contact center：把 fingertip 拉向 motion data 中的 object contact centroid。
- regularization：meta action rate、full latent rate、finger primitive limit、joint/action penalty 等。

termination 侧包括：

- object position 或 z-position 偏离 reference 超阈值。
- root/body tracking error 超阈值。
- table task 中 hand-table contact。
- contact phase 开始后若在 grace frames 内没形成有效 grasp，则 terminate。

这些项对应论文里的 object pose term、contact-gated grasp term 和 interaction early termination。

## 5. 地形/场景感知控制器：Scene-Aware Tracker

### 5.1 对应论文思路

论文 Sec. 3.3 / Appendix B.2 里，scene-aware tracker 针对 stairs、curbs、slopes、sitting 等场景。
这些任务没有明确手-物抓取，但有强烈的 body-scene 约束：脚要落在台阶上，身体要跨过障碍，坐下时
root/torso 要对齐椅子。

因此它不是外挂 latent adaptor，而是：

- 给 SONIC encoder 增加局部 height map。
- 端到端微调 controller 的 encoder、FSQ、decoder 和 height-map encoder。
- 使用 motion tracking + regularization reward，不启用 manipulation-specific grasp/object rewards。

论文描述的 height map 是以机器人为中心的 11x11 网格，范围 1.5 m，分辨率 0.15 m。每个网格点向下
raycast 到 scene mesh，得到 robot body frame 下的局部几何。

### 5.2 代码入口

地形线 release config 是：

```text
imports/SONIC/gear_sonic/config/exp/manager/universal_token/scene/terrain_tracking.yaml
```

关键配置：

- `actor_critic: universal_token/single_mlp_hmap_proj`
- `robot.type: g1_model_12`：29 DOF body controller，无 dexterous hand。
- `add_object: true`：scene/terrain/chair 作为 object USD 加入场景。
- `terrain_motion_dir`：dataset root，要求能自动发现 `robot/*.pkl` 和 `object_usd/*.usd` 的 1:1 stem 匹配。
- `enable_depth_camera: false`：当前 release 不用 depth camera。
- `commands.motion.use_height_map: true`
- tokenizer observation 使用 `single_token_noz_hmap_proj`。
- policy observation 使用 `local_dir_hist_obj`，仍包含 object pose/delta，因为 chair/terrain 也作为 scene object 表达。

论文附录里 height encoder 写的是 3 层 CNN `[64, 128, 256]` 并输出 1024 维特征；当前 release 配置
`single_mlp_hmap_proj.yaml` 是一个更轻的 conv2d projector：`input_shape: [1, 11, 11]`，
channels `[16, 32]`，maxpool 后接 MLP，输出 128 维。理解论文时看方法思想；复现实验时以配置文件为准。

### 5.3 Height map 怎样生成

height map 的生成在 `TrackingCommand` 中完成：

1. 初始化时，如果 `use_height_map` 为 true，创建 `MultiMeshRaycaster`。
2. raycaster 只看每个 env 的 `/World/envs/env_<id>/Object`，也就是 terrain/chair/scene USD。
3. 构造 11x11 grid：
   - `height_map_size = 1.5`
   - `height_map_resolution = 0.15`
   - 所以每个方向 `1.5 / 0.15 + 1 = 11` 个点。
4. 每一步 `_update_command()`：
   - 从 robot root 出发，按 robot yaw 旋转局部 ray direction。
   - 对 scene object 做 raycast。
   - 将 hit point 限制在地面以上。
5. observation `height_map_z_flat` 取 `(B, 11, 11, 3)` 中的 z 通道并 flatten 成 `(B, 121)`。

注意这里的“地形感知”不是视觉 RGB/depth policy，而是 physics tracking 阶段直接使用 scene mesh
几何生成的局部 height map。论文后续的 sim-to-real 部分会把 tracking policy distill 到 egocentric
visual policy，那是另一个阶段。

### 5.4 Network 与 observation flow

`single_token_noz_hmap_proj.yaml` 的 tokenizer terms 包括：

- `encoder_index`
- `command_multi_future_nonflat`
- `command_z_multi_future_nonflat`
- `motion_anchor_ori_b_mf_nonflat`
- `height_map_z_flat`

`single_mlp_hmap_proj.yaml` 里，G1 encoder 的 inputs 是：

```text
command_multi_future_nonflat
motion_anchor_ori_b_mf_nonflat
height_map_z_flat
```

其中 `height_map_z_flat` 会先过 conv2d projector，再与其它 tokenizer feature 拼接。actor 仍然是
`UniversalTokenModule`，所以它不是像物体线那样另起一个 adaptor MLP，而是在 SONIC token pipeline
内部加入 terrain context。

policy observation `local_dir_hist_obj` 还给 actor/decoder 提供常规 proprioception 和 object/scene
relative state。critic 使用 `privileged_mf_hist_obj`，包含更完整的 privileged tracking state。

### 5.5 Scene data pairing

terrain config 支持两种数据加载方式：

- 新布局：`terrain_motion_dir=<dataset_root>`，自动扫描：
  - `<dataset_root>/robot/**/*.pkl`
  - `<dataset_root>/object_usd/**/*.usd`
  - 两者必须同 stem，例如 `stairs_000.pkl` 对应 `stairs_000.usd`。
- 旧布局：显式传 `terrain_usd_path` 和 `terrain_motion_keys_path`。

`ModularTrackingEnvCfg` 会按 rank 做 USD 和 motion key slicing，并写出
`/tmp/rank_<rank>_motion_keys.txt`，随后 `TrackingCommand` 用这个文件过滤 motion lib。这样多 GPU
训练时，每个 rank 的 terrain USD 和 motion reference 可以保持配对。

`flat_to_terrain_ratio` 用来混入 flat motions。`R=0` 表示全 terrain；`R=3` 表示每 4 个 env 中 1 个
terrain、3 个 flat。GRAIL 当前 release config 默认 `flat_to_terrain_ratio: 0`。

## 6. 两个 tracker 的核心差异

| 维度 | 物体感知控制器 | 地形/场景感知控制器 |
| --- | --- | --- |
| 任务 | pick-up、table/ground manipulation | stairs、curbs、slopes、sitting |
| 机器人 | `g1_43dof`，29 body + 14 finger | `g1_model_12`，29 body |
| SONIC 使用方式 | 冻结 SONIC base，外接 adaptor 输出 residual + hand primitive | 直接微调 SONIC token pipeline，并加入 height-map projector |
| actor | `hoi_staged_mlp_aux`，3 层 MLP | `UniversalTokenModule` with `single_mlp_hmap_proj` |
| 动作 | 66 维 meta-action -> 29 body + 14 hand | 29 维 body joint target |
| 感知 | object pose、future object delta、BPS、hand-object transform、contact force、table pose | local height map z、scene/object pose/delta、proprioception |
| reward | motion + object tracking + grasp/contact + regularization | motion tracking + regularization |
| checkpoint | `models/sonic_manipulation_base/` 作为 frozen action transform module | 可从 `models/terrain_stairs/` 等 checkpoint warm resume |

一句话概括：

- 物体线是“**冻结身体大脑，外挂一个物体/手部小脑**”。
- 地形线是“**把地形几何喂进 SONIC，让全身 tracker 自己适应场景**”。

## 7. 推荐代码阅读路线

如果你想从代码而不是论文复盘一遍，建议按这个顺序：

1. 训练入口：
   - `imports/SONIC/gear_sonic/train_agent_trl.py`
   - 看 env 创建、observation shape 回填、`meta_action_dim` 覆盖。
2. Hydra 实验配置：
   - 物体：`gear_sonic/config/exp/manager/universal_token/hoi/pnp_table.yaml`
   - 地形：`gear_sonic/config/exp/manager/universal_token/scene/terrain_tracking.yaml`
3. 环境骨架：
   - `gear_sonic/config/manager_env/base_env.yaml`
   - `gear_sonic/envs/manager_env/modular_tracking_env_cfg.py`
4. motion command：
   - `gear_sonic/envs/manager_env/mdp/commands.py`
   - 重点看 `TrackingCommand`、height map 初始化、`_update_command()`。
5. motion library：
   - `gear_sonic/utils/motion_lib/motion_lib_base.py`
   - 看 `object_motion_file`、`bps_dir`、hand action、contact center 如何加载。
6. observation terms：
   - `gear_sonic/envs/manager_env/mdp/observations.py`
   - 重点看 `object_bps`、`object_pos_delta_multi_future`、`object_ori_delta_multi_future_6d`、`height_map_z_flat`。
7. 网络：
   - 物体：`gear_sonic/config/actor_critic/hoi_staged_mlp_aux.yaml`
   - 地形：`gear_sonic/config/actor_critic/universal_token/single_mlp_hmap_proj.yaml`
   - SONIC 主体：`gear_sonic/trl/modules/universal_token_modules.py`
8. action wrapper：
   - `gear_sonic/envs/wrapper/manager_env_wrapper.py`
   - 看 `step()` 中 meta-action 拆分、latent residual、finger primitive、43 DOF 合并。
9. reward/termination：
   - `gear_sonic/envs/manager_env/mdp/rewards.py`
   - `gear_sonic/envs/manager_env/mdp/terminations.py`

## 8. 常见误解与调试提示

**误解 1：物体线的 actor 直接输出机器人 43 DOF。**

不是。actor 输出 66 维 meta-action。wrapper 先用冻结 SONIC 把前 64 维 residual 转成 29 DOF body
action，再把后 2 维 primitive 映射成 14 DOF fingers。

**误解 2：地形线用了深度相机。**

当前 release config 明确 `enable_depth_camera: false`。训练时用的是 scene mesh raycast 得到的
height map。视觉 policy 是后续 sim-to-real distillation 阶段。

**误解 3：object pose observation 只是 reference pose。**

关键项是 future delta：reference future object pose 相对 simulated current object pose 的差。这让
policy 能对仿真中的 object drift 做闭环纠偏。

**误解 4：论文架构尺寸和当前 YAML 一定完全一致。**

方法是一致的，但 release 配置可能更轻或经过工程化调整。比如论文 Appendix B.2 描述了 3 层
height-map CNN 输出 1024 维；当前 `single_mlp_hmap_proj.yaml` 用 2 层 conv2d projector 输出 128 维。

**调试 1：先看 Hydra 最终配置。**

很多关键开关都在 config override 中，例如 `object_usd_path`、`motion_lib_cfg.motion_file`、
`object_motion_file`、`bps_dir`、`terrain_motion_dir`。路径通常按 `cd imports/SONIC` 后的相对路径解析。

**调试 2：物体线先确认 BPS 是否加载。**

`MotionLibBase` 会打印 `[BPS] Loaded ...`。如果 `bps_dir` 不存在，`object_bps` 会退化成 zeros。

**调试 3：地形线看三类日志。**

terrain auto-discovery 会打印 `[TerrainAutoDiscover]`、`[PerRankUSD]`、`[PerRankMotion]`。它们能确认
USD 和 motion key 是否按 rank 配对。

**调试 4：43 DOF 顺序问题优先查 joint mapping。**

body joint 和 hand joint indices 在 `gear_sonic/envs/env_utils/joint_utils.py`，wrapper 合并 action 时依赖
这些 index。motion library 内部还会处理 MuJoCo order 与 IsaacLab order 的映射。

## 9. 与论文的对应关系

论文主文 Sec. 3.3 讲的是完整思想：GRAIL 先用 robot-proportioned reconstruction 和 GMR 得到 G1
joint-space reference，再训练 SONIC-based task-general trackers。Appendix B 给了观测、网络、reward 和
训练细节。

代码中的对应关系可以这样记：

- `π_phi` object adaptor -> `hoi_staged_mlp_aux.yaml` + `ManagerEnvWrapper.action_transform_module`
- `Δz_t` -> actor 输出前 64 维，wrapper 作为 `latent_residual`
- `a_hand_t` -> actor 输出后 2 维，或由 `use_motion_hand_actions` 从 motion data 覆盖
- `BPS(o)` -> `grail/retargeting/compute_bps.py` 生成，`MotionLibBase.get_object_bps()` 读取
- `h_t` height map -> `TrackingCommand` raycast，`height_map_z_flat` observation
- `epsilon_h` height encoder -> `single_mlp_hmap_proj.yaml` 的 conv2d `input_projectors`
- object future delta -> `object_pos_delta_multi_future` / `object_ori_delta_multi_future_6d`
- task-general motion pool -> `motion_lib_cfg.motion_file` + `object_motion_file` + rank-aware filtering

掌握这张对应表后，再看 GRAIL 的 tracking 代码就不会被 Hydra 层级和 SONIC 的 token 命名绕晕。
