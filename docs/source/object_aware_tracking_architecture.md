# Object-Aware Tracking 架构与 DiT 扩展接口梳理

本文依据 GRAIL 论文第 3.3 节、附录 B.1、表 5/6，并结合本仓库当前代码，梳理 Object-Aware Tracking（物体感知跟踪）的网络、输入输出、奖励、训练数据流和扩展边界。目标是为后续加入 DiT 类模仿学习策略时提供一份可执行的接口说明，而不是仅复述论文。

> 代码版本说明：本文对应当前工作区中的 `imports/SONIC` vendored release。论文描述、默认 term 配置和 `pnp_*` 发布实验配置并不处处相同；下文明确标注三者差异。训练的主实现位于 `imports/SONIC`，顶层 `grail/` 主要负责数据生成、重建、重定向、筛选和导出。

## 1. 一句话设计

Object-Aware Tracking 不重新训练 SONIC 全身控制器，而是在冻结的 SONIC 编码器—FSQ—动作解码器外增加一个小型 actor：actor 根据机器人状态、未来参考动作、当前/未来物体状态、接触和物体形状，预测一个 64 维 latent residual；该残差以 0.1 缩放后加到 SONIC 编码器的量化前 latent 上，再由冻结的 FSQ 和 decoder 生成 29 个身体关节目标。理论上 actor 还预测左右手各 1 个开/合 primitive，并映射为每手 7 个手指关节目标。

```text
retargeted robot/object motion + BPS + contact labels
                         │
                         ▼
       actor observation / privileged critic observation
                         │
          Object-Aware actor MLP: [512, 256, 128], SiLU
                         │
                meta-action ∈ R^66
                  ┌──────┴──────┐
                  │             │
             Δz ∈ R^64     hand primitive ∈ R^2
                  │             │
             × λ, λ=0.1     threshold / lookup table
                  │             │
SONIC encoder(q_ref) + Δz        14 finger targets
                  │
            FSQ quantizer (frozen)
                  │
        SONIC action decoder (frozen)
                  │
          29 body joint targets
                  └──────┬──────┘
                         ▼
              G1 43-DoF position targets
```

对应的总入口配置是 [`pnp_table.yaml`](../../imports/SONIC/gear_sonic/config/exp/manager/universal_token/hoi/pnp_table.yaml)，actor/critic 结构在 [`hoi_staged_mlp_aux.yaml`](../../imports/SONIC/gear_sonic/config/actor_critic/hoi_staged_mlp_aux.yaml)，meta-action 到 43-DoF 动作的拼装在 [`manager_env_wrapper.py`](../../imports/SONIC/gear_sonic/envs/wrapper/manager_env_wrapper.py)。

## 2. 网络架构

### 2.1 冻结的 SONIC 主干

论文中的基础控制器可写为：

\[
z_t = \mathcal{E}(q_t^r),\qquad
a_t^{body}=\mathcal{G}(Q(z_t+\lambda\Delta z_t)),\quad \lambda=0.1,
\]

其中 `E` 是参考动作编码器，`Q` 是 finite scalar quantization（FSQ），`G` 是动作 decoder。当前配置通过：

- `action_transform_module_cfg: models/sonic_manipulation_base/model_config.yaml`
- `action_transform_module_checkpoint: models/sonic_manipulation_base/last.pt`
- `use_latent_residual: true`
- `latent_residual_mode: pre_quantization`
- `latent_residual_scale: 0.1`

加载并调用主干。量化前残差相加的实际代码位于 [`universal_token_modules.py`](../../imports/SONIC/gear_sonic/trl/modules/universal_token_modules.py) 的 `forward(..., latent_residual, latent_residual_mode)`：先以 `(batch, token_num, token_dim)` reshape 64 维 residual，与未量化 encoder latent 相加，再经过 quantizer。当前 token 形状为 `2 × 32 = 64`，decoder 输出 29 个无手指身体关节目标。

### 2.2 Object-Aware actor

actor 是 3 层 MLP：

| 项目 | 当前设计 |
|---|---|
| 隐藏层 | `[512, 256, 128]` |
| 激活 | SiLU |
| 输出 | 66 维 meta-action |
| latent 部分 | 前 64 维，量化前 residual |
| hand 部分 | 后 2 维，左/右手 primitive |
| residual 正则 | `LatentL2Loss`，只取前 64 维 |

结构由 [`hoi_staged_mlp_aux.yaml`](../../imports/SONIC/gear_sonic/config/actor_critic/hoi_staged_mlp_aux.yaml) 实例化。`LatentL2Loss` 在 [`token_losses.py`](../../imports/SONIC/gear_sonic/trl/losses/token_losses.py) 中计算 `mean(action_mean[..., :64]^2)`，用于限制 adaptor 不要过度偏离预训练 locomotion prior。

### 2.3 critic

critic 是独立的 `[512, 256, 128]` SiLU MLP，输出标量 value。它使用 privileged observation：除 actor 输入外，还包含完整 body position/orientation 和数据集接触标志。配置见 [`grab_43dof_critic_mf_delta.yaml`](../../imports/SONIC/gear_sonic/config/manager_env/observations/critic/hoi_manip/grab_43dof_critic_mf_delta.yaml)。这些 privileged 字段只能用于 RL 训练的 value estimation，不能成为未来可部署 DiT policy 的必需输入。

### 2.4 手部 primitive 到 14 个关节

每个手部标量被离散化为 open/close，然后查表生成 7 个手指关节位置；左右手合计 14 维。具体关节顺序和 `pos_0/pos_1` 在各 `pnp_*` 配置的 `finger_primitive.primitive_action_map` 中定义。最终由 wrapper 将 29 维 body action 和 14 维 hand action按 Isaac Lab joint index 合并为 43-DoF 环境动作。

需要特别注意：论文写的是 sigmoid 后以二值信号开/合；代码路径 `_convert_primitive_to_finger_actions` 使用配置化阈值/离散映射，数据动作约定为 `-1=open, +1=closed`。新增策略必须复用 wrapper 的映射，不应在模型内部硬编码 14 个目标。

## 3. Actor 输入设计

actor observation 由 [`grab_43dof_policy_mf_delta_bps.yaml`](../../imports/SONIC/gear_sonic/config/manager_env/observations/policy/hoi_manip/grab_43dof_policy_mf_delta_bps.yaml) 按列出的顺序拼接。下面的维度为论文表 5 的语义维度；`joint_pos`、`joint_vel`、previous action 等 proprioception 在训练 wrapper 中使用 10 帧历史。未来参考共有 10 帧，时间间隔由 `dt_future_ref_frames: 0.1` 配置。

| 类别 | observation term | 维度 | 坐标系/语义 | 代码 |
|---|---|---:|---|---|
| proprioception | joint position | 43 | 相对默认关节角 | `joint_pos_rel` |
| proprioception | joint velocity | 43 | 关节角速度 | `joint_vel_rel` |
| proprioception | base linear velocity | 3 | body frame | `base_lin_vel` |
| proprioception | base angular velocity | 3 | body frame | `base_ang_vel` |
| proprioception | previous action | 66（论文）/ 43（当前 term） | 论文为上一步 meta-action；当前 `actions` term 调用 Isaac Lab `last_action`，对应执行到环境的 43-DoF action | 分层策略另有 `last_meta_action` 实现，接入时须核验最终 Hydra composition |
| reference | current command | 58 | 29 joint position + 29 joint velocity | `generated_commands` |
| reference | motion anchor position | 3 | target root position in body frame | `motion_anchor_pos_b` |
| reference | motion anchor orientation | 6 | target root 6D orientation in body frame | `motion_anchor_ori_b` |
| reference | multi-future command | 580 | 10 × (29 q + 29 qdot) | `command_multi_future` |
| reference | multi-future anchor orientation | 60 | 10 × 6D | `motion_anchor_ori_b_mf` |
| object | current object position | 3 | body frame | `object_pos_b` |
| object | current object orientation | 6 | body frame, 6D rotation | `object_ori_b_6d` |
| object | future position delta | 30 | 10 × (`ref_future - sim_current`) | `object_pos_delta_multi_future` |
| object | future orientation delta | 60 | 10 × relative 6D rotation | `object_ori_delta_multi_future_6d` |
| object | target object position | 3 | reference object position in base frame | `get_target_object_pos_in_base_frame` |
| shape | object BPS | 10 | 每物体静态 shape descriptor | `object_bps` |
| scene | table position | 3 | body frame | `table_pos_b` |
| scene | table orientation | 6 | body frame, 6D rotation | `table_ori_b` |
| interaction | hand-to-object transform | 9 | 当前实现只配 right hand；3 position + 6 orientation | `hand_object_transform_6d` |
| interaction | fingertip forces | 12（论文）/ 24（当前 `pnp_table`） | 论文按 4 个接触点；函数实际展平 `num_links × 3`，当前配置列出 palm + 7 个 finger segments，共 8 links | `get_finger_tips_contact_force` |

这些函数集中在 [`observations.py`](../../imports/SONIC/gear_sonic/envs/manager_env/mdp/observations.py)。`object_pos_delta_multi_future` 和 `object_ori_delta_multi_future_6d` 是物体跟踪的核心闭环量：它们不是简单的参考轨迹，而是把未来参考与当前仿真物体状态做差，使策略能够纠正已经发生的物体偏差。previous-action 和 fingertip-force 的当前维度也说明，论文表 5 不能直接当作 flat observation 的切片表；必须从 resolved config 和运行时 observation manager 获取最终 shape。

### 3.1 BPS 数据路径

[`motion_lib_base.py`](../../imports/SONIC/gear_sonic/utils/motion_lib/motion_lib_base.py) 从 `motion_lib_cfg.bps_dir` 加载每个 object stem 对应的 `.npy`，并按 motion/object key 构造 `_motion_object_bps`。未设置目录、目录不存在或 object stem 不匹配时，BPS 会退化为全零。因此 DiT 数据导出必须保存稳定的 `object_id/object_stem`，并在训练前检查 BPS 非零率和维度，而不能只检查 observation schema 中存在 `object_bps`。

### 3.2 Critic 特权输入

critic 在上述状态基础上增加：

- 完整 body positions 和 body orientations；
- `grab_contact_flag`（ground-truth contact label）；
- 同样的未来 body/object reference，但无 observation noise。

当前 critic 配置没有包含 `object_bps`。这符合“actor 需要 shape-conditioned action，critic 只评估状态价值”的设计，但未来若对不同形状的价值函数产生明显欠拟合，可以单独做 critic-BPS ablation，不要默认改变部署输入。

## 4. 输出和环境动作契约

### 4.1 论文目标接口

\[
(\Delta z_t, a_t^{hand})=\pi_\phi(s_t,o_t),\quad
\Delta z_t\in\mathbb{R}^{64},\quad a_t^{hand}\in\mathbb{R}^{2}.
\]

wrapper 在 [`manager_env_wrapper.py`](../../imports/SONIC/gear_sonic/envs/wrapper/manager_env_wrapper.py) 中执行：

1. 将 66 维输出拆成 `[:64]` 和 `[64:]`；
2. 将 residual 乘 0.1；
3. 以 `pre_quantization` 模式送入冻结 SONIC；
4. 将 2 维 primitive 转成 14 个 finger targets；
5. 合并 decoder 的 29 个 body targets，送出 43 个 joint position targets。

### 4.2 当前发布配置的实际行为

`pnp_table.yaml` 当前同时设置：

```yaml
use_motion_hand_actions: true
use_finger_primitive: true
meta_action_dim: 66
tokenizer_action_dim: 64
hand_action_dim: 2
```

当 `use_motion_hand_actions=true` 时，wrapper 会在拆分 meta-action 后，用 motion library 的 `hand_action_left/right` 覆盖网络预测的后 2 维。因此：

- 当前有效控制输出主要是 64 维 residual；
- 66 维张量的最后 2 维仍存在于 policy distribution、history/正则路径中，但不直接决定手指动作；
- 如果未来 DiT 训练 label 来自实际执行动作，必须明确 hand label 是“actor raw output”还是“motion override 后的 primitive”；二者不能混用；
- 若目标是端到端预测手部开合，应在实验配置中关闭 `use_motion_hand_actions`，并重新验证 finger primitive loss、探索噪声和 rollout 稳定性。

这是当前代码相对论文最重要的接口差异。

该差异不是所有 manipulation release config 的共同状态：当前 `pnp_ground.yaml`、`advanced_manip_table.yaml` 和 `advanced_manip_ground.yaml` 均设置 `use_motion_hand_actions: false`，只有所查的 `pnp_table.yaml` 开启 override。数据集必须记录其来源 config，不能按 task-family 猜测 hand action 语义。

## 5. 奖励函数

论文总奖励为：

\[
R_t=R_t^{motion}+R_t^{reg}+R_t^{obj}+\mathbb{1}[C_t]R_t^{grasp}.
\]

奖励的组合配置见 [`grab_rewards_no_lift_ucw_contact_lr_wrist_ori.yaml`](../../imports/SONIC/gear_sonic/config/manager_env/rewards/hoi_manip/grab_rewards_no_lift_ucw_contact_lr_wrist_ori.yaml)，实现集中于 [`rewards.py`](../../imports/SONIC/gear_sonic/envs/manager_env/mdp/rewards.py)。但当前 `pnp_table.yaml` 实际 override 的是 `grab_rewards_no_lift_ucw`，并进一步覆盖若干权重。因此阅读奖励时应以“Hydra 合成后的运行配置”为准。

### 5.1 Motion tracking

共同形式为高斯核：

\[
r_i=w_i\exp\left(-\frac{\lVert x_i^r-x_i\rVert^2}{\sigma_i^2}\right).
\]

主要 term 如下：

| term | 默认 weight | std | 误差 |
|---|---:|---:|---|
| `tracking_anchor_pos` | 0.5 | 0.3 | root/anchor world position |
| `tracking_anchor_ori` | 0.5 | 0.4 | root quaternion angular error |
| `tracking_relative_body_pos` | 1.0 | 0.3 | anchor-relative per-body position |
| `tracking_relative_body_ori_weighted` | 1.0 | 0.4 | anchor-relative per-body orientation |
| `tracking_body_linvel` | 1.0 | 1.0 | per-body linear velocity |
| `tracking_body_angvel` | 1.0 | 3.14 | per-body angular velocity |

发布的 `pnp_table` 又把 anchor orientation 和 relative body orientation 权重改为 `2.5`、`5.0`。左右 wrist 的 per-body orientation weight 可通过 `body_weights` 调节。论文称 wrist 权重被增强；当前所查 term 文件中 wrist 显式值是 `1.0`（与默认 body 权重相同），因此不能仅凭 term 名认定当前发布配置确实进行了数值增强，必须检查运行时 Hydra resolved config。

### 5.2 Object tracking

代码比论文正文更一般，包含 pose 和 velocity：

\[
r^{obj}=\mathbb{1}[C_{sim}]
\left(w_{op}e^{c_p\lVert e_p\rVert}+w_{or}e^{c_r e_R}
+w_{ov}e^{c_v\lVert e_v\rVert}+w_{oav}e^{c_{av}\lVert e_\omega\rVert}\right).
\]

默认 [`object_tracking_reward.yaml`](../../imports/SONIC/gear_sonic/config/manager_env/rewards/terms/object_tracking_reward.yaml) 为：`w_op=0.5`、其余三项权重为 0、position coefficient `-100`，并以仿真的 finger-object force 是否超过 1.0 作为 gate。这里与论文文字存在差异：论文公式把 contact label 用于 grasp gate，并直接描述 object pose reward；代码的 object reward 自身由模拟接触 gate。

更重要的是，当前 `pnp_table.yaml` 将 `object_tracking_reward.weight` 覆盖为 `0.0`，所以该发布配置下 object tracking term 实际关闭。若 DiT 数据来自这个配置的 rollout，不应假定动作教师受到 object-pose reward 的直接监督。

### 5.3 Grasp reward

每只手由三类信号组成：

1. **接触数量** `reward_grasp`：对超过 force threshold 的 finger/contact link 计数，计算 `clamp(N_contact / N_min, max=1)`；可由 motion library 的逐帧 `object_in_contact_{left/right}` gate。
2. **对向抓取姿态** `reward_grasp_finger_direction`：计算从物体中心（或标注 contact center）指向 thumb、index/middle 的方向，奖励相对的手指方向；仅在 reference contact phase 生效。
3. **接触中心接近** `reward_grasp_contact_center`：计算若干 fingertip 到参考 contact centroid 的平均距离，奖励 `exp(exp_coeff * mean_distance)`；默认 `exp_coeff=-10`，并由逐帧 contact label gate。

论文表 6 规定左手 grasp/contact-center 权重为右手的一半；term 文件只是分别提供左右手配置，是否为一半仍取决于最终实验 override。当前 `pnp_table` 的有效 grasp 参数为 right hand、weight `5.0`、`min_contacts=8`、`gate_with_contact_label=true`，finger-direction weight `10.0` 且 `use_contact_center=true`。

### 5.4 正则、安全和辅助损失

- `LatentL2Loss`：actor loss 中的辅助项，约束前 64 维 residual；论文值为 0.01，而当前 `pnp_table` 最后覆盖为 0.1。
- `meta_action_rate_l2`：`||m_t-m_{t-1}||²`，当前 `pnp_table` weight `-0.1`。
- `full_latent_rate_l2`：约束实际进入 decoder 的完整量化 latent 变化率，当前 weight `-0.01`。
- `undesired_contacts_no_ankle_hand`：惩罚脚踝/手之外的 body contact。
- `hand_table_contact_penalty`、`approach_velocity_penalty`、termination penalty：避免拍击桌面、过快接近和失败状态。
- `finger_primitive_limit`：惩罚越出 primitive 合法范围；当前 `pnp_table` 将其 weight 设为 0。

辅助 loss 属于优化目标但不是 Isaac Lab reward，导出训练数据时应单独记录，不要与 episode reward 混为一列。

## 6. Episode 初始化、终止与 PPO

论文/配置共同的重要设定：

- reference state initialization；
- 初始帧从前 30 帧采样，并限制在 reference contact 之前；
- 多条 task-family motion 共同训练，而不是一条动作一个 policy；
- PPO：`gamma=0.99`、GAE `lambda=0.95`、clip `0.2`、entropy `0.01`、5 epochs、4 mini-batches、24 steps/env；
- actor learning rate `2e-5`，critic `1e-3`，adaptive schedule 的 desired KL 为 `0.01`；
- `max_grad_norm=0.1` 由 `pnp_table` 覆盖；
- 论文报告 64 × L40、每 GPU 1024 env、30000 iterations；当前 release config 是 `num_envs=4096`、`num_learning_iterations=20000`，而 `docs/source/tracking.md` 的示例命令还会覆盖为 90000。实际复现实验必须保存 resolved config。

终止配置入口为 [`hoi_grab_no_lift.yaml`](../../imports/SONIC/gear_sonic/config/manager_env/terminations/tracking/hoi_grab_no_lift.yaml)，包括 anchor position/orientation、end-effector body position、motion timeout 和 object position deviation。论文给出 object z deviation 0.4 m、root height 0.25 m、root orientation 1.0 rad；当前 `pnp_table` 又把通用 `object_pos_deviation` threshold 覆盖为 0.1 m，并增加 hand-table contact termination。因此不能把论文阈值当成当前代码默认值。

## 7. 数据进入训练和导出的位置

### 7.1 输入 motion library

Object-Aware Tracking 至少依赖：

```text
<motion_lib>/
├── robot/       # 29-DoF retargeted q/qdot、root/body states
├── objects/     # object_root_pos/quaternion、contact points/labels
├── object_usd/  # 仿真碰撞/渲染资产
├── bps/         # <object_stem>.npy + optional basis metadata
└── meta/
```

[`motion_lib_base.py`](../../imports/SONIC/gear_sonic/utils/motion_lib/motion_lib_base.py) 加载并插值 `object_root_pos`、`object_root_quat`、左右手 contact center、逐帧 in-contact label 和 hand action，同时从 object trajectory 预计算 linear/angular velocity。contact center/label 可以由 `contact_points_left_hand/right_hand` 插值得到。

### 7.2 成功 rollout 导出

[`export_successful_rollouts.py`](../../grail/data_export/export_successful_rollouts.py) 将评估轨迹重新导出为标准 motion library，并保存：

- 仿真产生的 robot trajectory；
- source object trajectory 与 contact points；
- 从 14 个手指关节动作聚合得到的 `hand_action_left/right`；
- object USD、texture 和 metadata。

这条路径很适合作为 DiT demonstration 生成入口，但目前它主要导出“物理执行轨迹”，不保证保存 actor 原始 observation、64 维 residual、量化前/后 latent、reward decomposition 和 override 前后的 hand primitive。新增 DiT 前应先扩充 rollout recorder/export schema。

## 8. 为 DiT 模仿学习策略定义的建议接口

### 8.1 先选择 DiT 要模仿的层级

推荐第一阶段让 DiT 替换 Object-Aware actor，而保留冻结 SONIC decoder 和环境 wrapper：

\[
\text{DiT}(O_{t-H+1:t},\,\text{future reference},\,\text{object context})
\rightarrow \Delta z_{t:t+K-1}\in\mathbb{R}^{K\times64}.
\]

理由是 64 维 latent action 已由预训练 decoder 约束在可执行的全身动作流形内，且不需要 DiT 直接学习 43 个关节的底层稳定控制。手部可分两阶段：先继续使用数据 primitive override；随后增加独立的 2 维 primitive head，并关闭 override 做联合蒸馏/微调。

不建议第一版直接预测 43-DoF joint target；这会绕过 SONIC locomotion prior，也使训练数据需要重新定义 action scaling、joint order、PD frequency 和 history。

### 8.2 建议统一样本 schema

每个 transition/chunk 至少保存：

```yaml
schema_version: 1
motion_key: string
object_id: string
fps: float
frame_idx: int

condition:
  policy_obs_terms: {term_name: float32[...]}
  policy_obs_flat: float32[obs_dim]
  obs_term_order: [string]
  history_length: 10
  future_offsets: float32[10]
  object_bps: float32[10]

teacher_action:
  latent_residual_raw: float32[64]
  latent_residual_scaled: float32[64]
  encoded_latent_pre_fsq: float32[64]
  decoder_latent: float32[64]
  hand_primitive_policy_raw: float32[2]
  hand_primitive_executed: float32[2]
  body_joint_target: float32[29]
  env_joint_target: float32[43]
  hand_override_enabled: bool

supervision:
  contact_label_left: bool
  contact_label_right: bool
  contact_center_left: float32[3]
  contact_center_right: float32[3]
  reward_terms: {term_name: float32}
  terminated: bool
```

同时保存 `resolved_config.yaml`、SONIC checkpoint hash、joint/body name order、normalization statistics 和 git revision。不要只保存 flat observation：term 顺序或 history composition 改变时，只有具名字段才能迁移旧数据。

### 8.3 建议的代码接入点

保持如下边界可以最小化改动：

1. 在 actor module/config 层增加 `DiTPolicy`，保持输出字典和现有 `Actor` 一致，仍输出 `actions/meta_action`。
2. 继续由 `ManagerEnvWrapper.step()` 负责 action split、0.1 scaling、SONIC encode/FSQ/decode 和 joint merge。
3. 为 wrapper 增加只读 recorder hook，记录 `_last_full_latent_flat`、raw/executed hand primitive 和 resolved observation dict。
4. 离线训练代码只消费稳定 schema；在线 evaluation 仍复用当前 manager environment。
5. 将 `action_mode` 明确为枚举：`residual`（现有 teacher）、`direct_latent`（现有 student/distillation）、未来可增加 `dit_residual_chunk`。不要通过 action shape 隐式推断模式。

### 8.4 DiT 张量约定建议

| 张量 | 推荐 shape | 说明 |
|---|---|---|
| state tokens | `[B, H, D_state]` | 具名 observation 经分组 projector 后形成，避免直接依赖单一 flat dim |
| future-reference tokens | `[B, F, D_ref]` | F=10，保留实际时间 offset |
| object token | `[B, 1, D_obj]` | pose、delta、BPS、table context |
| noisy action chunk | `[B, K, 64]` | residual 的 diffusion target |
| hand head（可选） | `[B, K, 2]` | 用 BCE/离散 diffusion；与连续 residual 分头 |
| padding/valid mask | `[B, K]` | motion 尾部 chunk 必需 |

连续 latent residual 建议使用训练集统计量标准化，但 0.1 的环境 scaling 应保留在 wrapper 中；数据集中同时保存 raw 和 scaled 值以避免二次缩放。手部开合是离散变量，不应和 64 维连续 residual 用同一个高斯噪声目标而不做额外论证。

### 8.5 必须先补的测试

- **Observation contract test**：Hydra 合成后逐 term 检查顺序、shape、history 和总维度。
- **Residual identity test**：`Δz=0` 时 wrapper 输出应与冻结 SONIC 原策略一致。
- **Scaling test**：确认数据 loader、DiT 和 wrapper 中只有一处应用 `0.1`。
- **Hand override test**：分别覆盖 `use_motion_hand_actions=true/false`，断言 executed primitive 来源正确。
- **BPS test**：object stem 全覆盖、shape 恒为 10、非零率符合预期。
- **Chunk alignment test**：future reference、object delta、contact label、teacher action 使用同一 frame/time base。
- **Joint-order test**：29 body + 14 fingers 合并后的 name order 与环境 asset 完全一致。
- **Offline/online parity test**：同一 observation 和 teacher residual 经过离线解码与在线 wrapper 得到相同 43-DoF target。

## 9. 当前实现中需要显式管理的差异与风险

| 问题 | 论文 | 当前代码/配置 | 对 DiT 的影响 |
|---|---|---|---|
| hand 输出 | policy 预测 2 维 primitive | 当前仅 `pnp_table` 开启 motion-action override；其余三个 manipulation release config 关闭 | teacher label 必须区分 raw 与 executed，并记录来源 config |
| object reward | pose reward 是总奖励组成 | `pnp_table` 权重为 0；实现还由模拟接触 gate | rollout 未必学到显式 object-pose objective |
| latent L2 | 0.01 | `pnp_table` 覆盖为 0.1 | residual 分布尺度与论文可能不同 |
| wrist boost | 论文称增强 | 所查 weighted term 中 wrist=1.0 | resolved config 必须归档 |
| iteration/env 数 | 30000、1024/GPU | release/示例存在 20000/90000、4096 | 数据来源不可只写“GRAIL policy” |
| termination | 论文 0.4/0.25/1.0 | release object threshold 可为 0.1，另有桌面碰撞终止 | demonstration 分布和成功判据变化 |
| observation | 论文按语义列维度 | Hydra + history + term override 决定最终 flat layout | 禁止硬编码 flat slice |
| BPS | 10 维 shape condition | 路径或 stem 错误会静默变零 | 训练前必须做数据审计 |

## 10. 推荐实施顺序

1. 先增加 observation/action/reward recorder 和版本化数据 schema，不改现有 RL policy。
2. 用成功 rollout 导出 `(condition, residual chunk)`，建立 offline dataset audit。
3. 实现单步 `K=1` 的 MLP/Transformer behavior-cloning baseline，验证 offline/online parity。
4. 再实现 `K>1` 的 DiT residual chunk policy，wrapper 每步执行 chunk 的第一个 action并滚动重规划。
5. 第一阶段保持 `use_motion_hand_actions=true`，只评估 body latent；第二阶段训练 hand head并关闭 override。
6. 对比 zero-residual SONIC、PPO adaptor、BC baseline、DiT，并分别报告 motion、object、contact、termination 和 latent-norm 指标。

这一路径把新增工作限制在“策略模型 + 数据记录/加载”，同时复用当前经过验证的 SONIC decoder、动作缩放、手指映射和 Isaac Lab 评估环境，便于定位 DiT 本身带来的收益或退化。
