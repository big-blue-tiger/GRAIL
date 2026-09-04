已追踪配置继承链。结论是：该训练中的 critic 是一个「特权状态输入 → 3 层 SiLU MLP → 标量 V(s)」网络，不使用相机、token、proprio history encoder，也不是 2048 宽网络。

配置入口：

[robocasa_pickup_table_mlp_decoder_latent_vector_obs.yaml:8](/home/tide/robot/GRAIL/imports/SONIC/gear_sonic/config/exp/manager/universal_token/distill/robocasa_pickup_table_mlp_decoder_latent_vector_obs.yaml:8)

其继承链为：

```text
train_agent_trl.py
  └─ +exp=manager/universal_token/distill/
       robocasa_pickup_table_mlp_decoder_latent_vector_obs
       │
       ├─ pickup_table_diffusion_decoder_latent_vector_obs
       │   └─ apple_heading_w10_diffusion_decoder_latent_vector_obs
       │       └─ apple_heading_w10_diffusion_decoder_latent
       │           └─ robocasa_visual_privileged_adaptor_dagger
       │               └─ robocasa_ego_distill
       │
       └─ 当前叶子配置只覆盖 actor.backbone
          EncoderVectorMlpPolicy
```

关键点：

```text
robocasa_ego_distill
  └─ override /actor_critic: camera_cnn_distill
                                      │
                                      ├─ actor: 被后续 vector MLP 配置覆盖
                                      └─ critic: 保持 camera_cnn_distill 中的 MLP
```

对应配置见：

- [robocasa_ego_distill.yaml:9-11](/home/tide/robot/GRAIL/imports/SONIC/gear_sonic/config/exp/manager/universal_token/distill/robocasa_ego_distill.yaml:9)
- [camera_cnn_distill.yaml:29-42](/home/tide/robot/GRAIL/imports/SONIC/gear_sonic/config/actor_critic/camera_cnn_distill.yaml:29)

## Critic 输入 observation

critic 使用的是：

```text
obs["critic"]
    ↓ ManagerEnvWrapper.process_raw_obs()
obs_dict["critic_obs"]
    ↓
critic.evaluate({"critic_obs": critic_obs})
```

[manager_env_wrapper.py:589-593](/home/tide/robot/GRAIL/imports/SONIC/gear_sonic/envs/wrapper/manager_env_wrapper.py:589)

当前 critic observation 配置来自：

[grab_43dof_critic_mf_delta.yaml:7-35](/home/tide/robot/GRAIL/imports/SONIC/gear_sonic/config/manager_env/observations/critic/hoi_manip/grab_43dof_critic_mf_delta.yaml:7)

在当前配置下：

- 机器人：G1 43 DoF
- motion reference：29 DoF
- tracked body：14 个
- future reference：10 帧
- contact links：8 个
- 物理 action：43D
- policy meta-action：66D，但不直接作为 critic 的 `actions` observation

按 observation term 顺序，critic 输入为：

```text
critic_obs ∈ R^[B, 953]
```

其中 `B` 是并行环境/训练 batch 维度。

```text
┌──────┬────────────────────────────────────────┬──────┬───────────────┐
│ 起点 │ observation term                      │ 维度 │ 索引区间        │
├──────┼────────────────────────────────────────┼──────┼───────────────┤
│  0   │ command                                │  58  │ [0, 58)        │
│ 58   │ motion_anchor_pos_b                    │   3  │ [58, 61)       │
│ 61   │ motion_anchor_ori_b                    │   6  │ [61, 67)       │
│ 67   │ body_pos                               │  42  │ [67, 109)      │
│109   │ body_ori                               │  84  │ [109, 193)     │
│193   │ base_lin_vel                            │   3  │ [193, 196)     │
│196   │ base_ang_vel                            │   3  │ [196, 199)     │
│199   │ joint_pos                               │  43  │ [199, 242)     │
│242   │ joint_vel                               │  43  │ [242, 285)     │
│285   │ actions                                 │  43  │ [285, 328)     │
│328   │ target_object_pos                       │   3  │ [328, 331)     │
│331   │ hand_object_transform_6d                │   9  │ [331, 340)     │
│340   │ finger_tips_force                       │  24  │ [340, 364)     │
│364   │ grab_contact_flag                       │   1  │ [364, 365)     │
│365   │ object_pos_b                             │   3  │ [365, 368)     │
│368   │ object_ori_b_6d                         │   6  │ [368, 374)     │
│374   │ table_pos_b                             │   3  │ [374, 377)     │
│377   │ table_ori_b                             │   6  │ [377, 383)     │
│383   │ object_pos_delta_multi_future           │  30  │ [383, 413)     │
│413   │ object_ori_delta_multi_future_6d        │  60  │ [413, 473)     │
│473   │ command_multi_future                    │ 420  │ [473, 893)     │
│893   │ motion_anchor_ori_b_mf                  │  60  │ [893, 953)     │
├──────┼────────────────────────────────────────┼──────┼───────────────┤
│      │ 总计                                   │ 953  │ [0, 953)       │
└──────┴────────────────────────────────────────┴──────┴───────────────┘
```

其中：

```text
command
  = reference_joint_pos[29]
  + reference_joint_vel[29]
  = 58

body_pos
  = 14 tracked bodies × 3
  = 42

body_ori
  = 14 tracked bodies × 6D rotation
  = 84

finger_tips_force
  = 8 contact links × xyz force
  = 24

object_pos_delta_multi_future
  = 10 future frames × 3
  = 30

object_ori_delta_multi_future_6d
  = 10 future frames × 6
  = 60

command_multi_future
  = 10 future frames × 14 bodies × 3
  = 420

motion_anchor_ori_b_mf
  = 10 future frames × 6
  = 60
```

`actions=43` 是底层 Isaac Lab action manager 的物理关节动作。`66D` 是 actor 输出给 `ManagerEnvWrapper` 的 meta-action：

```text
66 = tokenizer latent 64 + hand primitive 2
```

它不是 critic 的 `actions` term。代码中 actor 的 `actions_dim` 后续才被设置为 66，用于 actor 输出维度；critic observation space 已经由底层环境实际建立。[train_agent_trl.py:429-454](/home/tide/robot/GRAIL/imports/SONIC/gear_sonic/train_agent_trl.py:429)

## Critic 网络结构

配置：

```yaml
critic:
  _target_: gear_sonic.trl.modules.actor_critic_modules.Critic
  running_mean_std: false
  backbone:
    _target_: gear_sonic.trl.modules.base_module.BaseModule
    process_output_dim: true
    module_config_dict:
      input_dim: [critic_obs]
      output_dim: [1]
      layer_config:
        type: MLP
        hidden_dims: [512, 256, 128]
        activation: SiLU
```

实际展开为：

```text
                         critic_obs
                       [B, 953]
                           │
                           │ dict unwrap:
                           │ input["critic_obs"]
                           ▼
                 Linear(953 → 512)
                           │
                         SiLU
                           │
                 Linear(512 → 256)
                           │
                         SiLU
                           │
                 Linear(256 → 128)
                           │
                         SiLU
                           │
                 Linear(128 → 1)
                           │
                           ▼
                    V(s_t) [B, 1]
```

高密度完整流程图：

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Isaac Lab ManagerBasedRLEnv                         │
│                                                                              │
│  robot state       motion reference       object/table       contact sensor  │
│  G1 43DoF          29DoF motion lib      scene objects       8 hand links   │
│      │                    │                    │                    │         │
│      └──────────────┬─────┴──────────────┬─────┴──────────────┬─────┘         │
│                     │                    │                    │               │
│                     ▼                    ▼                    ▼               │
│       ┌─────────────────────┐ ┌────────────────────┐ ┌────────────────────┐  │
│       │ proprio/state terms │ │ reference terms    │ │ manipulation terms │  │
│       │                     │ │                    │ │                    │  │
│       │ joint_pos     43    │ │ command       58   │ │ target_obj_pos  3 │  │
│       │ joint_vel     43    │ │ anchor_pos     3   │ │ hand_obj_6d     9 │  │
│       │ base_lin_vel    3   │ │ anchor_ori     6   │ │ finger_force   24 │  │
│       │ base_ang_vel    3   │ │ body_pos       42   │ │ contact_flag    1 │  │
│       │ actions        43   │ │ body_ori       84   │ │ object_pos_b    3 │  │
│       │                     │ │ cmd_mf        420   │ │ object_ori_6d    6 │  │
│       │                     │ │ anchor_ori_mf  60   │ │ table_pos_b      3 │  │
│       │                     │ │                  │ │ table_ori_6d     6 │  │
│       │                     │ │                  │ │ obj_delta_mf     30 │  │
│       │                     │ │                  │ │ obj_rot_delta    60 │  │
│       └──────────┬──────────┘ └──────────┬─────────┘ └──────────┬─────────┘  │
│                  └──────────────────────┴──────────────────────┴────────────┘  │
│                                           │                                   │
│                                           ▼                                   │
│                ObservationManager: concatenate_terms=True                   │
│                                           │                                   │
│                                           ▼                                   │
│                         obs["critic"] : [B, 953]                            │
│                                           │                                   │
│                                           ▼                                   │
│              ManagerEnvWrapper.process_raw_obs()                             │
│                                           │                                   │
│                         critic_obs = obs["critic"]                            │
│                                           │                                   │
│                                           ▼                                   │
│                  Critic.evaluate({"critic_obs": critic_obs})                 │
│                                           │                                   │
│                    running_mean_std = false                                  │
│                    batch_norm       = false                                  │
│                    image encoder    = none                                   │
│                    temporal buffer  = none                                   │
│                                           │                                   │
│                                           ▼                                   │
│                    BaseModule.forward(dict input)                           │
│              input = input["critic_obs"] : [B, 953]                          │
│                                           │                                   │
│                                           ▼                                   │
│                         Linear(953, 512)                                     │
│                                           │                                   │
│                                         SiLU                                 │
│                                           │                                   │
│                         Linear(512, 256)                                     │
│                                           │                                   │
│                                         SiLU                                 │
│                                           │                                   │
│                         Linear(256, 128)                                     │
│                                           │                                   │
│                                         SiLU                                 │
│                                           │                                   │
│                         Linear(128, 1)                                       │
│                                           │                                   │
│                                           ▼                                   │
│                              value estimate V(s_t)                          │
│                                  shape [B, 1]                                │
│                                           │                                   │
│                                           ▼                                   │
│             PPO GAE / clipped value loss / value optimizer update            │
└──────────────────────────────────────────────────────────────────────────────┘
```

实现上，`Critic.evaluate()` 只对 `critic_obs` 做可选归一化，然后调用 backbone；本配置 `running_mean_std=false`，因此直接进入 MLP。[actor_critic_modules.py:730-752](/home/tide/robot/GRAIL/imports/SONIC/gear_sonic/trl/modules/actor_critic_modules.py:730)

MLP 是普通全连接网络，没有 residual、LayerNorm、Dropout、GRU、Transformer 或 CNN。[base_module.py:279-304](/home/tide/robot/GRAIL/imports/SONIC/gear_sonic/trl/modules/base_module.py:279)

所以最终可以概括为：

```text
Critic:
    V(s_t) = Linear_128→1(
                SiLU(
                  Linear_256→128(
                    SiLU(
                      Linear_512→256(
                        SiLU(
                          Linear_953→512(critic_obs)
                        )
                      )
                    )
                  )
                )
             )
```

注意：`proprio_obs=[5,161] → 805D` 和 `privileged_obs=48D` 是当前 vector student 的 actor 条件输入，不属于这个 critic 的输入。