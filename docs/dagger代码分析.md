## 1：蒸馏学生策略模型架构

当前 Git 工作区中，原命令指定的 `robocasa_pickup_table_bc_decoder_latent_vector_obs` 已不存在；新增配置及训练文档均已改为：

```text
+exp=manager/universal_token/distill/robocasa_pickup_table_mlp_decoder_latent_vector_obs
```

以下按这一最新替代配置分析，并保留用户覆盖项 `ppo_loss_coef=1.0、num_envs=8、headless=False`。新增的 3 个网络单测已通过。

```text
                              EncoderVectorMlpPolicy
┌────────────────────────────────── INPUT ─────────────────────────────────────┐
│                                                                             │
│ proprio_obs [B,5,161]                         privileged_obs [B,48]           │
│                                                                             │
│ 每帧 proprio 161D：                          当前任务状态 48D：               │
│   joint_pos                  43                 object BPS               10   │
│   joint_vel                  43                 table corners            12   │
│   projected_gravity           3                 object position           3   │
│   base angular velocity       3                 object orientation 6D     6   │
│   base linear velocity        3                 hand-object transform     9   │
│   previous meta-action       66                 contact magnitudes        8   │
│                           ─────                                         ───── │
│                             161                                            48 │
│                                                                             │
│ enable_corruption=False                       enable_corruption=False        │
└─────────────────────┬───────────────────────────────────┬────────────────────┘
                      │ flatten [5,161]→[805]             │ [48]
                      ▼                                    ▼
         ┌──────────────────────────┐         ┌──────────────────────────┐
         │ Proprio running standard│         │ Privileged running std.  │
         │                          │         │                          │
         │ xp'=(xp-μp)/sqrt(vp)     │         │ xv'=(xv-μv)/sqrt(vv)     │
         │ clip to [-5,+5]          │         │ clip to [-5,+5]          │
         │ EMA momentum = 0.05      │         │ EMA momentum = 0.05      │
         │ std epsilon  = 10⁻⁴      │         │ std epsilon  = 10⁻⁴      │
         └────────────┬─────────────┘         └────────────┬─────────────┘
                      ▼                                    ▼
         ┌──────────────────────────┐         ┌──────────────────────────┐
         │ Linear 805 → 1024        │         │ Linear 48 → 512         │
         │ SiLU                     │         │ SiLU                     │
         │ Linear 1024 → 512        │         │ Linear 512 → 512        │
         └────────────┬─────────────┘         └────────────┬─────────────┘
                      │ fp∈R⁵¹²                            │ fv∈R⁵¹²
                      └──────────────────┬─────────────────┘
                                         ▼
                            c = concat(fp,fv) ∈ R¹⁰²⁴
                                         │
                                         ▼
                       ┌────────────────────────────────┐
                       │ 独立、确定性 MLP action_head   │
                       │                                │
                       │ Linear 1024 → 1024             │
                       │ SiLU                           │
                       │ Linear 1024 → 1024             │
                       │ SiLU                           │
                       │ Linear 1024 → 512              │
                       │ SiLU                           │
                       │ Linear 512 → 66                │
                       └───────────────┬────────────────┘
                                       │ ŷnorm∈R⁶⁶
                                       ▼
                 ŷ = clip(ŷnorm,-5,+5)⊙sqrt(vtarget)+μtarget
                                       │
                     ┌─────────────────┴─────────────────┐
                     ▼                                   ▼
          direct decoder latent [64]           hand primitives [2]
                     │                                   │
                     ▼                                   │
          reshape [64] → [2 tokens,32]                   │
                     │                                   │
                Frozen FSQ                               │
                     │                                   │
                     └─────────────────┬─────────────────┘
                                       ▼
┌────────────────────── FROZEN SONIC ACTION-TRANSFORM MODULE ─────────────────┐
│                                                                             │
│ quantized token [64] + policy_atm proprioception                            │
│                              │                                              │
│                              ▼                                              │
│ Frozen g1_dyn decoder：                                                     │
│ input → 2048 → 2048 → 1024 → 1024 → 512 → 512 → body joint action          │
│                              │                                              │
│ body joint action + two hand primitives → G1 43-DOF joint targets           │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
                                Isaac Sim execution
```

教师标签路径为：

```text
clean actor_obs
      │
      ▼
Frozen Teacher MLP
input → 512 → 256 → 128 → 66
      │
      ├── teacher latent residual δT = μT[:64]
      └── μT[64:66] 不作为本实验手部标签
                         │
clean policy_atm/tokenizer
      │
      ▼
Frozen ATM encoder → encoder latent EATM
      │
      ▼
zT = FSQ(EATM + 0.1·δT) ∈ R⁶⁴
      │
reference motion ───────────────► href∈R²
      │                              │
      └─────────── concat ───────────┘
                     │
            yT=[zT,href]∈R⁶⁶
                     │
       target running standardization
                     │
                     ▼
  LBC = mean[(ŷnorm-normalize(yT))²]
```

与旧回答相比，当前架构的关键变化是：

```text
旧：condition + zero action + timestep embedding
    → 1346→2048→2048→1024→66 diffusion-shaped denoiser

新：condition 1024
    → 1024→1024→1024→512→66 independent action_head
```

新模型已显式执行：

```python
del self.time_encoder
del self.denoiser
```

因此不存在可训练的 timestep encoder、denoiser、噪声输入、扩散时间采样或多步推理。配置中遗留的 diffusion 参数仅来自父类配置继承，不参与新 action head 的前向计算。学生网络约有 `4,295,746` 个权重参数，外层高斯策略另有 66 个可训练标准差参数。

## 2：奖励函数名称、权重与具体计算公式

环境控制周期为：

\[
\Delta t=\text{decimation}\times\text{sim\_dt}=4\times0.005=0.02\text{ s},
\]

Isaac Lab 单步聚合奖励为：

\[
R_t=\Delta t\sum_i w_i f_i(t).
\]

| 名称 | 权重 \(w_i\) | 原始奖励/惩罚函数 \(f_i\) |
|---|---:|---|
| `tracking_anchor_pos` | \(+0.5\) | \(\displaystyle f=\exp\left(-\frac{\|\hat p_a-p_a\|_2^2}{0.3^2}\right)\) |
| `tracking_anchor_ori` | \(+0.5\) | \(\displaystyle f=\exp\left(-\frac{e_q(\hat q_a,q_a)^2}{0.4^2}\right)\)，\(e_q\) 为四元数角误差 |
| `tracking_relative_body_pos` | \(+1.0\) | \(\displaystyle f=\exp\left[-\frac{\frac1N\sum_b\|\hat p_b^{rel}-p_b\|_2^2}{0.3^2}\right]\) |
| `tracking_relative_body_ori` | \(+1.0\) | \(\displaystyle f=\exp\left[-\frac{\frac1N\sum_b e_q(\hat q_b^{rel},q_b)^2}{0.4^2}\right]\) |
| `tracking_body_linvel` | \(+1.0\) | \(\displaystyle f=\exp\left[-\frac{\frac1N\sum_b\|\hat v_b-v_b\|_2^2}{1.0^2}\right]\) |
| `tracking_body_angvel` | \(+1.0\) | \(\displaystyle f=\exp\left[-\frac{\frac1N\sum_b\|\hat\omega_b-\omega_b\|_2^2}{3.14^2}\right]\) |
| `undesired_contacts_no_ankle_hand` | \(-0.1\) | \(\displaystyle f=\sum_{b\notin\{\text{ankles,hands}\}}\mathbf1[\max_h\|F_{b,h}\|_2>1.0]\) |
| `hand_table_contact_penalty` | \(-0.5\) | \(\displaystyle f=\min\left(\sum_{i,j}\|F^{table-hand}_{ij}\|_2,1\right)\) |
| `grasp_reward` | \(+1.0\) | 令 \(n_c=\sum_j\mathbf1[\|F_j^{object-hand}\|_2>0.1]\)，则 \(\displaystyle f=\min(n_c/3,1)\) |
| `grasp_finger_direction` | \(+0.1\) | \(\displaystyle u_T=\frac{p_{thumb}-p_o}{\|p_{thumb}-p_o\|+\epsilon}\)，\(u_S\) 为 index/middle 相对物体方向均值的单位向量；\(\displaystyle f=-u_T^\top u_S\,I_{\rm should-contact}\) |
| `finger_primitive_limit` | \(-10.0\) | 对两个手部 primitive \(h_j\)：\(\displaystyle f=\sum_j[\max(-h_j-0.5,0)+\max(h_j-0.5,0)]\) |
| `meta_action_rate_l2` | \(-0.01\) | `token_only=false`，因此 \(\displaystyle f=\|a_t^{meta}-a_{t-1}^{meta}\|_2^2\)，覆盖全部 66 维 |
| `full_latent_rate_l2` | \(-0.01\) | \(\displaystyle f=\|z_t^{full}-z_{t-1}^{full}\|_2^2\)。但当前 DAgger 使用 `mixed` action mode，而 wrapper 仅在纯 `residual` 模式更新 full-latent buffer，因此本次运行该项实际保持为 0 |
| `is_terminated` | \(-10.0\) | \(\displaystyle f=\mathbf1[\text{本步为非 timeout 终止}]\) |
| `object_tracking_reward` | \(+1.0\) | 一般式见下；当前只有位置分量非零，因此 \(\displaystyle f=0.5I_C\exp[-100\|\hat p_o-p_o\|_2]\) |

`object_tracking_reward` 的完整代码公式为：

\[
\begin{aligned}
f_{\rm object}=I_C\big(&
0.5e^{-100\|\hat p_o-p_o\|_2}
+0.0e^{-100e_q(\hat q_o,q_o)}\\
&+0.0e^{-5\|\hat v_o-v_o\|_2}
+0.0e^{-5\|\hat\omega_o-\omega_o\|_2}\big),
\end{aligned}
\]

其中：

\[
I_C=
\mathbf1\left[
\sum_{i,j}\|F^{object-hand}_{ij}\|_2>1.0
\right].
\]

所以最终单步奖励为：

\[
\begin{aligned}
R_t=0.02\big[&
0.5r_{a,p}+0.5r_{a,q}
+r_{b,p}+r_{b,q}+r_{b,v}+r_{b,\omega}\\
&-0.1N_{\rm bad-contact}
-0.5P_{\rm hand-table}
+r_{\rm grasp}
+0.1r_{\rm finger-dir}\\
&-10P_{\rm primitive}
-0.01\|\Delta a^{meta}\|_2^2
-0.01\|\Delta z^{full}\|_2^2\\
&-10I_{\rm terminated}
+0.5I_Ce^{-100\|\hat p_o-p_o\|_2}
\big].
\end{aligned}
\]

## 3：高精度训练数据流与流程管道

```text
┌───────────────────────────── TRAINING ITERATION k ──────────────────────────┐
│ num_envs=8                                                                 │
│ rollout length=16 steps/env                                                │
│ collection size=8×16=128 transitions                                       │
│ PPO epochs=3，mini-batches=4                                               │
│ 每个 mini-batch=2 条完整 env trajectory×16=32 transitions                 │
│ 每轮 optimizer updates=3×4=12，无额外 micro-batching                      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌──────────────────────────── ONLINE OBSERVATION ─────────────────────────────┐
│                                                                             │
│ Student input：                                                             │
│   proprio_obs       [8,5,161]                                               │
│   privileged_obs    [8,48]                                                  │
│   当前两组 enable_corruption=False                                          │
│                                                                             │
│ Critic input：                                                              │
│   critic_obs        [8,Dcritic]                                             │
│                                                                             │
│ Teacher/label input：                                                       │
│   actor_obs, policy_atm, tokenizer                                          │
│   trainer 暂时清除 observation term noise 后重新计算 clean copy            │
│   reference_hand_actions [8,2] 在当前 motion frame 立即读取                 │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌──────────────────────────── FIXED DAgger SPLIT ─────────────────────────────┐
│ dagger_student_ratio=0.5                                                   │
│                                                                             │
│ randperm(8) → 精确分配 4 student env + 4 teacher env                       │
│ mask 在本轮全部 16 steps 内固定，下一训练轮重新采样                         │
│ Student 与 Frozen Teacher 每一步都对完整 8-env batch 前向                  │
└─────────────────────────┬───────────────────────────┬───────────────────────┘
                          ▼                           ▼
              ┌─────────────────────┐     ┌────────────────────────┐
              │ Student             │     │ Frozen Teacher         │
              │                     │     │                        │
              │ proprio/privileged  │     │ clean actor_obs        │
              │ → dual encoders     │     │ → MLP 512/256/128      │
              │ → MLP action_head   │     │ → μT∈R⁶⁶              │
              │ → μS∈R⁶⁶           │     │ → aT~πT               │
              │ → aS~N(μS,σS²)      │     │                        │
              └──────────┬──────────┘     └───────────┬────────────┘
                         └──────────────┬──────────────┘
                                        ▼
┌──────────────────────────── MIXED ENV EXECUTION ────────────────────────────┐
│                                                                             │
│ aexec[e] = aT[e]，e∈teacher mask                                            │
│          = aS[e]，e∈student mask                                            │
│                                                                             │
│ Teacher env：                                                               │
│   δT=aT[:64]                                                                │
│   zT=FSQ(EATM(policy_atm,tokenizer)+0.1δT)                                  │
│   hand primitive=reference motion                                           │
│                                                                             │
│ Student env：                                                               │
│   direct latent=aS[:64]                                                     │
│   zS=FSQ(reshape(aS[:64]))，完全跳过 ATM encoder                           │
│   hand primitive=aS[64:66]                                                  │
│                                                                             │
│ z + policy_atm proprioception → Frozen g1_dyn decoder                       │
│ → body action + hand primitive → 43-DOF target → env.step                  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌──────────────────────────── ROLLOUT STORAGE ────────────────────────────────┐
│ 每步保存：                                                                  │
│                                                                             │
│ live student observations：proprio、privileged、critic 等                   │
│ clean label observations：actor_obs、policy_atm、tokenizer                  │
│ current-frame reference_hand_actions                                        │
│ student sampled action aS [8,66]                                            │
│ old student μold、σold、logπold(aS|s)                                      │
│ critic value Vold                                                           │
│ reward、done、timeout                                                       │
│                                                                             │
│ 关键语义：即使 teacher-mask 环境实际执行 aT，storage 仍保存 aS 及其         │
│ log-prob。因此命令将 ppo_loss_coef 改为 1 后，teacher-mask 的 64 个         │
│ transitions/iteration 是 DAgger 混合动力学数据，不是严格 on-policy PPO：   │
│ next-state/reward 来自 aT，概率比使用的 action 却是 aS。                    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                              repeat 16 steps
                                       ▼
┌───────────────────────────── GAE / RETURNS ─────────────────────────────────┐
│ timeout bootstrap：                                                         │
│   r'ₜ = rₜ + γ·timeoutₜ·V(sₜ)                                               │
│                                                                             │
│ temporal-difference residual：                                              │
│   δₜ = r'ₜ + γ(1-dₜ)V(sₜ₊₁)-V(sₜ)                                         │
│                                                                             │
│ generalized advantage：                                                     │
│   Aₜ = δₜ + γλ(1-dₜ)Aₜ₊₁                                                  │
│   γ=0.99，λ=0.95                                                           │
│                                                                             │
│ return：                                                                    │
│   Gₜ=Aₜ+V(sₜ)                                                              │
│                                                                             │
│ A 在全部 env×time 样本上执行均值/标准差归一化                              │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌────────────────────────────── LABEL BUILDING ───────────────────────────────┐
│ 每个 mini-batch、每个 PPO epoch 都重新执行：                               │
│                                                                             │
│ clean actor_obs → Frozen Teacher → μT → δT=μT[:64]                         │
│                                        │                                    │
│ clean policy_atm/tokenizer → Frozen ATM encoder → EATM                      │
│                                        │                                    │
│                  zT=FSQ(EATM+0.1δT) ──┘                                    │
│                                        │                                    │
│ stored reference hands href [2] ───────┤                                    │
│                                        ▼                                    │
│                         yT=concat(zT,href)∈R⁶⁶                              │
│                                                                             │
│ 标签统计：                                                                  │
│   首个 batch：μy←batch_mean，vy←batch_var                                  │
│   后续：μy←0.95μy+0.05 batch_mean                                          │
│         vy←0.95vy+0.05 batch_var                                            │
│   ỹT=clip((yT-μy)/sqrt(max(vy,10⁻⁸)),-5,+5)                                │
│                                                                             │
│ 同一批 rollout 会经过 3×4=12 次训练 forward，因此 observation/target       │
│ running statistics也会在一轮内更新 12 次，而不是仅在数据收集时更新一次。    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌──────────────────────────── STUDENT + PPO FORWARD ──────────────────────────┐
│ Student：                                                                   │
│   normalized proprio/privileged                                             │
│   → EncoderVectorMlpPolicy                                                  │
│   → ŷnorm=action_head(c)                                                     │
│   → μθ=denormalize(ŷnorm)                                                   │
│                                                                             │
│ BC loss：                                                                   │
│   LBC = mean[(ŷnorm-ỹT)²]                                                    │
│   bc_loss_coef=1.0，aux_loss_scale=1.0                                      │
│                                                                             │
│ PPO probability ratio：                                                     │
│   ρθ=exp(logπθ(aS|s)-logπold(aS|s))                                         │
│                                                                             │
│ clipped policy loss：                                                       │
│   Lclip=mean[max(-Aρθ,-A·clip(ρθ,0.8,1.2))]                                │
│                                                                             │
│ clipped value loss：                                                        │
│   LV=mean[max((Vθ-G)²,                                                       │
│               (clip(Vθ,Vold-0.2,Vold+0.2)-G)²)]                            │
│                                                                             │
│ entropy term：                                                              │
│   Lentropy=-mean[H(πθ)]                                                      │
│                                                                             │
│ LPPO=Lclip+1.0LV+0.01Lentropy                                               │
│                                                                             │
│ 本命令最终优化目标：                                                        │
│   Ltotal = ppo_loss_coef·LPPO + aux_loss_scale·bc_loss_coef·LBC             │
│          = 1.0·LPPO + 1.0·LBC                                               │
│                                                                             │
│ distill_teacher_loss_coef=0：不再额外计算 student_mean/teacher_mean MSE     │
│ `bc_target/*`、`bc_pred/*` 仅为统计指标，未配置系数，因此不进入 Ltotal       │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌──────────────────────────── OPTIMIZATION LOOP ──────────────────────────────┐
│ backward                                                                    │
│ → global gradient clipping，max_norm=1.0                                   │
│ → optimizer.step                                                            │
│ → optimizer.zero_grad                                                       │
│                                                                             │
│ 初始 LR=2×10⁻⁴                                                             │
│ desired_KL=0.01 的自适应调节范围=[10⁻⁵,2×10⁻⁴]                            │
│ Student 与 Critic 更新；Frozen Teacher、Frozen ATM 始终 no-grad/eval       │
│                                                                             │
│ 3 epochs×4 mini-batches 完成后：                                            │
│ → iteration k+1                                                            │
│ → 重新抽取 4/4 DAgger environment mask                                     │
│ → 再收集下一批 128 transitions                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```