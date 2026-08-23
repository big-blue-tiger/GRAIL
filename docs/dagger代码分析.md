这个配置实际使用的是 `TRLAuxLossPPOTrainer`，它继承 `TRLPPOTrainer`。每个 iteration 的核心结构是：

```text
student rollout 采集 16 × 16 条 transition
        ↓
critic 计算 value、GAE return、advantage
        ↓
按环境维度打乱并分成 4 个 minibatch
        ↓
每个 minibatch 中重新运行 frozen teacher
        ↓
teacher residual → frozen ATM → 66 维 decoder target
        ↓
student flow-matching forward
        ↓
diffusion flow loss
        ↓
backward → 梯度裁剪 → optimizer.step
```

当前配置继承后的重要参数为：

| 参数 | 值 | 含义 |
|---|---:|---|
| `num_envs` | 16 | 单进程并行环境数量 |
| `num_steps_per_env` | 16 | 每次 iteration rollout 步数 |
| `num_learning_epochs` | 3 | 同一批 rollout 数据重复训练 3 次 |
| `num_mini_batches` | 4 | 每个 epoch 分成 4 个 minibatch |
| `per_device_train_batch_size` | `null` | 自动设置为 local minibatch size |
| `num_micro_batches` | 1 | 每个 minibatch 不再细分 |
| `actor_learning_rate` | `2e-4` | optimizer 的基础学习率 |
| `max_grad_norm` | `1.0` | 梯度范数裁剪阈值 |
| `ppo_loss_coef` | `0.0` | PPO loss 不参与最终梯度 |
| `distill_teacher_loss_coef` | `0.0` | 普通 teacher/student action MSE 关闭 |
| `diffusion_decoder_distill` | `true` | 使用 teacher 构造 diffusion target |
| `diffusion_loss_coef` | `1.0` | flow-matching loss 权重 |
| `aux_loss_scale` | `1.0` | auxiliary loss 全局权重 |
| `num_inference_steps` | 4 | rollout 时 flow ODE 的 Euler 积分步数 |

因此单进程每个 iteration：

- 采集 `16 env × 16 steps = 256 transitions`
- 但 minibatch 是按环境维度划分，不是把 256 条 transition 完全打散
- 每个 minibatch 包含 `4 env × 16 steps = 64 transitions`
- 3 个 PPO epoch、每个 epoch 4 个 minibatch
- 所以每个 iteration 通常执行 `3 × 4 = 12` 次 `optimizer.step()`

---

## 一、每个 iteration 的函数执行顺序

入口是 [ppo_trainer.py](/home/tide/robot/GRAIL/imports/SONIC/gear_sonic/trl/trainer/ppo_trainer.py:1924) 中的：

```python
TRLPPOTrainer.train()
```

不过实例的实际类型是：

```python
TRLAuxLossPPOTrainer
```

因为配置继承链最终包含：

```yaml
override /trainer: trl_ppo_aux
```

对应：

```yaml
_target_: gear_sonic.trl.trainer.ppo_trainer_aux_loss.TRLAuxLossPPOTrainer
```

### 1. 更新调度参数

在 `train()` 每轮开始：

```python
scheduler.update_scheduled_params(...)
```

功能：

- 根据 `global_step` 更新可能随训练变化的参数
- 例如 reward 权重、环境参数或训练超参数

输出：

```python
self.scheduled_params_dict
```
for i in range(num_steps_per_env): 

目标：

- 用当前 student policy 和环境交互
- 连续采集 `num_steps_per_env=16` 步
- 保存 observation、action、log-prob、reward、done 等数据
- rollout 结束后计算 value、return 和 advantage

开始时依次执行：

```python
self._train_rollout_mode()
policy_model.init_rollout()
self.storage.clear()
```

#### `_train_rollout_mode()`

作用：

- `self.model.eval()`
- policy mode 设置为 `"train_rollout"`
- 关闭训练时 transform
- 环境保持训练状态

注意：这里是 eval mode，因此 BatchNorm/Dropout 使用推理行为，但 diffusion sampling 仍然会显式生成随机高斯噪声。

#### `policy_model.init_rollout()`

作用：

- 初始化 policy 的 rollout 状态
- 如果模型具有历史缓存、transformer memory 或 recurrent state，在这里初始化

#### `storage.clear()`

作用：

- 清空上一个 iteration 的 rollout storage
- 将 storage 写入位置重置为 0

---

## 二、rollout 中每个 step 的推理过程

`_rollout_step()` 内部执行：

```python
for i in range(self.num_steps_per_env):
```

也就是每个 iteration 执行 16 次环境 step。

### 2.1 `policy_step(policy_model, obs_dict, cur_dones=dones)`

位置：[ppo_trainer.py](/home/tide/robot/GRAIL/imports/SONIC/gear_sonic/trl/trainer/ppo_trainer.py:835)

输入：

- 当前 16 个环境的 observation
- 上一步的 done 标记
- student policy

核心调用：

```python
policy_state_dict = policy_model.rollout(
    obs_dict=actor_obs_dict,
    episode_attnmask=None,
    cur_dones=cur_dones,
)
```

这里运行的是 student，而不是 teacher。

当前 student backbone 为：

```python
EncoderVectorDiffusionPolicy
```

它读取：

```yaml
observation_key: student_obs
observation_input_dim: 138
```

因此 rollout 中 student 的条件输入是 138 维结构化 observation。

### 2.2 student diffusion 推理

`EncoderVectorDiffusionPolicy` 继承了 `EncoderRgbDiffusionPolicy` 的采样逻辑。

推理时走：

```python
return self._sample(cond)
```

由于配置为：

```yaml
diffusion_objective: flow_matching
```

所以进入：

```python
_sample_flow(cond)
```

首先生成初始噪声：

```python
sample = torch.randn(
    *cond.shape[:-1],
    action_dim,
)
```

然后进行 4 步 Euler 积分：

```python
for step in range(num_inference_steps):
    t = step / num_inference_steps
    pred_velocity = self._predict_noise(sample, t, cond)
    sample = sample + dt * pred_velocity
```

对应：

\[
x_{k+1}=x_k+\Delta t\,v_\theta(x_k,t_k,c)
\]

最终输出：

```python
sample = self._denormalize_target(sample)
```

其维度是 66：

- 前 64 维：直接送入 frozen decoder 的 latent
- 后 2 维：hand action

### 2.3 `distill_rollout_with_action_mean`

配置中：

```yaml
distill_rollout_with_action_mean: true
```

于是 `policy_step()` 执行：

```python
policy_state_dict["actions"] = (
    policy_state_dict["action_mean"].detach()
)
```

这意味着不再额外从 Actor 的 Gaussian distribution 中采样，而是使用 `action_mean` 执行环境动作。

但需要注意：这里并非完全确定性推理。因为 flow sampling 本身从：

```python
torch.randn(...)
```

开始，所以 diffusion 输出仍然包含随机性；只是不会再叠加 Actor Gaussian 的第二层采样噪声。

### 2.4 计算 rollout action log-prob

```python
actions_log_prob = policy_model.get_actions_log_prob(actions).unsqueeze(1)
```

结果被保存，用于后续 PPO ratio 和统计计算。

虽然当前：

```yaml
ppo_loss_coef: 0.0
```

PPO policy loss 最终不产生有效梯度，但原始 PPO 数据和相关指标仍然会被计算。

---

### 2.5 `_compute_clean_obs_dict(obs_dict)`

位置：[ppo_trainer.py](/home/tide/robot/GRAIL/imports/SONIC/gear_sonic/trl/trainer/ppo_trainer.py:957)

目标：

- 重新构造不包含 observation noise 的 teacher observation
- student 可以使用带噪声/受限 observation
- teacher 使用干净、privileged observation 生成监督目标

结果是：

```python
clean_obs_dict
```

随后由：

```python
_store_clean_teacher_obs(clean_obs_dict)
```

保存为 storage 中的：

```text
clean_actor_obs
clean_policy_atm
clean_tokenizer
...
```

因此 teacher observation 和 student observation 在 rollout storage 中是分开保存的。

注意：rollout 阶段并没有运行 teacher 网络，只是在准备以后 teacher forward 所需的 clean observation。

---

### 2.6 保存 transition

依次保存：

```python
self.storage.update_key(obs_key, obs_value)
self._store_clean_teacher_obs(clean_obs_dict)
self.storage.update_key(policy_output_key, value)
```

保存的数据主要包括：

- student observation
- clean teacher observation
- student action
- student action mean/std
- action log-prob

---

### 2.7 `env.step(policy_state_dict)`

执行 student 输出的 66 维 meta action。

由于配置：

```yaml
use_student_direct_latent: true
```

环境 action transform 将：

- student 前 64 维作为完整 decoder latent
- 后 2 维作为 hand action
- 使用 frozen decoder 转换为机器人实际控制 action

环境返回：

```python
obs_dict, rewards, dones, infos
```

然后 storage 保存：

```python
rewards
dones
time_outs
```

---

### 2.8 `_process_env_step(rewards, dones, infos)`

位置：[ppo_trainer.py](/home/tide/robot/GRAIL/imports/SONIC/gear_sonic/trl/trainer/ppo_trainer.py:1198)

功能：

```python
self.policy_model.reset(dones)
self.value_model.reset(dones)
self.episode_env_tensors.add(infos["to_log"])
```

目标：

- 对已经结束的环境清理 policy/critic 历史状态
- 累计环境指标
- 为下一步 rollout 做准备

---

## 三、rollout 结束后的 value、return 和 advantage

完成 16 步环境交互后：

```python
policy_model.clear_rollout()
```

清除 rollout cache。

### 3.1 `_chunked_value_evaluate()`

位置：[ppo_trainer.py](/home/tide/robot/GRAIL/imports/SONIC/gear_sonic/trl/trainer/ppo_trainer.py:886)

输入序列长度为：

```text
16 rollout observations + 1 final observation
```

即近似形状：

```text
[num_envs=16, time=17, obs_dim]
```

critic 一次性计算：

```python
all_values
```

拆分为：

```python
values = all_values[:-1]
last_values = all_values[-1]
```

其中：

- `values`：16 个 rollout step 的 value
- `last_values`：最后 observation 的 bootstrap value

### 3.2 timeout bootstrap

```python
new_rewards = (
    rewards
    + gamma * time_outs * values
)
```

目标：

- 对由于 time limit 而结束、但不是真正 terminal 的 episode 做 value bootstrap
- 避免将 timeout 错误地当成价值为零的终止状态

### 3.3 `_compute_returns()`

位置：[ppo_trainer.py](/home/tide/robot/GRAIL/imports/SONIC/gear_sonic/trl/trainer/ppo_trainer.py:2384)

从最后一步向前计算 GAE：

\[
\delta_t =
r_t+\gamma(1-d_t)V_{t+1}-V_t
\]

\[
A_t =
\delta_t+\gamma\lambda(1-d_t)A_{t+1}
\]

\[
R_t=A_t+V_t
\]

配置：

```text
gamma = 0.99
lambda = 0.95
```

随后 advantage 做标准化：

```python
advantages = (
    advantages - advantages.mean()
) / (
    advantages.std() + 1e-8
)
```

多 GPU 时先 gather 全部 rank 的 advantage，再进行全局标准化。

最后写入 storage：

```python
values
returns
advantages
```

不过当前配置 `ppo_loss_coef=0`，这些值不会直接驱动参数更新，主要保留 PPO trainer 的统一流程和统计。

---

## 四、`_get_rollout_data()`：整理完整 rollout

位置：[ppo_trainer.py](/home/tide/robot/GRAIL/imports/SONIC/gear_sonic/trl/trainer/ppo_trainer.py:1266)

storage 内部主要采用：

```text
[time, env, ...] = [16, 16, ...]
```

该函数将其转置为：

```text
[env, time, ...] = [16, 16, ...]
```

输出包括：

```python
{
    "all_obs_dict": ...,
    "clean_obs_dict": ...,
    "actions": ...,
    "logprobs": ...,
    "values": ...,
    "returns": ...,
    "advantages": ...,
    "old_mu_batch": ...,
    "old_sigma_batch": ...,
    "dones": ...,
}
```

其中：

- `all_obs_dict`：student 使用的 rollout observation
- `clean_obs_dict`：teacher 使用的无噪声/privileged observation

这一步之后调用：

```python
self._train_mode()
```

将模型切换回训练状态：

```python
self.model.train()
model.set_mode("train")
model.transform_train()
```

---

# 五、minibatch 与 microbatch 划分

## 5.1 batch size 的定义

该 trainer 中：

```python
args.local_batch_size = env.num_envs
```

所以当前：

```text
local_batch_size = 16
```

这里的 batch 单位是环境轨迹，不是单条 transition。

每个 batch element 都包含完整 16 步序列：

```text
一个 batch element = 一个环境的 16-step trajectory
```

因此完整 rollout tensor 约为：

```text
[16 env, 16 time, ...]
```

## 5.2 minibatch 划分

配置：

```text
num_mini_batches = 4
```

所以：

```python
local_mini_batch_size = 16 // 4 = 4
```

每个 minibatch 形状约为：

```text
[4 env, 16 time, ...]
```

即每个 minibatch 包含 64 条 transition，但时间维没有打散。

这样设计可以：

- 保留完整 episode/trajectory 序列
- 正确构造 episode attention mask
- 支持 transformer 或依赖历史信息的 policy

## 5.3 每个 epoch 重新打乱

配置：

```yaml
ppo_shuffle_every_epoch: true
```

每个 epoch 都执行：

```python
b_inds = torch.randperm(args.local_batch_size)
```

即重新打乱 16 个环境轨迹，然后每 4 条轨迹组成一个 minibatch。

时间步之间不进行随机 shuffle。

## 5.4 microbatch

由于：

```yaml
per_device_train_batch_size: null
```

初始化时将其设置为：

```python
per_device_train_batch_size = local_mini_batch_size = 4
```

因此：

```text
num_micro_batches = 4 // 4 = 1
```

当前每个 minibatch 只有一个 microbatch：

```text
microbatch shape = [4 env, 16 steps, ...]
```

如果以后把 `per_device_train_batch_size` 设置为 2，那么每个 minibatch 会拆成两个：

```text
[2 env, 16 steps]
[2 env, 16 steps]
```

---

## 六、`_get_mb_rollout_data()`

位置：[ppo_trainer.py](/home/tide/robot/GRAIL/imports/SONIC/gear_sonic/trl/trainer/ppo_trainer.py:1360)

对当前 microbatch 的环境索引切片：

```python
mb_obs_dict = all_obs_dict[micro_batch_inds]
mb_teacher_obs_dict = clean_obs_dict[micro_batch_inds]
```

得到两套输入：

```text
student:
    mb_obs_dict
    例如 student_obs，可能为受限/带噪声 observation

teacher:
    mb_teacher_obs_dict
    clean、privileged observation
```

同时切出：

- action
- old log-prob
- old mean/std
- value
- return
- advantage
- done
- padding mask

然后根据 `mb_dones` 构造：

```python
episode_attnmask = rl.compute_episode_attnmask(mb_dones)
```

防止模型跨 episode termination 使用历史信息。

---

# 七、teacher/student 的划分

teacher/student 的核心分离发生在 `_forward_model()`。

位置：[ppo_trainer.py](/home/tide/robot/GRAIL/imports/SONIC/gear_sonic/trl/trainer/ppo_trainer.py:1414)

## 7.1 teacher forward

配置中：

```python
self.compute_distill_teacher_loss = False
self.diffusion_decoder_distill = True
```

因此：

```python
needs_teacher = (
    self.compute_distill_teacher_loss
    or self.diffusion_decoder_distill
)
```

结果仍然是 `True`。

teacher forward：

```python
self.ref_model.eval()

with torch.no_grad():
    teacher_results = self.ref_model.act(
        obs_dict=mb_teacher_obs_dict,
        episode_attnmask=episode_attnmask,
    )
```

特点：

- teacher 使用 clean/privileged observation
- teacher 为 eval mode
- teacher 位于 `torch.no_grad()`
- teacher 参数不接收梯度
- teacher 每个 minibatch 都重新运行一次
- 同一 rollout 数据会训练 3 个 epoch，所以 teacher 也会对同一轨迹运行 3 次

teacher 输出：

```python
teacher_results["action_mean"]
```

是原始 teacher meta action，约为：

```text
[batch_env, time, 66]
```

其中：

- 前 64 维是 teacher latent residual
- 后 2 维是 hand action

## 7.2 `_build_diffusion_decoder_target()`

位置：[ppo_trainer.py](/home/tide/robot/GRAIL/imports/SONIC/gear_sonic/trl/trainer/ppo_trainer.py:1487)

先拆分 teacher action：

```python
latent_residual = teacher_action_mean[..., :64].detach()
hand_action = teacher_action_mean[..., 64:66].detach()
```

因为配置：

```yaml
diffusion_target_latent_mode: decoder_input
```

不会直接监督 teacher residual，而是将 residual 送入 frozen action-transform module：

```python
action_transform_module(
    flat_atm_obs_dict,
    latent_residual=scaled_residual,
    latent_residual_mode=latent_mode,
)
```

然后取得：

```python
full_latent = atm_module._last_full_latent_flat
```

最终 target：

```python
diffusion_target = torch.cat(
    [full_latent.detach(), hand_action],
    dim=-1,
)
```

也就是：

```text
64 维完整 decoder latent + 2 维 hand action
```

这样 student 推理时可以直接预测 decoder 输入，而不需要复现 teacher 的 residual-addition 过程。

整个 target 构造路径都无梯度：

```text
teacher            frozen
ATM encoder/path   frozen
full_latent        detach
hand_action        detach
```

## 7.3 student forward

构造：

```python
policy_kwargs = {
    "obs_dict": mb_obs_dict,
    "actions": mb_actions,
    "episode_attnmask": episode_attnmask,
    "diffusion_target": diffusion_target,
}
```

然后：

```python
model.forward(
    modes=["policy", "value"],
    ...
)
```

`Actor` 因为配置：

```yaml
has_aux_loss: true
```

且当前处于 train mode，会向 diffusion backbone 传入：

```python
compute_aux_loss=True
```

因此 student 此时不会执行 4 步 flow sampling，而是执行单步随机时间的 flow-matching 训练。

---

# 八、flow-matching loss 计算

位置：[diffusion_policy_modules.py](/home/tide/robot/GRAIL/imports/SONIC/gear_sonic/trl/modules/diffusion_policy_modules.py:412)

## 8.1 observation condition

student 先将 138 维 `student_obs` 编码为 condition：

```python
cond = self._encode_condition(input)
```

大致得到：

```text
[4 env, 16 time, cond_dim=512]
```

## 8.2 target 标准化

```python
normalized_target = self._normalize_target(target)
```

target running mean/variance 在：

```python
with torch.no_grad():
    self._update_target_stats(target)
```

中更新，不进入梯度图。

记标准化后的 teacher target 为：

\[
x_1
\]

## 8.3 生成 noise 和 flow time

```python
noise = torch.randn_like(normalized_target)
t = self._sample_flow_time(...)
```

记：

\[
x_0=\epsilon,\qquad \epsilon\sim\mathcal N(0,I)
\]

时间 \(t\) 不是简单均匀采样，而是根据配置的 Beta distribution 生成：

```yaml
noise_beta_alpha: 1.5
noise_beta_beta: 1.0
noise_s: 0.999
```

## 8.4 构造插值状态

```python
noisy_action = (
    (1.0 - t) * noise
    + t * normalized_target
)
```

即：

\[
x_t=(1-t)x_0+t x_1
\]

目标速度：

```python
velocity = normalized_target - noise
```

即：

\[
v^\*=x_1-x_0
\]

## 8.5 student 预测速度场

```python
pred_velocity = self._predict_noise(
    noisy_action,
    t_discretized,
    cond,
)
```

内部依次经过：

```text
student_obs
    ↓
condition encoder
    ↓
cond embedding

t
    ↓
sinusoidal time embedding
    ↓
time encoder

[x_t, cond, time_feature]
    ↓
denoiser MLP
    ↓
pred_velocity
```

## 8.6 flow loss

```python
diffusion_loss = F.mse_loss(
    pred_velocity,
    velocity,
)
```

即：

\[
L_{\mathrm{flow}}
=
\mathbb E_{x_0,x_1,t}
\left[
\left\|
v_\theta(x_t,t,c)-(x_1-x_0)
\right\|_2^2
\right]
\]

它作为：

```python
aux_losses["diffusion_flow"]
```

返回。

注意这里返回的：

```python
"action_mean": target.detach()
```

只是为了兼容 Actor/PPO 接口。它不代表 student 当前通过 4 步积分得到的预测，也不承担 flow-matching 梯度。

真正带梯度的是：

```python
aux_losses["diffusion_flow"]
```

---

# 九、loss 汇总顺序

由于实际 trainer 是 `TRLAuxLossPPOTrainer`，调用顺序为：

```text
TRLAuxLossPPOTrainer._compute_loss()
        ↓
TRLPPOTrainer._compute_loss()
        ↓
TRLPPOTrainer._compute_ppo_loss()
        ↓
TRLAuxLossPPOTrainer._compute_aux_loss()
```

## 9.1 `_compute_ppo_loss()`

仍然计算：

\[
L_{\mathrm{PPO}}
=
L_{\mathrm{policy}}
+c_vL_{\mathrm{value}}
+c_eL_{\mathrm{entropy}}
\]

其中包括：

- clipped policy surrogate
- clipped value loss
- entropy loss
- KL
- ratio
- clip fraction

但是父类汇总时：

```python
loss = ppo_loss * ppo_loss_coef
```

当前：

```yaml
ppo_loss_coef: 0.0
```

所以：

\[
0\cdot L_{\mathrm{PPO}}=0
\]

PPO loss 被计算和记录，但不提供有效梯度。

不过 `_compute_ppo_loss()` 中的 KL 自适应学习率逻辑仍会执行，这一点值得注意：

```python
_adjust_learning_rate_based_on_kl(kl_mean, optimizer)
```

也就是说，即便 PPO loss 权重为 0，计算得到的 policy KL 仍可能修改 optimizer learning rate。

## 9.2 普通 teacher MSE

以下函数不会执行：

```python
_compute_distill_teacher_loss()
```

因为：

```yaml
distill_teacher_loss_coef: 0.0
```

因此不存在：

\[
L_{\mathrm{direct-BC}}
=
\|\mu_{\mathrm{student}}-\mu_{\mathrm{teacher}}\|^2
\]

## 9.3 `_compute_aux_loss()`

位置：[ppo_trainer_aux_loss.py](/home/tide/robot/GRAIL/imports/SONIC/gear_sonic/trl/trainer/ppo_trainer_aux_loss.py:102)

计算：

```python
total_aux_loss_unscaled = (
    diffusion_loss_coef
    * diffusion_flow_loss
)

total_aux_loss = (
    aux_loss_scale
    * total_aux_loss_unscaled
)
```

当前两个系数都是 1：

\[
L_{\mathrm{aux}}=L_{\mathrm{flow}}
\]

## 9.4 最终 loss

```python
loss_dict["loss"] += aux_loss_result["total_aux_loss"]
```

因此当前实际优化目标为：

\[
L_{\mathrm{total}}
=
0\cdot L_{\mathrm{PPO}}
+
0\cdot L_{\mathrm{direct-BC}}
+
1\cdot L_{\mathrm{flow}}
\]

也就是：

\[
\boxed{
L_{\mathrm{total}}=L_{\mathrm{flow}}
}
\]

---

# 十、梯度下降与参数更新策略

每个 microbatch 执行：

```python
with accelerator.accumulate(model):
```

当前：

```yaml
gradient_accumulation_steps: 1
num_micro_batches: 1
```

所以实际上每个 minibatch 都立即更新一次，不跨 minibatch 累积梯度。

## 10.1 `accelerator.backward(loss)`

位置：[ppo_trainer.py](/home/tide/robot/GRAIL/imports/SONIC/gear_sonic/trl/trainer/ppo_trainer.py:2082)

```python
accelerator.backward(loss_dict["loss"])
```

梯度路径是：

```text
diffusion_flow_loss
        ↓
pred_velocity
        ↓
denoiser
        ↓
time encoder
        ↓
condition encoder
        ↓
student parameters
```

不产生梯度的路径：

```text
teacher                 torch.no_grad()
teacher action          detach()
action transform module torch.no_grad()
full decoder latent     detach()
target statistics       torch.no_grad()
noise / sampled t       非模型参数
```

由于 `ppo_loss_coef=0`：

- actor PPO gradient 为 0
- critic value gradient也被整体乘以 0
- critic 虽然 forward 并计算 value loss，但不会因该 loss 更新
- 实际主要更新 diffusion student 的 condition encoder、time encoder、denoiser

## 10.2 `_gradient_clipping()`

位置：[ppo_trainer.py](/home/tide/robot/GRAIL/imports/SONIC/gear_sonic/trl/trainer/ppo_trainer.py:2332)

先逐参数检查：

```python
torch.isnan(param.grad)
torch.isinf(param.grad)
```

若存在 NaN/Inf：

```python
optimizer.zero_grad()
return None
```

随后跳过本次 `optimizer.step()`。

梯度正常时：

```python
accelerator.clip_grad_norm_(
    model.parameters(),
    max_grad_norm=1.0,
)
```

将全局梯度范数限制在 1.0。

## 10.3 `optimizer.step()`

```python
if grad_norm is not None:
    optimizer.step()
```

更新 student 参数。

之后：

```python
optimizer.zero_grad()
```

清空梯度，为下一个 minibatch 做准备。

## 10.4 每 iteration 的更新次数

当前每个 iteration：

```text
3 epochs
× 4 minibatches/epoch
× 1 microbatch/minibatch
= 12 backward
= 12 optimizer.step
```

同一批 rollout 数据会被使用 3 次。

但是每次 student flow forward 都重新采样：

- Gaussian noise
- flow time \(t\)

所以即使使用相同的 observation 和 teacher target，三个 epoch 中构造的 \(x_t\) 和 velocity training sample 也不同。这相当于对同一个 teacher target 做新的 diffusion Monte Carlo 采样。

## 10.5 learning rate 更新

每个 microbatch 的 `_compute_ppo_loss()` 都会调用：

```python
_adjust_learning_rate_based_on_kl()
```

规则为：

```text
KL > 2 × desired_kl:
    lr = lr / 1.5

0 < KL < desired_kl / 2:
    lr = lr × 1.5

否则:
    lr 不变
```

并限制在：

```text
adaptive_lr_min = 1e-5
adaptive_lr_max = 2e-4
desired_kl = 0.01
```

iteration 结束后还执行：

```python
self.lr_scheduler.step()
```

但配置的 scheduler 是：

```yaml
lr_scheduler_type: constant
```

因此 scheduler 本身通常不会改变学习率；主要变化来自 KL adaptive adjustment。

---

# 十一、iteration 结束阶段

完成全部 12 次更新后，依次执行：

### `sync_running_mean_std()`

同步多 GPU 上的 normalization 统计量，包括可能存在的 observation/target running statistics。

### `sync_adaptive_sampling()`

同步环境 motion adaptive sampling 状态。

### `_get_train_metrics()`

汇总：

- PPO loss 指标
- KL
- entropy
- value loss
- flow auxiliary loss
- gradient/update 统计

`TRLAuxLossPPOTrainer` 还会额外记录：

```text
loss/aux/diffusion_flow_avg
loss/total_aux_loss_unscaled_avg
loss/total_aux_loss_avg
```

### `log(metrics)`

记录：

- rollout reward
- episode length
- collection time
- learning time
- FPS
- 当前 learning rate
- iteration/global step

### `lr_scheduler.step()`

推进 constant scheduler。

### `callback_handler.on_step_end()`

处理：

- checkpoint 保存
- evaluation
- early stopping
- callback 状态更新

配置中通常每 500 iteration 保存/评估一次。

---

## 十二、当前实现的几个关键特征

1. 这是 on-policy DAgger 数据采集，但不是 teacher-action rollout。

环境中的状态分布由当前 student 产生：

```text
student action → env.step → next observation
```

teacher 只对 student 到达的 observation 提供训练标签。这正是 DAgger 的核心思想之一：在 learner 自己访问的状态分布上请求 expert label。

2. rollout 阶段不执行 teacher。

teacher 只在 minibatch training forward 中运行。因此每轮 teacher forward 次数是：

```text
3 epochs × 4 minibatches = 12 次
```

而不是 rollout 的 16 个 step 分别调用 teacher。

3. 时间维不打散。

minibatch 按环境轨迹划分：

```text
完整数据：[16 env, 16 steps]
minibatch：[4 env, 16 steps]
```

这样保留时序结构和 episode attention mask。

4. PPO 框架主要充当训练基础设施。

当前有效目标只有 flow matching：

```text
PPO loss：计算但权重为 0
直接 teacher MSE：关闭
flow loss：有效
```

因此这里更接近：

```text
on-policy DAgger data collection
+ frozen teacher relabeling
+ flow-matching behavior distillation
```

而不是通常意义上的 PPO policy-gradient 训练。

5. critic 当前不会得到有效梯度。

因为完整 `ppo_loss` 在父类中整体乘以：

```python
ppo_loss_coef = 0
```

value loss 也包含在这个整体 PPO loss 中。因此 critic 虽然参与 rollout 后的 value/GAE 计算和训练 forward，但并不会通过 value loss 更新。对于纯 diffusion DAgger 训练而言，这些计算主要是沿用 PPO trainer 框架产生的额外开销。