# GR00T Flow Matching 向 Sugar-IL Generator 的迁移计划

> 执行更新（2026-07-27）：根据后续要求，Phase 1 已改为一次性纯 flow-matching
> 切换，不保留 DDIM scheduler、旧构造参数或 checkpoint 兼容层；模型源码迁移到
> `sugar_il/model/flowmatching`，训练配置及训练/IsaacLab workspace 同步切换。下文中
> 关于 Phase 1A 临时兼容 DDIM、Phase 1B 再清理配置的描述仅保留为原始方案记录，不再是
> 当前实现要求。

## 1. 目标与本轮边界

最终目标是把本仓库 `Isaac-GR00T` 中 GR00T N1.7 action head 的 flow-matching
训练与生成机制，分阶段迁移到：

`GRAIL/imports/SONIC/sugar_il/sugar_il/policy/generator.py`

迁移必须先保持 Sugar-IL 当前的数据、训练 workspace 和 SONIC 推理接口稳定，再逐步替换
DiT、动作编解码器和条件编码器。本计划的第一实现阶段只迁移 flow-matching 的数学目标和
Euler 生成过程，不立即照搬 GR00T 的 VLM、AlternateVLDiT 或 multi-embodiment 层。

本文件分析的是当前工作区中的 **Isaac-GR00T N1.7 默认配置**。如果后续指定某个 GR00T
checkpoint，应以该 checkpoint 自带的 `config.json` 为最终参数来源，因为 Hugging Face
checkpoint 可以覆盖源码默认值。

本轮只新增此计划文件，没有修改 `generator.py`。

## 2. 源码地图：哪些文件构成 GR00T flow matching 的核心

### 2.1 必须理解和迁移的核心代码

1. `Isaac-GR00T/gr00t/model/gr00t_n1d7/gr00t_n1d7.py`

   - 第 38-119 行 `Gr00tN1d7ActionHead.__init__`：组装 DiT、state encoder、
     action encoder、action decoder、时间采样器。
   - 第 170-173 行 `sample_time`：从变换后的 Beta 分布采样连续 flow time。
   - 第 182-286 行 `forward`：构造线性概率路径、速度目标并计算 masked MSE。
   - 第 325-443 行 `get_action_with_features`：从高斯噪声出发，用显式 Euler 法积分
     速度场。
   - 这是第一阶段最直接的迁移来源。

2. `Isaac-GR00T/gr00t/configs/model/gr00t_n1d7.py`

   - 第 73-123 行定义 action head、DiT、flow time、推理步数、horizon 等默认参数。
   - 迁移时不能从调用现场猜参数，必须从这里或目标 checkpoint 配置读取。

3. `Isaac-GR00T/gr00t/model/modules/dit.py`

   - 第 61-71 行 `TimestepEncoder`：DiT 的扩散/flow 时间嵌入。
   - 第 74-98 行 `AdaLayerNorm`：用时间嵌入调制每个 transformer block。
   - 第 100-219 行 `BasicTransformerBlock`：单个 attention 加 FFN。
   - 第 222-336 行 `DiT`：交错 self-attention 与 cross-attention。
   - 第 339 行起 `AlternateVLDiT`：进一步在 cross-attention block 中交替关注图像和
     非图像 token。

4. `Isaac-GR00T/gr00t/model/modules/embodiment_conditioned_mlp.py`

   - 第 26-56 行 `SinusoidalPositionalEncoding`：action encoder 内使用的时间正弦
     编码。
   - 第 143-174 行 `CategorySpecificMLP`：按 embodiment 选择独立权重的 state
     encoder 和 decoder。
   - 第 177-238 行 `MultiEmbodimentActionEncoder`：联合编码动作、flow time 和
     embodiment。

### 2.2 完整 GR00T 输入链路中的相关代码

5. `Isaac-GR00T/gr00t/model/modules/qwen3_backbone.py`

   - Qwen3-VL/Cosmos-Reason2-2B 产生 `[B, S, 2048]` 的视觉语言 token。
   - 同时返回 `backbone_attention_mask` 和 `image_mask`。
   - `AlternateVLDiT` 依赖这两个 mask，因此不能直接接到只有状态 token 的 Sugar-IL。

6. `Isaac-GR00T/gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py`

   - 负责 state/action 的补齐、归一化和 `action_mask`。
   - GR00T 的 masked loss 依赖这里提供的有效 action 维度和 horizon mask。

### 2.3 Sugar-IL 当前对应代码

1. `sugar_il/policy/generator.py`

   - 第 40-53 行是 DDIM scheduler 生成循环。
   - 第 75-95 行是 DDIM loss、latent 归一化和 hand loss。

2. `sugar_il/model/diffusion/transformer_for_action_diffusion.py`

   - 第 12-23 行是 timestep condition token 的编码器。
   - 第 26-43 行是同时包含 self/cross attention 的 RDT block。
   - 第 46-78 行是当前 12 层、256 hidden 的 action diffusion transformer。
   - 第 81-92 行 `HandPrimitiveHead` 是独立于 latent diffusion 的二分类输出头。

3. `sugar_il/model/encoder/generator_state_encoder.py`

   - 把 object-aware 观测编码成 4 个 256 维 condition token。

4. `sugar_il/config/train_generator_workspace.yaml`

   - 当前 DDIM scheduler、50 个训练 timestep、16 个推理 step 和网络超参数。

5. `sugar_il/config/task/ObjectAware.yaml`、
   `sugar_il/dataset/generator_dataset.py`、
   `sugar_il/workspace/run_generator_isaaclab.py`

   - 共同固定了预测 horizon 40、latent 64、hand 2。
   - 部署端一次预测 40 帧，默认执行前 20 帧；预测 horizon 和执行 horizon 不能混为一谈。

## 3. GR00T N1.7 的 flow-matching 算法

设归一化后的真实动作轨迹为 `x_data`，独立高斯噪声为 `z`。

### 3.1 训练时间采样

默认参数：

- `u ~ Beta(alpha=1.5, beta=1.0)`
- `t = (1 - u) * noise_s`
- `noise_s = 0.999`
- 连续 `t` 再离散为 `t_bucket = floor(t * 1000)`
- `num_timestep_buckets = 1000`

`t` 的均值约为 `0.3996`。Beta 采样并非 DDIM 的均匀离散 timestep。
GR00T 特意在 CPU/FP32 上构造 Beta 分布，再把样本移动到目标 device/dtype，以避免
meta-device 或 bf16 初始化改变时间分布。

### 3.2 线性概率路径与监督目标

GR00T 使用从噪声到数据的直线路径：

```text
x_t = (1 - t) * z + t * x_data
v_target = x_data - z
```

网络预测 `v_theta(x_t, t, condition)`，loss 为：

```text
element_loss = (v_theta - v_target)^2 * action_mask
loss = element_loss.sum() / (action_mask.sum() + 1e-6)
```

因此模型预测的是 **velocity/向量场**，不是 DDIM 的 `epsilon`，也不是直接预测
`x_data`。

### 3.3 推理/生成

默认：

- 初始 `x_0 ~ N(0, I)`
- `num_inference_timesteps = 4`
- `dt = 1 / 4`
- 时间从噪声端向数据端递增：`t = 0, 0.25, 0.5, 0.75`
- 输入网络的 bucket 为 `0, 250, 500, 750`
- 显式 Euler 更新：

```text
x_{k+1} = x_k + dt * v_theta(x_k, t_k, condition)
```

GR00T 另有 RTC overlap/inpainting、冻结前缀和速度 ramp。这不是基础 flow matching
成立所必需的部分，不进入第一阶段。

## 4. GR00T 默认结构与关键参数

### 4.1 总体形状

| 项目 | GR00T N1.7 默认值 |
|---|---:|
| VLM backbone | `nvidia/Cosmos-Reason2-2B` |
| VLM token 维度 | 2048 |
| 最大 state 维度 | 132 |
| 最大 action 维度 | 132 |
| state history | 1 |
| 预测 action horizon | 40 |
| action/state token 输入维度 | 1536 |
| DiT 内部宽度 | `32 heads * 48 = 1536` |
| DiT 输出维度 | 1024 |
| DiT 层数 | 16 |
| attention heads | 32 |
| head dimension | 48 |
| dropout | 0.2 |
| 推理积分步数 | 4 |
| flow time buckets | 1000 |
| state dropout | 0.8 |
| 最大 embodiment 数 | 32 |

这里的预测 horizon 40 是模型生成 chunk 的长度，不代表部署端必须执行全部 40 步。

由于 category-specific linear 为 32 个 embodiment 各保存一套权重，默认 state encoder、
action encoder 和 action decoder 三者合计约 **325.8M 参数**（分别约 54.7M、233.1M、
37.9M），尚未包含 DiT 和约 2B 的 VLM backbone。这也是 Sugar 单任务版本不应直接复制
32 份 projector 的主要原因。

### 4.2 输入 encoder

1. VLM 条件：

   - Qwen3-VL/Cosmos backbone 输出 `[B, S, 2048]`。
   - 先经过 LayerNorm；可选的 VL self-attention 默认未配置时为 Identity。

2. state encoder：

   - 输入 `[B, 1, 132 * state_history]`。
   - embodiment-specific 两层 MLP：`132 -> 1024 -> 1536`。
   - 中间激活为 ReLU。
   - 默认训练时以 0.8 概率丢弃整个 state token。

3. action encoder：

   - embodiment-specific `W1: 132 -> 1536`。
   - flow timestep 扩展到所有 horizon token，并编码成 1536 维 sin/cos。
   - 动作嵌入与时间嵌入拼接后：
     `W2: 3072 -> 1536`，Swish，再经 `W3: 1536 -> 1536`。
   - action token 额外加一个长度上限 1024 的 learned sequence embedding。

### 4.3 两种时间/位置编码必须区分

GR00T action head 中实际上有三类编码：

1. **Action encoder 的 flow-time 正弦编码**

   - 频率为 `exp(-i * log(10000) / half_dim)`。
   - 输出顺序为 `[sin, cos]`。
   - 同一个样本的 flow time 被复制到全部 action horizon token。

2. **DiT 的 flow-time 编码**

   - diffusers `Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=1)`。
   - 再经 `TimestepEmbedding(256 -> 1536)`。
   - 用于每个 block 的 AdaLayerNorm，并再次调制最终输出 norm。

3. **Action horizon 位置编码**

   - action head 使用 learned `nn.Embedding(max_seq_len=1024, dim=1536)`。
   - 默认 `diffusion_model_cfg.positional_embeddings=None`，所以 DiT block 内不再添加
     sinusoidal sequence position。

### 4.4 DiT 设计

默认使用 `AlternateVLDiT`：

- 共 16 个 block。
- 偶数 block 做 cross-attention，奇数 block 做 self-attention。
- 每个 block 只有一次 attention，然后接 FFN；不是在同一 block 中依次做
  self-attention 和 cross-attention。
- cross-attention block 按 `attend_text_every_n_blocks=2` 在非图像 token 与图像
  token 之间交替。
- 每个 block 的输入 norm 是 time-conditioned AdaLayerNorm。
- FFN 使用 diffusers `FeedForward`，默认激活来自 `DiT` 构造器的
  `gelu-approximate`。
- 最终用 timestep 产生 shift/scale，调制无 affine 的 LayerNorm，再线性投影
  `1536 -> 1024`。

### 4.5 输出 decoder

- embodiment-specific 两层 MLP：`1024 -> 1024 -> 132`。
- 中间 ReLU。
- DiT 序列包含一个 state token 和 40 个 action token，decoder 后只截取最后
  40 个 action token。
- 数据处理器提供 `action_mask`，屏蔽某个 embodiment 不存在的 action 维度或补齐位置。

## 5. 与 Sugar-IL 当前 DDIM 的逐项对比

| 维度 | GR00T flow matching | Sugar-IL 当前 DDIM | 迁移含义 |
|---|---|---|---|
| 训练路径 | `(1-t)z + t*x` | scheduler 的 `q(x_t|x_0)` | 第一阶段必须替换 |
| 监督目标 | `x-z` velocity | 默认 `epsilon` | 旧 checkpoint 语义不兼容 |
| 训练时间 | 变换 Beta 连续采样后映射到 1000 buckets | `[0,49]` 均匀整数 | 替换采样器和时间尺度 |
| 推理方向 | `t: 0 -> 1` | scheduler timestep 递减 | 不能复用 DDIM loop |
| 推理更新 | 4 步 Euler | 16 步 `DDIMScheduler.step` | 第一阶段替换 |
| scheduler | 不需要 | diffusers DDIMScheduler | 第一阶段停止调用，随后移除配置 |
| 预测 horizon | 40 | 16 | 第一阶段保持 Sugar 的 16 |
| 连续动作维度 | 最大 132 | latent 64 | 第一阶段保持 64 |
| 离散 hand | 包含在通用 action 维度/掩码体系 | 独立 2 维 BCE head | 第一阶段保留 Sugar hand head |
| 条件输入 | VLM token + 1 state token | 4 个 object-aware state token | 第一阶段保留 Sugar encoder |
| action token 宽度 | 1536 | 256 | 第一阶段保留 256 |
| DiT 深度 | 16 | 12 | 后续独立迁移 |
| attention | 32 heads, 48/head | 8 heads, 32/head | 后续独立迁移 |
| block 拓扑 | 每层 cross 或 self，交错 | 每层 self + cross + FFN | 后续独立迁移 |
| norm | LayerNorm + time AdaLN | RMSNorm，time 作为 condition token | 后续独立迁移 |
| action position | learned embedding，最大 1024 | learned `[1,16,256]` | 第一阶段可保持 |
| condition position | VLM 自身位置体系/mask | learned `[1,5,256]` | 第一阶段可保持 |
| action loss mask | 有，支持补齐维度/horizon | 无，固定 `[16,64]` | 当前固定形状下不阻塞 |
| 归一化 | GR00T processor 的 state/action 归一化 | `LinearNormalizer["latent"]` | 第一阶段必须保留 Sugar 归一化 |

Sugar-IL 当前默认网络的实测参数量：

- action diffusion transformer：12,802,368
- hand head：2,112,002
- observation encoder：303,616
- 三者合计：15,217,986

Sugar DiT 的具体结构为：

- action：`Linear(64 -> 256)` 加 learned 40-frame position embedding；
- condition：4 个 observation token，加 1 个 256 维 timestep token，共 5 token；
- 12 个 RDT block，每个 block 顺序执行 RMSNorm/self-attention、
  RMSNorm/cross-attention、RMSNorm/4x FFN；
- 8 heads，hidden 256，FFN hidden 1024，dropout 0.1；
- final RMSNorm 后 `Linear(256 -> 64)`。

## 6. 迁移原则与关键决策

1. **先迁移 objective，再迁移 architecture。**

   Sugar 当前 DiT 已经能够输出与输入同形状的张量，因此它可以先学习 velocity。
   flow matching 不要求先换成 GR00T 的 16 层 DiT。

2. **第一阶段不改变 Sugar 的外部接口。**

   以下接口必须保持：

   - `compute_loss(batch, training=True) -> dict`
   - `forward(...) -> scalar loss`
   - `predict_action(obs_dict, gen_attn_map=False) -> dict`
   - 输出 `latent`、`hand_logits`、`hand_probability`、`hand_primitive`、`action`
   - latent `[B,16,64]`，hand `[B,16,2]`

3. **第一阶段不迁移 AlternateVLDiT。**

   Sugar 没有 image token、text token 或 image mask。直接移植 AlternateVLDiT 会制造
   虚假的条件分组。后续如迁移 GR00T block，应先用基础 `DiT` 的交错
   self/cross 结构，`cross_attention_dim=256`。

4. **第一阶段不迁移 multi-embodiment 权重。**

   Sugar 当前只有一套固定 action schema。按 32 个 embodiment 复制大矩阵会显著增加
   参数量且没有收益。后续需要多机器人时再增加轻量 embodiment embedding 或
   category-specific projector。

5. **旧 DDIM checkpoint 必须显式拒绝。**

   旧权重输出 `epsilon`，新 Euler loop 期待 velocity。两者张量形状相同，普通
   `load_state_dict` 可能静默成功但推理完全错误。第一阶段需在 `Generator` 注册一个
   persistent objective/version buffer；新 checkpoint 含该标记，旧 checkpoint 在默认
   `strict=True` 加载时必须失败。

6. **预测 horizon 与执行 horizon 分开管理。**

   当前预测 40 帧，SONIC 执行前 20 帧后重新规划。

## 7. 分阶段实施计划

### Phase 0：建立 DDIM 基线与测试护栏

目标：在修改 objective 前记录当前行为，避免把既有接口问题误判为 flow-matching 问题。

工作：

1. 为 `Generator` 建立 CPU 单元测试，使用小 batch 和固定随机种子。
2. 记录：

   - `compute_loss` 返回键和标量形状；
   - `predict_action` 每个输出的形状、dtype、device；
   - `gen_attn_map=True/False` 两种路径；
   - hand BCE 与 latent loss 的组合关系；
   - normalizer 前后的范围。

3. 保存当前 DDIM 的短训练 loss 曲线和固定验证 batch 指标，作为迁移前基线。

验收：

- 测试能在不启动 Isaac Lab 的情况下执行。
- 不依赖真实大 checkpoint。

### Phase 1A：只替换 `generator.py` 的 loss 与生成数学

这是第一份代码变更，目标文件仅为：

`sugar_il/policy/generator.py`

具体修改：

1. 构造参数新增：

   - `num_inference_steps=4`
   - `noise_beta_alpha=1.5`
   - `noise_beta_beta=1.0`
   - `noise_s=0.999`
   - `num_timestep_buckets=1000`
   - `objective_type="flow_matching"`

2. 暂时接受现有 `noise_scheduler` 参数以兼容 Hydra 配置，但不再调用
   `set_timesteps`、`add_noise` 或 `step`。在 Phase 1B 才从 YAML 删除 scheduler。

3. 仿照 GR00T 在 CPU/FP32 上构造 `Beta`，实现独立的 `_sample_time`。

4. `compute_loss` 改为：

   ```text
   x = normalize(batch["action"]["latent"])
   z = randn_like(x)
   t = transformed_beta_sample              # [B,1,1]
   x_t = (1-t)*z + t*x
   target_velocity = x-z
   time_bucket = floor(t*1000)              # [B]
   pred_velocity = model(x_t, time_bucket, condition)
   latent_loss = mse(pred_velocity, target_velocity)
   total_loss = latent_loss + hand_loss_weight*hand_loss
   ```

5. 保留返回键 `latent_loss`，避免 workspace/logging 破坏；可以同时增加
   `flow_loss` 作为同一 tensor 的语义明确别名。

6. hand 分支保持原样：

   - `HandPrimitiveHead(condition)`
   - `binary_cross_entropy_with_logits`
   - 推理时 sigmoid + 0.5 threshold

7. `conditional_sample` 改为从高斯噪声开始的正向 4 步 Euler：

   ```text
   dt = 1 / num_inference_steps
   for k in range(num_inference_steps):
       t = k / num_inference_steps
       bucket = floor(t * num_timestep_buckets)
       velocity, maps = model(x, bucket, condition)
       x = x + dt * velocity
   ```

8. `attention_maps` 的 key 改为实际 flow bucket，例如默认
   `0/250/500/750`，并在文档/测试中明确不再是 DDIM scheduler timestep。

9. 在 `Generator` 注册持久化版本标记，例如 `_objective_version=1`。旧 DDIM
   checkpoint 因缺少该 key 而在严格加载时失败。不得通过 `strict=False` 把旧权重直接
   用于 flow 采样。

10. 不修改 `obs_encoder`、`TransformerForActionDiffusion`、normalizer、hand head、
    horizon 或 latent dimension。

Phase 1A 单元测试：

1. **路径代数测试**：固定 `x/z/t`，精确验证 `x_t` 和 `x-z`。
2. **时间分布测试**：验证样本在 `[0, noise_s]`，并校验变换 Beta 的均值/方差。
3. **time bucket 测试**：模型收到 `[B]` long tensor，范围 `[0,999]`。
4. **常速度 Euler 测试**：mock 模型输出常数速度，精确验证 N 次更新结果。
5. **调用次数测试**：默认推理恰好调用模型 4 次。
6. **输出契约测试**：所有现有输出键、shape、dtype、device 不变。
7. **梯度测试**：flow loss 有限，能对 DiT 参数反向传播；hand loss 仍能更新 hand head。
8. **scheduler 隔离测试**：传入会在任何方法调用时抛错的 fake scheduler，训练和推理
   仍应成功，以证明 DDIM 已完全退出数据路径。
9. **checkpoint guard 测试**：缺少 objective version 的旧 state dict 严格加载失败。

Phase 1A 验收：

- `generator.py` 中不存在 scheduler 驱动的训练或采样调用。
- 默认测试推理为 4 次网络 forward。
- `[B,16,64]` latent 和 `[B,16,2]` hand 输出不变。
- 旧 DDIM checkpoint 不会静默加载。

### Phase 1B：配置切换与短训练验证

目标文件：

- `sugar_il/config/train_generator_workspace.yaml`
- 必要的 generator 单元测试/说明文档

修改：

1. 删除 `diffusers.DDIMScheduler` 配置块。
2. 明确配置：

   ```yaml
   num_inference_steps: 4
   noise_beta_alpha: 1.5
   noise_beta_beta: 1.0
   noise_s: 0.999
   num_timestep_buckets: 1000
   objective_type: flow_matching
   ```

3. 禁止通过 `start_ckpt_path` 恢复旧 DDIM checkpoint。必须从新初始化训练，或从明确
   标记为 flow-matching 的 checkpoint 恢复。
4. 做一个小数据集 overfit：

   - loss 能持续下降；
   - 预测 latent MSE 优于随机输出；
   - 无 NaN/Inf；
   - 4 步生成稳定。

5. 再做完整训练对比，记录 DDIM 与 flow matching 的：

   - wall-clock/step；
   - inference latency（16 次 forward 对 4 次 forward）；
   - latent MSE；
   - left/right hand F1；
   - rollout 成功率和动作平滑度。

只有 Phase 1B 通过后，才开始迁移网络结构。

### Phase 2：迁移 GR00T 的时间条件方式，不扩大主干

目的：单独验证 GR00T 的时间编码和 AdaLN 是否带来收益，避免和网络扩容混杂。

建议：

1. 保持 hidden 256、horizon 40、latent 64 和现有 Sugar condition token。
2. 把当前“time 作为第 5 个 cross-attention condition token”改为：

   - 256 通道 sinusoidal timestep projection；
   - timestep MLP 到 hidden 256；
   - 用 timestep 对每个 block 做 AdaLayerNorm；
   - 最终 norm 也用 timestep shift/scale。

3. 可选地加入 GR00T 风格 action encoder：

   - `64 -> 256` action projection；
   - 与 256 维 timestep sin/cos 拼接；
   - `512 -> 256 -> 256`；
   - 单 embodiment 普通 Linear，不使用 32 份 category-specific 权重。

4. 保留 Sugar learned 40-frame action position embedding。
5. 与 Phase 1 的纯 flow-matching 模型做单变量消融。

验收：

- 参数增长可解释；
- 相同数据和 seed 下训练稳定；
- 4 步 Euler 指标不退化；
- 不改变外部输出协议。

### Phase 3：迁移 GR00T DiT block 拓扑

推荐先做“Sugar 尺度适配版”，再决定是否上 GR00T 原始宽度。

适配版：

- hidden 256 或 512；
- 16 blocks；
- 偶数 block cross-attention，奇数 block self-attention；
- 每个 block 均有 FFN；
- time AdaLayerNorm；
- cross condition 仍为 Sugar 的 4 个 object-aware token；
- 不使用 AlternateVLDiT 的 image/non-image mask；
- decoder 输出 64 维 velocity。

严格尺寸版（仅在数据量、显存和延迟允许时评估）：

- token width 1536；
- 16 blocks；
- 32 heads，48 dim/head；
- DiT output 1024；
- 新建 `1024 -> 1024 -> 64` 的单 embodiment decoder；
- condition cross-attention dimension 保持 Sugar encoder 的 256，或先明确投影策略；
- 不复制 GR00T 的 32 embodiment 巨型权重。

比较指标：

- 参数量和显存；
- 单次 forward latency；
- 4-step 端到端 latency；
- validation latent MSE；
- rollout 成功率；
- 是否出现因小数据集导致的过拟合。

### Phase 4：评估 state/action encoder 与 decoder 的进一步迁移

1. state encoder：

   - GR00T 把全部 state 拼成 1 个 token；
   - Sugar 当前保留 BPS/current/target/proprioception 共 4 个语义 token。

   推荐先保留 Sugar 4-token 结构，因为它对应当前任务和 attention 可解释性。另开消融
   比较“4 token”与“flatten 后单 token”，不要直接覆盖。

2. state dropout：

   - GR00T 默认整 token dropout 0.8；
   - Sugar 只对 proprioception token 使用 0.1 dropout。

   两者条件信息不同，不能直接把 Sugar dropout 提到 0.8。建议测试
   `0.1/0.3/0.5`，并分别记录 object/target/proprioception 的 dropout。

3. decoder：

   - latent velocity 可从当前 `Linear(256 -> 64)` 逐步换为 GR00T 风格两层 MLP；
   - hand 仍保留独立 BCE head，除非另有实验表明应把 binary hand 纳入连续 flow。

4. action mask：

   - 当前固定 `[16,64]` 时不需要 mask；
   - 只有引入可变 horizon、可变 action schema 或 padding 后，才增加 GR00T 风格 masked
     flow loss。

### Phase 5：40-frame horizon（已实现）

训练和部署已统一为连续 50 Hz 的 40-frame prediction horizon，部署端每次执行前
20 帧。旧的 waypoint 插值路径已删除。

再根据闭环成功率和延迟决定。

### Phase 6：可选的完整 GR00T 条件输入与 RTC

只有当目标明确需要图像/语言输入时，才迁移：

- Cosmos/Qwen3-VL backbone；
- image/text masks；
- AlternateVLDiT；
- 2048 维 VLM condition；
- 对应 processor/collator。

这将把 Sugar object-aware generator 变成 VLA 模型，属于独立项目，不应作为基础
flow-matching 迁移的默认范围。

RTC overlap/inpainting 也应在基础 4-step flow rollout 稳定后单独加入，并为
overlap、frozen prefix、ramp 和延迟补偿分别测试。

## 8. 推荐提交顺序

1. `test: lock current generator API and output shapes`
2. `feat: replace DDIM objective and sampling in generator.py with flow matching`
3. `config: remove DDIM scheduler and enable four-step flow sampling`
4. `test: add flow path, Beta time, Euler and checkpoint compatibility tests`
5. `experiment: train and benchmark Sugar DiT with flow objective`
6. `feat: add GR00T-style timestep AdaLN behind a config flag`
7. `experiment: compare current RDT blocks with interleaved GR00T DiT blocks`
8. `feat: migrate selected encoders/decoder only after ablation`
9. `experiment: evaluate horizon 40 as an independent change`

每个阶段使用独立 checkpoint 目录和明确的 `objective_type`/architecture version，禁止把
DDIM、flow-v1、AdaLN 或新 DiT checkpoint 混用。

## 9. 第一阶段完成定义

只有同时满足以下条件，才认为“flow-matching loss 计算与生成已迁移”：

- 训练 noisy trajectory 使用线性路径 `(1-t)z + t*x`；
- target 是 `x-z` velocity；
- 时间来自 GR00T 默认变换 Beta 分布，并映射到 1000 buckets；
- 推理从高斯噪声出发，默认执行 4 次正向 Euler 更新；
- 训练和推理均不调用 DDIM scheduler；
- Sugar observation encoder、normalizer、hand head、输出字典和 `[16,64]+[16,2]`
  契约保持不变；
- 旧 DDIM checkpoint 被显式拒绝；
- 单元测试、短 overfit 和一次 SONIC 闭环 smoke test 通过；
- 结果文档明确称为“GR00T flow objective + Sugar architecture”，而不是误称为完整
  GR00T DiT 迁移。
