# `sugar_il` 代码导读：Object-Aware Action Diffusion Transformer

本文面向已经理解扩散模型和 Transformer、但尚未系统学习 DiT 的读者。分析对象是当前目录中的实际代码，主线限定为目前可工作的 `task=ObjectAware`。文中的 `B` 表示 batch size，`H=16` 表示动作预测长度，`D_a=64` 表示连续 latent 维数，`D=256` 表示 Transformer 隐藏维数，`N_c=4` 表示观测条件 token 数，`N_c+1=5` 表示加入扩散时间步后的条件 token 数。

> 先给出最重要的判断：代码把类命名为 `TransformerForActionDiffusion`，README 称其为 DiT，但它不是原始图像 DiT 的逐项复现。它没有图像 patch、AdaLN/FiLM 调制、zero-gate 或 class token；它更准确地说是一个用于动作扩散的 encoder-decoder 式 Transformer：带噪动作序列做 self-attention，观测与时间步 token 通过 cross-attention 注入。理解这一差异，可以避免把经典 DiT 论文中的机制误认为代码已经实现。

## 1. 全局视角

### 1.1 项目要解决什么问题

给定当前物体状态、目标物体状态、物体 BPS 特征以及上一帧动作，模型一次生成未来 16 帧动作：

- 连续部分：`combined_latent_pre_fsq`，每帧 64 维；
- 离散部分：左右手开合 primitive，每帧 2 个独立二值量。

连续 latent 用条件扩散模型生成；手部 primitive 不参与扩散，而由独立 Transformer decoder 直接分类。最终每帧拼成 66 维 meta-action。在线执行采用 receding horizon：每次预测 16 帧，只执行前 8 帧，然后重新观测并重规划。

### 1.2 网络分类

| 子网络 | 定义位置 | 输入 → 输出 | 性质 |
|---|---|---|---|
| `GeneratorStateObsEncoder` | `model/encoder/generator_state_encoder.py` | 9 个观测字段 → 4 个 `[D]` token | 4 个彼此独立的 MLP 编码器 |
| `TransformerForActionDiffusion` | `model/diffusion/transformer_for_action_diffusion.py` | `[B,16,64]` 带噪 latent、时间步、4 条件 token → `[B,16,64]` | 12 层动作 self-attention + 条件 cross-attention，预测噪声 |
| `HandPrimitiveHead` | 同上 | `[B,4,256]` 条件 → `[B,16,2]` logits | 2 层 Transformer decoder，直接二分类 |
| `Generator` | `policy/generator.py` | 封装上述网络、normalizer 和 DDIM scheduler | 完整训练损失与采样策略 |

### 1.3 静态嵌套关系

```text
TrainGeneratorWorkspace
└── Generator policy
    ├── GeneratorStateObsEncoder
    │   ├── bps_net:         MLP(10 → 256 → 256)
    │   ├── object_net:      MLP(18 → 256 → 256)
    │   ├── target_net:      MLP(18 → 256 → 256)
    │   └── last_action_net: MLP(66 → 256 → 256)
    ├── TransformerForActionDiffusion
    │   ├── input_emb: Linear(64 → 256)
    │   ├── TimestepEmbedder: sinusoidal(256) → MLP(256 → 256 → 256)
    │   ├── 12 × RDTBlock
    │   │   ├── RMSNorm → self-attention → residual
    │   │   ├── RMSNorm → cross-attention(condition) → residual
    │   │   └── RMSNorm → FFN(256 → 1024 → 256) → residual
    │   └── RMSNorm → Linear(256 → 64)
    ├── HandPrimitiveHead
    │   └── 16 learned queries → 2 × TransformerDecoderLayer → Linear(256 → 2)
    ├── diffusers.DDIMScheduler
    └── LinearNormalizer
```

Hydra 在 `config/train_generator_workspace.yaml` 中递归实例化这一树：workspace 实例化 `cfg.policy`；`Generator` 的 `noise_scheduler` 和 `obs_encoder` 参数也因包含 `_target_` 而先被 Hydra 实例化。

### 1.4 训练执行流程

```text
*.object_aware.pkl
  → GeneratorDataset.__getitem__
  → DataLoader 拼 batch
  → Generator.compute_loss
      → 观测逐字段归一化
      → GeneratorStateObsEncoder: 9 字段 → 4 条件 token
      → 真实 latent 归一化为 x₀
      → 随机 t、ε，DDIMScheduler.add_noise 得 x_t
      → TransformerForActionDiffusion(x_t,t,condition) 得 ε̂
      → MSE(ε̂,ε)
      → HandPrimitiveHead(condition) 得 logits
      → BCEWithLogits(logits, hand target)
      → 两项加权求和
  → backward → gradient clipping → AdamW → cosine LR scheduler
  → TensorBoard / JSON / CSV / PNG / checkpoint
```

### 1.5 推理执行流程

```text
Isaac Lab 当前世界状态
  → world_pose_to_body 构造与训练同名的 9 个观测张量
  → 归一化 → 4 条件 token
  → x_T ~ N(0,I), shape [B,16,64]
  → 16 个 DDIM 反向时间步：ε̂θ(x_t,t,c) → scheduler.step → x_{t-1}
  → latent 反归一化，shape [B,16,64]
  → hand_head(c) → sigmoid → threshold 0.5，shape [B,16,2]
  → 拼成 [B,16,66]
  → 在线脚本把手值 0/1 映射到 SONIC 的 -1/+1
  → 执行前 8 帧 → 重新规划
```

## 2. 从经典 DiT 到本项目实现

### 2.1 经典 DiT 的核心思想

扩散模型需要一个去噪器 `ε_θ(x_t,t,c)`。U-Net 是常见选择；DiT 用 Transformer 替代 U-Net。经典图像 DiT 通常把 latent image 切成 patch token，把时间与类别条件映射为调制向量，再用 adaptive LayerNorm 调制各 block：

\[
(\gamma,\beta,\alpha)=f(t,c),\qquad
\operatorname{adaLN}(x)=\gamma\odot\operatorname{LN}(x)+\beta,
\]

\[
x\leftarrow x+\alpha_{\rm attn}\odot \operatorname{Attn}(\operatorname{adaLN}(x)),
\quad
x\leftarrow x+\alpha_{\rm ffn}\odot \operatorname{FFN}(\operatorname{adaLN}(x)).
\]

本项目保留了“Transformer 充当扩散去噪器”这一高层定义，但条件注入方式改为 cross-attention。代码中不存在 `γ/β/α` 的生成层，也没有 adaptive norm。因此，本项目的“调制机制”应准确理解为：条件 token 改变 cross-attention 的 key/value 及注意力权重，进而以加性残差改变动作表示；不是 AdaLN 的逐通道仿射调制。

### 2.2 时间步编码：`TimestepEmbedder`

定义：`model/diffusion/transformer_for_action_diffusion.py:12-23`。由 `TransformerForActionDiffusion.forward` 调用。

对 batch 中每个整数扩散步 `t∈[0,49]`，先生成 256 维固定正余弦编码。令 `i=0,…,127`：

\[
\omega_i=\exp\left(-\log(10000)\frac{i}{128}\right),
\]

\[
e_{\rm sin}(t)=
[\cos(t\omega_0),\ldots,\cos(t\omega_{127}),
 \sin(t\omega_0),\ldots,\sin(t\omega_{127})].
\]

代码对应关系：

- `half = frequency_embedding_size // 2`：128 个频率；
- `torch.exp(-log(10000) * arange(half) / half)`：`ω_i`；
- `timestep[:,None] * frequencies[None]`：从 `[B]` 广播为 `[B,128]`；
- `cat(cos,sin)`：得到 `[B,256]`；
- `Linear → SiLU → Linear`：学习频率特征的非线性组合，仍为 `[B,256]`。

随后 `forward` 把它 `unsqueeze(1)` 成 `[B,1,256]`，拼在 4 个观测 token 后面，得到 `condition∈R^{B×5×256}`。因此时间步不是加到每个动作 token 上，而是一个可被 16 个动作位置查询的条件 token。

### 2.3 动作 token 和位置编码

定义：`TransformerForActionDiffusion.__init__/forward`，文件第 47-78 行。

输入 `sample=x_t∈R^{B×16×64}`。每一帧是一个 token：

\[
X^{(0)}=x_tW_{in}+b_{in}+P_a,
\quad X^{(0)}\in\mathbb{R}^{B\times16\times256}.
\]

`input_emb` 完成 `64→256`，可学习的 `pos_emb` 形状是 `[1,16,256]`。它表达序列中“未来第几帧”，而扩散时间 `t` 表达“当前噪声强度”；二者不是同一种时间。

条件序列同样加可学习 `cond_pos_emb∈R^{1×5×256}`，用位置区分 BPS、当前状态、目标状态、上一动作、扩散步。这里的位置实际也承担 token 类型标识的一部分，因为代码没有独立 type embedding。

### 2.4 `RDTBlock`：注意力、归一化和残差

定义：同文件第 26-43 行。由 `TransformerForActionDiffusion.forward` 的 12 层循环逐层调用。

每层是 pre-norm 结构。对输入 `X_l∈R^{B×16×256}`：

\[
\tilde X=\operatorname{RMSNorm}_1(X_l),
\qquad
X'=X_l+\operatorname{MHA}_{self}(\tilde X,\tilde X,\tilde X),
\]

\[
\hat X=\operatorname{RMSNorm}_2(X'),
\qquad
X''=X'+\operatorname{MHA}_{cross}(\hat X,C,C),
\]

\[
X_{l+1}=X''+\operatorname{FFN}(\operatorname{RMSNorm}_3(X'')).
\]

RMSNorm 不减均值。对单个 256 维 token：

\[
\operatorname{RMSNorm}(x)=g\odot
\frac{x}{\sqrt{\frac1D\sum_{j=1}^{D}x_j^2+10^{-6}}}.
\]

`g` 是可学习缩放参数，但与条件无关，所以不是 adaptive normalization。pre-norm 与三条恒等残差路径让梯度可以绕过 attention/FFN 子层，有助于深层训练稳定。

8 头 attention 中每头维数 `d_h=256/8=32`。单头计算为：

\[
Q=XW_Q,\quad K=YW_K,\quad V=YW_V,
\]

\[
A=\operatorname{softmax}(QK^\top/\sqrt{32}),
\qquad \operatorname{head}=AV.
\]

self-attention 取 `Y=X`，其注意力矩阵是 `[B,8,16,16]`，让不同未来帧协调；cross-attention 取 `Y=C`，矩阵是 `[B,8,16,5]`，让每个未来帧选择不同条件。代码只在 `gen_attn_map=True` 时返回 cross-attention 权重；self-attention 始终传 `need_weights=False`。

FFN 为 `Linear(256,1024) → GELU(tanh approximation) → Dropout → Linear(1024,256)`。attention 层自身也使用配置的 dropout 0.1。没有 causal mask：16 个未来动作位置双向可见，这符合一次联合去噪整段轨迹，而不是自回归生成。

### 2.5 条件融合与“调制”到底发生在哪里

四个观测 token 在每一个 `RDTBlock` 都作为同一组 key/value 被复用。第 `h` 个头、动作位置 `i` 对条件 token `j` 的权重为：

\[
a_{hij}=\operatorname{softmax}_j
\left(\frac{q_{hi}^{\top}k_{hj}}{\sqrt{32}}\right).
\]

输出 `Σ_j a_{hij}v_{hj}` 经 output projection 后加到动作表示。条件会同时改变 `K,V`，查询也随上一层状态改变，因此融合是动态的、逐层的、逐动作位置的。配置项 `use_attn_mask: True` 被 `Generator.__init__(**kwargs)` 接收后忽略，当前实际没有 attention mask。

### 2.6 最终噪声预测头

12 层后执行：

\[
\hat\epsilon_\theta=operatorname{Linear}_{256\to64}
(\operatorname{RMSNorm}(X_{12})),
\]

形状 `[B,16,64]`。输出头没有 zero initialization；所有默认 Linear 使用 PyTorch 默认初始化，只有动作/条件位置参数显式用 `N(0,0.02²)` 初始化。这也不同于常见 DiT 的 adaLN-Zero/零初始化输出设计。

## 3. 扩散过程与训练目标

### 3.1 前向加噪

`Generator.compute_loss` 定义于 `policy/generator.py:75-95`，由训练和验证循环直接调用。真实连续轨迹先归一化为 `x_0∈R^{B×16×64}`，再独立采样：

- `ε ~ N(0,I)`，与 `x_0` 同形状；
- 每个样本一个 `t ~ Uniform{0,…,49}`，形状 `[B]`。

`DDIMScheduler.add_noise` 实现标准闭式前向过程：

\[
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,
\qquad \bar\alpha_t=\prod_{s=1}^{t}(1-\beta_s).
\]

配置使用 50 个训练步、`squaredcos_cap_v2` beta schedule、`β_start=10^{-4}`、`β_end=0.02`。对该 diffusers schedule，具体 beta 序列由 scheduler 计算，不能简单视为线性插值。

### 3.2 噪声预测损失

配置 `prediction_type: epsilon`，所以监督目标就是采入的 `ε`：

\[
\mathcal L_{latent}
=\mathbb E_{x_0,t,\epsilon}
\left[\|\epsilon-\epsilon_\theta(x_t,t,c)\|_2^2\right].
\]

`F.mse_loss` 默认对 batch、16 帧和 64 通道全部取平均。代码也支持 `prediction_type: sample`，此时目标改成 `x_0`；不支持 `v_prediction`。

### 3.3 DDIM 反向采样

`Generator.conditional_sample` 从 `[B,16,64]` 标准高斯开始，调用 `set_timesteps(16)`，即从 50 个训练噪声级中选择 16 个推理步。每步先预测噪声，再由 `DDIMScheduler.step` 计算 `prev_sample`。概念上先估计：

\[
\hat x_0=rac{x_t-\sqrt{1-\bar\alpha_t}\hat\epsilon}{\sqrt{\bar\alpha_t}},
\]

再更新到较小噪声的 `x_{t-1}`。scheduler 默认 `eta=0` 时 DDIM 更新在给定初始噪声后是确定性的；但每次调用仍重新采样初始高斯，所以整体输出仍随机。`clip_sample=True` 会让 scheduler 在更新内部裁剪预测的 `x_0`，与训练数据被 limits normalizer 映射到约 `[-1,1]` 相配。

### 3.4 手部分类分支

`HandPrimitiveHead` 使用 16 个 learned query `Q_h∈R^{1×16×256}`。扩展到 batch 后，2 层标准 `TransformerDecoderLayer(norm_first=True)` 让 query 之间 self-attend，并 cross-attend 到 4 个观测 token，最后输出 `[B,16,2]` logits：

\[
\mathcal L_{hand}=
-\frac1{32B}\sum_{b,i,k}
[y_{bik}\log\sigma(z_{bik})+(1-y_{bik})\log(1-\sigma(z_{bik}))].
\]

总损失：

\[
\mathcal L=\mathcal L_{latent}+\lambda_{hand}\mathcal L_{hand},
\qquad \lambda_{hand}=1.
\]

两个手通道使用独立 sigmoid，不是二选一 softmax。推理阈值为 0.5。该分支只看观测条件，不看采样出来的 latent，也不看扩散时间步，所以连续动作和手动作仅通过共享 encoder、联合反向传播间接耦合。

## 4. 输入、输出和张量物理含义

### 4.1 单样本与 batch 形状

`GeneratorDataset.__getitem__` 返回单样本；DataLoader 在最前面增加 `B` 维。

| 字段 | 单样本 | batch | 物理含义 | encoder 分组 |
|---|---:|---:|---|---|
| `object_bps` | `[1,10]` | `[B,1,10]` | 物体固定 BPS/形状描述 | BPS token |
| `object_pos_b` | `[1,3]` | `[B,1,3]` | 当前物体相对当前机器人根的平移 | 当前 token |
| `object_ori_b_6d` | `[1,6]` | `[B,1,6]` | 当前物体相对姿态的 6D 表示 | 当前 token |
| `hand_object_transform_6d` | `[1,9]` | `[B,1,9]` | 当前手-物体变换特征；名称虽含 6d，实际 9 维 | 当前 token |
| `target_object_pos_b` | `[1,3]` | `[B,1,3]` | 轨迹最终物体世界位置在当前机器人根坐标系的表达 | 目标 token |
| `target_object_ori_b_6d` | `[1,6]` | `[B,1,6]` | 最终物体姿态相对当前机器人根的 6D 表达 | 目标 token |
| `target_hand_object_transform_6d` | `[1,9]` | `[B,1,9]` | episode 最后一帧手-物体特征 | 目标 token |
| `last_latent` | `[1,64]` | `[B,1,64]` | `t-1` 已执行连续 latent | 上一动作 token |
| `last_hand_primitive` | `[1,2]` | `[B,1,2]` | `t-1` 已执行左右手二值动作 | 上一动作 token |
| action `latent` | `[16,64]` | `[B,16,64]` | 真实帧 `t:t+16` 的连续动作 | 扩散目标 |
| action `hand_primitive` | `[16,2]` | `[B,16,2]` | 同期左右手二值动作 | BCE 目标 |

### 4.2 四个条件 token

`GeneratorStateObsEncoder.forward` 先强制每个观测形状为 `[B,1,D_i]`，去掉长度为 1 的时间轴，再构造：

\[
c_{bps}=MLP_{10}(bps),
\]

\[
c_{current}=MLP_{18}([p_o^b,r_o^{b,6d},h_o]),
\]

\[
c_{target}=MLP_{18}([p_g^b,r_g^{b,6d},h_g]),
\]

\[
c_{last}=MLP_{66}([a_{t-1}^{latent},a_{t-1}^{hand}]).
\]

每个 MLP 都是 `Linear(input,256) → LayerNorm → GELU → Dropout → Linear(256,256) → LayerNorm`。stack 后为 `[B,4,256]`。训练时以样本为单位用概率 0.1 把整个上一动作 token 置零；其他三个 token 不做 classifier-free condition dropout，代码也未实现 CFG。

### 4.3 输出字典

`Generator.predict_action` 返回：

- `latent [B,16,64]`：已从 normalized space 反归一化；
- `hand_logits [B,16,2]`：未归一化分类 logits；
- `hand_probability [B,16,2]`：sigmoid 概率；
- `hand_primitive [B,16,2]`：0/1 阈值结果；
- `action [B,16,66]`：`cat(latent, hand_primitive, dim=-1)`；
- 可选 `attention_maps`：以 scheduler timestep 为 key，每个值是 12 层 cross-attention map 列表，每张通常为 `[B,8,16,5]`，且已 detach 到 CPU。

## 5. 坐标变换与数据流水线

### 5.1 文件发现和严格校验

`dataset/generator_dataset.py` 的 `_resolve_paths` 接受单路径、目录、glob 或路径序列，最终只收集排序后的 `*.object_aware.pkl`。`_load_episode` 使用 pickle，因此只允许可信输入；它验证：

- `schema_version == 3`；
- pose 与 policy input 是 pre-step 对齐；
- quaternion 为 `wxyz`；
- 8 个必需数组均为 `[T,D]`、长度相同且有限；
- `T≥17`，手动作严格为 0/1，四元数近似单位长度；
- 同一 episode 中 10 维 `object_bps` 不变。

README 强调旧文件中预先计算的 body-frame `object_pos_b`、`object_ori_b_6d`、`target_object_pos` 被故意忽略；当前代码只信 world-frame pose，并在读取时重新计算相对坐标。

### 5.2 world frame 到 body frame

定义：`common/geometry.py`。训练的 `_pose` 与推理 wrapper 都调用同一个 `world_pose_to_body`。

机器人世界位姿为 `(p_r^w,q_r^w)`，物体世界位姿为 `(p_o^w,q_o^w)`：

\[
p_o^b=(R_r^w)^T(p_o^w-p_r^w),
\qquad q_o^b=(q_r^w)^{-1}\otimes q_o^w.
\]

`quaternion_to_matrix_wxyz` 先归一化四元数，再显式构造旋转矩阵；`quaternion_inverse_wxyz` 对单位四元数取共轭；`quaternion_multiply_wxyz` 实现 Hamilton product。`rotation_6d_columns` 取旋转矩阵前两列所在的 `[... ,3,2]` 子块后按 row-major reshape 为 6 维。它的具体顺序是行内交错的 `[r00,r01,r10,r11,r20,r21]`，消费端必须保持同一约定。

当前状态使用物体第 `t` 帧；目标状态使用物体最后一帧，但机器人坐标系仍取第 `t` 帧。因此目标是“从当前机器人视角看到的最终物体位姿”，不是最终机器人坐标系下的物体位姿。

### 5.3 窗口索引和数据划分

`GeneratorDataset.__init__` 先按 episode 随机打乱并划分验证集，避免同一 episode 的相邻窗口跨 train/val 泄漏。若至少两个 episode 且 `val_ratio>0`，验证 episode 数为 `min(N-1,max(1,round(N·ratio)))`。

每个 episode 的起点：

```python
t in range(1, T - horizon + 1)
```

所以 `t=1,…,T-16`，样本数 `T-16`。`t-1` 是上一已执行动作，`t:t+16` 是 16 个真实、未插值的监督帧。499 帧产生 483 个样本。

### 5.4 归一化

`get_normalizer` 遍历训练集的所有窗口，收集除二值 `last_hand_primitive` 外的每个观测字段，以及所有 action latent。默认 `mode='limits'`，每个通道独立拟合：

\[
s_j=\frac{2}{x_j^{max}-x_j^{min}},\qquad
o_j=-1-s_jx_j^{min},\qquad \tilde x_j=s_jx_j+o_j.
\]

近常量通道用 `range_eps=10^{-4}` 特判，使其落在输出区间中点附近。统计量和 scale/offset 被存为 `requires_grad=False` 的 `ParameterDict`，因此会随 policy checkpoint 的 `state_dict` 保存并随模型迁移设备。

需要注意，normalizer 是按“窗口出现次数”统计，不是按唯一帧统计：中间动作帧会落入多个 16 帧窗口，被重复计数。min/max 不受重复影响，但保存的 mean/std 会受窗口位置权重影响；当前 limits 模式实际只用 min/max。

二值上一手动作不归一化，未来手 target 也不归一化。连续预测采样完成后才用相同 affine transform 的逆变换恢复。

### 5.5 DataLoader

配置默认 train/val batch 都为 256，8 workers，pin memory 和 persistent workers 开启；训练 shuffle，验证不 shuffle。nested dict 由 PyTorch 默认 collate 递归堆叠。训练循环再用 `dict_apply` 递归执行 `.to(device, non_blocking=True)`。

## 6. 配置、训练、验证、日志和检查点

### 6.1 Hydra 配置

主配置是 `config/train_generator_workspace.yaml`。命令的 `task=ObjectAware` 把 `config/task/ObjectAware.yaml` 合入；`${...}` 做字段引用，`eval` resolver 用于遗留任务中的表达式。Hydra 把输出目录设为 `data/outputs/日期/时间_train_generator_ObjectAwareSONIC`，并写出最终配置、Hydra 自身配置和 override 记录到 `.hydra/`。

网络与扩散关键默认值：H=16，latent=64，D=256，12 layers，8 heads，dropout=0.1，50 train diffusion steps，16 inference steps。配置中的 `training.device` 没有用于选择训练设备；代码读取 `LOCAL_RANK`、直接调用 `torch.cuda.set_device`，再由 Accelerate 放置设备。因此当前训练入口实际要求 CUDA。mixed precision 也在代码中硬编码为 BF16。

### 6.2 workspace 初始化

`TrainGeneratorWorkspace.__init__`：设置 PyTorch/NumPy/Python seed；Hydra 实例化 `Generator`；可选复制 EMA 模型；若 `resume=False` 就把 optimizer 排除在 checkpoint 外；可选从 `start_ckpt_path` 加载；初始化 `global_step` 和 `epoch`。

`run` 依次建立 Accelerator、TensorBoard tracker、optimizer、dataset/DataLoader、normalizer、validation dataset、LR scheduler、可选 EMA、空 env runner 和 Top-K manager，最后交给 `accelerator.prepare` 包装分布式对象。

### 6.3 优化器和学习率

`Generator.get_optimizer` 按参数维度分组：`dim≥2` 的矩阵参数使用 AdamW weight decay `1e-4`；bias、norm scale 等一维参数 weight decay 为 0。默认 `lr=5e-4`，`betas=(0.95,0.999)`，CUDA 可用且 PyTorch 支持时启用 fused AdamW。

`model/common/lr_scheduler.py:get_scheduler` 是 diffusers scheduler factory 的薄封装。当前用 cosine schedule，先 warmup 2000 optimizer steps，再在预计总训练 steps 内 cosine 衰减。

每 batch：`compute_loss → loss/k → backward → clip global grad norm to 0.5 → optimizer.step/zero_grad → scheduler.step`。默认 `gradient_accumulate_every=1`。若改成大于 1，当前条件 `global_step % k == 0` 会在第一个 micro-batch 就 step，存在相位偏移，而且没有使用 Accelerate 的 `accumulate/no_sync`；不能把它视为已经严格实现的分布式梯度累积。

### 6.4 EMA

`model/diffusion/ema_model.py:EMAModel` 用 warmup decay：

\[
d_s=\operatorname{clip}\left(1-(1+s/\gamma)^{-p},d_{min},d_{max}\right),
\quad \theta_{ema}\leftarrow d_s\theta_{ema}+(1-d_s)\theta.
\]

BatchNorm 参数直接复制，其他可训练参数做 EMA。默认配置 `use_ema=False`，而主 YAML 没有提供 `cfg.ema` 节点；直接打开该开关会在实例化 `cfg.ema` 时失败，需先补配置。当前网络没有 BatchNorm，注释中的 BatchNorm 警告属于通用模板背景。

### 6.5 验证与采样指标

验证调用相同的 `compute_loss(training=False)`：上一动作 token 不 dropout，但仍重新随机 `t` 和 `ε`，所以单次 `val_loss` 是 Monte Carlo 估计，会有随机波动。Accelerate gather 各进程 batch loss 后直接求平均；若最后一个 batch 更小，这相当于 batch 等权而非样本等权。

默认 `val_every=10000`，而默认只训练 1000 epochs；由于 epoch 从 0 开始，通常只会在 epoch 0 验证一次。若要让验证损失有可解释的曲线或用于 Top-K，应显式改小 `training.val_every`。

`sample_every` 触发完整 DDIM 采样，并记录 latent MSE、逐元素 hand accuracy、整段 hand sequence accuracy、左右手 F1。训练 sample 使用“本 epoch 最后一个 batch”，不是固定样本。代码创建了 attention-map 输出路径变量，但没有执行 `pickle.dump`，因此当前不会真正导出 attention map 文件。

`GeneratorRunner.run` 当前只返回空字典，所以 `rollout_every` 并没有真实环境评估。

### 6.6 日志与结果导出

- Accelerate TensorBoard：输出到 `tb/train_generator/events...`；
- `JsonLogger`：把数值字段逐行写入 `logs.json.txt`，启动时截断不完整尾行，可续写；
- `save_loss_curve`：每个配置周期重写 `loss_curve.csv`，并用 matplotlib 导出 `loss_curve.png`；
- Hydra：`.hydra/config.yaml`、`hydra.yaml`、`overrides.yaml`；
- normalizer：主进程另行 pickle 为 `normalizer.pkl`，供多进程同步加载；policy checkpoint 内也包含其 state；
- 预测 API：以内存字典返回 latent、概率、logits、二值动作与可选 attention maps，本身不落盘。

### 6.7 checkpoint

`BaseWorkspace.save_checkpoint` 遍历 workspace 属性：凡有 `state_dict/load_state_dict` 的对象保存 state；`global_step`、`epoch` 和 `_output_dir` 以 dill bytes 保存；整个 `cfg` 一并保存。默认异步线程先递归把 state copy 到 CPU，再 `torch.save`。

`TopKCheckpointManager` 根据 `monitor_key` 保留最优 k 个文件并删除被淘汰文件。默认按 epoch 平均 `train_loss` 最小保留 5 个。若改为 `val_loss`，没有验证结果的 epoch 会安全跳过。每 100 epoch 还在 `epoch_checkpoints/epoch=N.ckpt` 保存一次，epoch 0 也保存。

默认 `resume=False` 导致 optimizer 不进入 checkpoint；`save_last_ckpt` 和 snapshot 也默认关闭。因此默认 Top-K checkpoint 可以加载模型用于推理，但不包含完整优化器状态，不能做到严格无缝续训。异步保存线程没有在训练结束显式 join；同一 epoch 连续发起多个保存时也会覆盖 `_saving_thread` 引用，这是长时间训练结束或快速退出时需要留意的可靠性边界。

## 7. 在线推理与闭环执行

### 7.1 `GeneratorWrapper`

定义：`wrapper/sugar_il_wrapper.py`。

- `load`：读取 checkpoint，依据其中 `cfg.policy` 重建网络并加载 `state_dicts['model']`；
- `observation_from_world`：把 11 个外部输入转为 float32/device，调用共享坐标变换，并确保所有模型字段带长度为 1 的时间轴；
- `predict_from_world`：组合上一方法和 `policy.predict_action`。

训练保存的 normalizer 是 `Generator` 子模块的一部分，所以加载整个 model state 后可直接推理。

### 7.2 Isaac Lab 脚本

`workspace/run_generator_isaaclab.py` 的 `main` 解析参数并启动 Isaac App；`_run` 必须在 app 启动后才导入 Isaac/SONIC 模块。它完成：

1. `resolve_motion_assets` 从 robot motion 推导 objects/USD/BPS/meta 配套路径并校验；
2. `_compose_sonic_config` 合成 SONIC 环境配置，固定 1 env、50 Hz、direct pre-quantization latent；
3. 加载 policy checkpoint，`_validate_generator` 检查 `[16,64,2]` 和仿真/动作数据均为 50 Hz；
4. reset 后验证起点位于首次接触前；
5. `_initial_atm_latent` 从 SONIC action transform module 取 reset 状态的 64D pre-FSQ latent；
6. `_initial_hand_primitive` 将 motion hand action 的符号转为 0/1；
7. `_fixed_object_goal` 从 motion 最后一帧读取固定目标物体世界位姿；
8. 循环构造观测、预测 16 帧、执行前 8 帧，并将预测更新为下一次的 `last_*`；
9. `binary_hand_to_sonic` 把 0/1 映射为环境需要的 -1/+1。

`_print_termination_terms` 在 episode 结束时输出各 termination term。`valid_start_upper_bound` 保证 reset 采样在接触前留 50 帧 margin。

### 7.3 一个明确的训练—推理差异

训练中 `target_hand_object_transform_6d` 来自 episode 最后一帧；但 `_generator_observation` 在线推理把当前 `hand_object` 同时传给 current 和 target 字段。除非当前手物体变换恰好就是目标变换，这构成分布偏移。代码没有从 goal motion 读取目标手物体变换。理解当前结果时应把它视为已知实现限制，而不是模型理论本身。

## 8. 逐文件、类与函数索引

以下“调用者”均指当前目录内可见代码。没有调用者表示通用/遗留设施，不代表外部工程绝不会使用。

### 8.1 核心网络与 policy

`model/diffusion/transformer_for_action_diffusion.py`

- `TimestepEmbedder`：由 `TransformerForActionDiffusion` 创建和调用；内部调用固定正余弦编码与 MLP；作用是把 scheduler 整数时间步变成条件 token。
- `RDTBlock`：由主干创建 12 个并循环调用；内部调用 3 个 RMSNorm、两种 MHA 和 FFN；作用是动作帧交互与条件融合。
- `TransformerForActionDiffusion`：由 `Generator.__init__` 创建，训练 `compute_loss` 和推理 `conditional_sample` 调用；调用上述 embedder/block；作用是连续 latent 去噪。
- `HandPrimitiveHead`：由 `Generator` 创建，训练和推理均调用；内部调用标准 Transformer decoder；作用是独立预测 16×2 手部 logits。

`model/encoder/generator_state_encoder.py`

- `_token_mlp`：仅由 encoder 构造函数调用，创建四条结构相同、参数独立的 MLP。
- `GeneratorStateObsEncoder`：由 Hydra 根据主配置创建，再传入 `Generator`；`forward` 由 loss/predict 调用；`_require_single_step` 校验输入；`output_shape` 由 `Generator.__init__` 查询隐藏维度。

`policy/generator.py`

- `Generator.__init__`：Hydra 调用，组合 encoder、DiT、hand head、scheduler、normalizer。
- `set_normalizer`：workspace 在 fit/load normalizer 后调用。
- `_normalize_obs`：loss 与 predict 内部调用；跳过二值上一手动作。
- `compute_loss`：训练与验证循环调用；实现加噪、两分支 forward 和联合损失。
- `forward`：返回单个总 loss，当前 workspace 没直接使用，便于标准 module 接口。
- `conditional_sample`：`predict_action` 调用；执行完整 16-step DDIM 链。
- `predict_action`：训练期采样评估、wrapper 和 Isaac 闭环调用；组织完整输出字典。
- `get_optimizer`：workspace 调用；构造带 decay/no-decay 参数组的 AdamW。

### 8.2 数据与几何

`dataset/generator_dataset.py`

- `_resolve_paths` → `_load_episode` → `GeneratorDataset.__init__`：完成文件发现和读取。
- `_load_episode`：校验 schema、shape、数值、四元数、BPS 和二值手动作。
- `GeneratorDataset.__init__`：Hydra 调用；完成 episode split 和窗口索引。
- `get_validation_dataset`：workspace 调用；复用预先划出的 validation episodes。
- `_pose`：`__getitem__` 调用；进而调用 `world_pose_to_body`。
- `__getitem__/__len__`：DataLoader 和 normalizer fit 调用；返回 nested tensor dict。
- `get_normalizer`：workspace 主进程调用；遍历训练窗口拟合 `LinearNormalizer`。

`dataset/base_dataset.py`

- `BaseLowdimDataset`：`GeneratorDataset` 的抽象式基类，也是 workspace 的类型约束；默认 validation 为空，其余关键接口待子类实现。
- `BaseImageDataset`：同类模板，当前 ObjectAware 链路未引用。

`common/geometry.py`

- `normalize_quaternion_wxyz` 被四元数转矩阵和求逆调用。
- `quaternion_to_matrix_wxyz` 被 rotation 6D、world→body、body→world 调用。
- `quaternion_multiply_wxyz`、`quaternion_inverse_wxyz` 被 `world_pose_to_body` 调用。
- `rotation_6d_columns` 被 `world_pose_to_body` 调用。
- `world_pose_to_body` 被 dataset 与 wrapper 共同调用，是 train/inference 坐标一致性的关键。
- `body_pose_to_world` 是位置变换的逆向诊断工具，当前目录内无调用者。

### 8.3 训练基础设施

`workspace/train_generator_workspace.py`

- `save_loss_curve` 由 epoch 末日志分支调用，导出 CSV/PNG。
- `TrainGeneratorWorkspace` 由 `main` 创建；继承 checkpoint 能力；`run` 调用几乎全部训练设施。
- 内部局部函数 `log_action_metrics` 由 train/val 完整采样分支调用。
- Hydra 装饰的 `main` 是训练 CLI 入口。

`workspace/base_workspace.py`

- `BaseWorkspace.output_dir` 从显式值或 Hydra runtime 取目录。
- `save_checkpoint/get_checkpoint_path/load_payload/load_checkpoint/create_from_checkpoint` 构成 state-dict checkpoint API；训练 workspace 调用保存及可选预加载。
- `save_snapshot/create_from_snapshot` 保存/读取整个 workspace 对象；默认关闭。
- `_copy_to_cpu` 被异步 checkpoint 递归调用。

`common/checkpoint_util.py`

- `TopKCheckpointManager.get_ckpt_path` 被训练 epoch 末调用；维护内存中的 path→metric 表、选择新路径并删除淘汰文件。

`common/json_logger.py`

- `JsonLogger` 作为训练循环 context manager；`start/stop` 管理文件，`log` 写 JSONL，`get_last_log` 提供最后一项副本。

`common/pytorch_util.py`

- `dict_apply` 被 workspace 做设备迁移，也被 normalizer 递归处理统计字典。

`model/common/module_attr_mixin.py`

- `ModuleAttrMixin` 被 encoder、Generator 和 DiT 继承；`device/dtype` 从第一个参数推断，采样据此创建 tensor。

`model/common/dict_of_tensor_mixin.py`

- `DictOfTensorMixin` 被 normalizer 继承；把任意嵌套参数字典注册成 `ParameterDict`，并在加载 state 时预建对应结构。

`model/common/normalizer.py`

- `LinearNormalizer` 管理多字段；`fit` 为每字段调用 `_fit`，`normalize/unnormalize` 调用 `_normalize`，`__getitem__` 返回单字段视图。
- `SingleFieldLinearNormalizer` 封装单字段拟合、手工/identity 创建和正逆变换；Generator 通过 `normalizer[key]` 使用。
- `_fit` 计算统计量及 affine 参数；`_normalize` 在保持原 shape 的前提下展平最后一维做正逆 affine。

`model/common/lr_scheduler.py`

- `get_scheduler` 由 workspace 调用；转发到 diffusers 的 scheduler factory，并检查 warmup/total steps 参数。

`model/diffusion/ema_model.py`

- `EMAModel.get_decay` 由 `step` 调用；`step` 由训练 batch 循环在可选 EMA 分支调用。默认关闭。

`env_runner/generator_runner.py`

- `GeneratorRunner` 由 Hydra/workspace 创建；`run` 被 rollout hook 调用但当前为空实现。

### 8.4 推理适配与执行

`wrapper/sugar_il_wrapper.py`

- `GeneratorWrapper.__init__` 固定 eval/device；`load` 是独立加载 API；`_time_axis` 被观测构造调用；`observation_from_world` 被 `predict_from_world` 和 Isaac helper 调用；`predict_from_world` 是最简外部接口。

`workspace/run_generator_isaaclab.py`

- `MotionAssets/stem` 表示一组配套 motion 资源；`resolve_motion_assets` 由 `main` 调用。
- `_parse_args/_resolve_from` 由 `main` 调用，处理 CLI 与启动目录相对路径。
- `_compose_sonic_config`、`_validate_generator`、`_initial_atm_latent`、`_initial_hand_primitive`、`_fixed_object_goal`、`_generator_observation` 均由 `_run` 调用。
- `binary_hand_to_sonic` 在执行每帧 action 时调用。
- `_print_termination_terms` 在 done 时调用。
- `_run` 由 `main` 调用，持有完整闭环。
- `main` 是在线推理 CLI 入口并负责 Isaac App 生命周期。

### 8.5 当前主线未使用的通用/遗留文件

`common/replay_buffer.py` 提供 NumPy/Zarr 两种 backend 的 episode replay buffer：顶层 `check_chunks_compatible`、`rechunk_recompress_array`、`get_optimal_chunks` 服务于 chunk/压缩；`ReplayBuffer` 的 create/copy/save 方法负责构建与持久化，data/meta/episode 属性和 mapping 方法负责访问，`add/drop/pop/extend/get_*` 负责 episode 增删与切片，`get/set_chunks`、`get/set_compressors` 负责存储布局。当前 pickle `GeneratorDataset` 没有导入它。

`common/streaming_replay_buffer.py` 的 `ZarrImageReference` 延迟索引图像；`StreamingReplayBuffer` 继承 `ReplayBuffer`，按需读取 Zarr、暴露 episode/step 切片和 mapping 接口。当前主线未使用。

`common/sampler.py` 的 `create_indices` 生成带前后 padding 的窗口索引，`get_val_mask/downsample_mask` 生成 episode mask，`SequenceSampler` 从 `ReplayBuffer` 采窗口。当前严格 pickle dataset 自己实现无 padding、无插值窗口，未调用这些设施。

`config/task/{CarryBox,KickBox,PickBottle,PushBox,SitChair,StandBottle}.yaml` 是遗留任务配置：它们传 `zarr_paths/n_obs_steps/pad_*`，而当前 `GeneratorDataset.__init__` 接受 `pickle_paths/horizon/val_ratio/seed`。因此不能直接与当前数据类组合运行。`ObjectAware.yaml` 才与当前实现一致。

### 8.6 包和生成物

- `config/train_generator_workspace.yaml`：Hydra 主配置；被训练入口的装饰器加载。
- `config/task/ObjectAware.yaml`：当前任务配置；由 `task=ObjectAware` 选择。
- `README.md`：给出训练/推理命令和数据语义摘要，不被代码调用。
- `pyproject.toml`：现代构建元数据与实际依赖；`setup.py` 是简化 setuptools 入口，自己的 `install_requires` 为空，安装行为取决于所用构建路径。
- `sugar_il.egg-info/`：安装生成的包元数据，不是运行逻辑。
- `__pycache__/`、`.pytest_cache/`：缓存，不是源码。
- `data/outputs/...`：一次实际训练生成的 Hydra 配置、日志、normalizer、TensorBoard event 和 checkpoints，可用于对照运行结果，不应反向当作源码。

## 9. 建议的阅读和调试顺序

1. 先在 `GeneratorDataset.__getitem__` 手算一个 `t` 的 9 个观测与两类 target，确认时间对齐。
2. 读 `GeneratorStateObsEncoder.forward`，在四次 concat 后打印 `[B,D_i]`，再确认 stack 为 `[B,4,256]`。
3. 单步读 `TransformerForActionDiffusion.forward`：重点区分动作序列位置、扩散时间步和条件 token 位置。
4. 展开一个 `RDTBlock`，检查 self map `[16,16]` 与 cross map `[16,5]` 的不同语义。
5. 回到 `Generator.compute_loss`，把 `x_0→x_t→ε̂` 与 DDPM 前向公式逐项对应。
6. 再读 `conditional_sample`，理解 scheduler 而非网络本身负责 `x_t→x_{t-1}` 数值更新。
7. 最后沿 workspace 的 batch loop 和 Isaac 的 8/16 receding-horizon loop 串起离线训练与在线控制。

在这个顺序下，DiT 不再是一个额外的黑盒概念：它只是把你已掌握的扩散去噪目标交给一个特定的 Transformer 条件网络来参数化；真正需要辨析的是 token 的物理语义、条件注入位置，以及网络输出和 scheduler 更新之间的职责边界。
