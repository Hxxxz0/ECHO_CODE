# Motion Safety Score (MSS)

## 概述

MSS 是一个评价动作生成模型安全性的离线指标，衡量生成的动作轨迹在 G1 机器人上安全执行的可能性。

- **输入**：测试集上 N 个生成的动作
- **输出**：一个标量 ∈ [0, 1]，越高越安全
- **特点**：所有阈值直接来源于训练代码，零超参数

## 数据格式

服务器端生成的动作数据使用 **Isaac 关节顺序**（左右交替排列）：

```python
# 服务器输出格式（Isaac顺序，29个关节）
# joint_pos: (T, 29) float32  — 关节角度（弧度）
# root_pos:  (T, 3)  float32  — 根部位置
# root_rot:  (T, 4)  float32  — 根部旋转（wxyz四元数）
# fps:       int               — 帧率

ISAAC_JOINT_ORDER = [
    # idx  关节名称                          所属部位
    #  0   left_hip_pitch_joint              左髋 pitch
    #  1   right_hip_pitch_joint             右髋 pitch
    #  2   waist_yaw_joint                   腰 yaw
    #  3   left_hip_roll_joint               左髋 roll
    #  4   right_hip_roll_joint              右髋 roll
    #  5   waist_roll_joint                  腰 roll
    #  6   left_hip_yaw_joint                左髋 yaw
    #  7   right_hip_yaw_joint               右髋 yaw
    #  8   waist_pitch_joint                 腰 pitch
    #  9   left_knee_joint                   左膝
    # 10   right_knee_joint                  右膝
    # 11   left_shoulder_pitch_joint         左肩 pitch
    # 12   right_shoulder_pitch_joint        右肩 pitch
    # 13   left_ankle_pitch_joint            左踝 pitch
    # 14   right_ankle_pitch_joint           右踝 pitch
    # 15   left_shoulder_roll_joint          左肩 roll
    # 16   right_shoulder_roll_joint         右肩 roll
    # 17   left_ankle_roll_joint             左踝 roll
    # 18   right_ankle_roll_joint            右踝 roll
    # 19   left_shoulder_yaw_joint           左肩 yaw
    # 20   right_shoulder_yaw_joint          右肩 yaw
    # 21   left_elbow_joint                  左肘
    # 22   right_elbow_joint                 右肘
    # 23   left_wrist_roll_joint             左腕 roll
    # 24   right_wrist_roll_joint            右腕 roll
    # 25   left_wrist_pitch_joint            左腕 pitch
    # 26   right_wrist_pitch_joint           右腕 pitch
    # 27   left_wrist_yaw_joint              左腕 yaw
    # 28   right_wrist_yaw_joint             右腕 yaw
]
```

## 安全约束来源

所有阈值直接来源于训练代码，与策略训练时完全一致：

| 约束 | 阈值 | 代码来源 |
|------|------|---------|
| 关节位置软限制 | 硬限制范围 × 0.9 | `locomotion.py:265-275`, `G1_tracking.yaml:143` |
| 关节速度限制 | ±10.0 rad/s | `simple_multimotion.py:115-116` |
| 关节加速度限制 | 100.0 rad/s² | `locomotion.py:261` |

### 关节位置硬限制（Isaac顺序）

以下限制来源于 `sim2real/assets/g1/g1.xml`，单位为弧度。
力矩限制来源于 `<actuator>` 段的 `ctrlrange`。

```
idx  关节名称                   下限(rad)   上限(rad)   力矩(Nm)
 0   left_hip_pitch_joint       -2.5307     2.8798      ±88
 1   right_hip_pitch_joint      -2.5307     2.8798      ±88
 2   waist_yaw_joint            -2.618      2.618       ±88
 3   left_hip_roll_joint        -0.5236     2.9671      ±88*
 4   right_hip_roll_joint       -2.9671     0.5236      ±88*
 5   waist_roll_joint           -0.52       0.52        ±50
 6   left_hip_yaw_joint         -2.7576     2.7576      ±88
 7   right_hip_yaw_joint        -2.7576     2.7576      ±88
 8   waist_pitch_joint          -0.52       0.52        ±50
 9   left_knee_joint            -0.087267   2.8798      ±139
10   right_knee_joint           -0.087267   2.8798      ±139
11   left_shoulder_pitch_joint  -3.0892     2.6704      ±25
12   right_shoulder_pitch_joint -3.0892     2.6704      ±25
13   left_ankle_pitch_joint     -0.87267    0.5236      ±50
14   right_ankle_pitch_joint    -0.87267    0.5236      ±50
15   left_shoulder_roll_joint   -1.5882     2.2515      ±25
16   right_shoulder_roll_joint  -2.2515     1.5882      ±25
17   left_ankle_roll_joint      -0.2618     0.2618      ±50
18   right_ankle_roll_joint     -0.2618     0.2618      ±50
19   left_shoulder_yaw_joint    -2.618      2.618       ±25
20   right_shoulder_yaw_joint   -2.618      2.618       ±25
21   left_elbow_joint           -1.0472     2.0944      ±25
22   right_elbow_joint          -1.0472     2.0944      ±25
23   left_wrist_roll_joint      -1.97222    1.97222     ±25
24   right_wrist_roll_joint     -1.97222    1.97222     ±25
25   left_wrist_pitch_joint     -1.61443    1.61443     ±5
26   right_wrist_pitch_joint    -1.61443    1.61443     ±5
27   left_wrist_yaw_joint       -1.61443    1.61443     ±5
28   right_wrist_yaw_joint      -1.61443    1.61443     ±5

* hip_roll: XML default 中 actuatorfrcrange 为 ±139，
  但 <motor> ctrlrange 限制为 ±88，以后者为准。
```

### 软限制计算

```
soft_factor = 0.9
range[j]      = upper[j] - lower[j]
center[j]     = (upper[j] + lower[j]) / 2
margin[j]     = range[j] × (1 - soft_factor) / 2    # 约 5% range
soft_lower[j] = center[j] - range[j] × soft_factor / 2
soft_upper[j] = center[j] + range[j] × soft_factor / 2
```

## 计算公式

### Step 1 — 差分求导

```python
velocity[t, j]     = (θ[t+1, j] - θ[t, j]) × fps       # shape: (T-1, 29)
acceleration[t, j] = (vel[t+1, j] - vel[t, j]) × fps    # shape: (T-2, 29)
```

### Step 2 — 归一化违规度

三个维度各自计算违规度 v ≥ 0（限制内为 0，超出后按比例增长）：

```python
# 位置：超出软限制的部分，以 margin 为单位归一化
v_pos[t,j] = max(0, soft_lower[j] - θ[t,j], θ[t,j] - soft_upper[j]) / margin[j]

# 速度：超出 10 rad/s 的部分
v_vel[t,j] = max(0, |velocity[t,j]| / 10.0 - 1.0)

# 加速度：超出 100 rad/s² 的部分
v_acc[t,j] = max(0, |acceleration[t,j]| / 100.0 - 1.0)
```

违规度含义：
- `v = 0` — 完全合规
- `v = 1` — 位置：刚好到硬限制 / 速度：达到 20 rad/s / 加速度：达到 200 rad/s²
- `v > 1` — 超出硬限制 / 严重超速 / 严重超加速

### Step 3 — 子分数（先聚合，后映射）

```python
s = 100.0  # MSS_SENSITIVITY

S_pos = exp( -s × mean(v_pos) )    # 先算平均违规度，再映射
S_vel = exp( -s × mean(v_vel) )
S_acc = exp( -s × mean(v_acc) )
```

**公式设计原理**：

传统做法 `mean(exp(-f(v)))` 有严重的"稀释问题"：当 99% 的 (t,j) 对合规（v=0，贡献 1.0），1% 的违规无论多严重都被淹没，分数永远在 0.99 附近。

改用 `exp(-s × mean(v))`：先聚合所有违规程度（包括 0），再映射到 [0,1]。这样 1% 的违规也能在分数上体现出来。

| mean(v) | S (s=100) | 含义 |
|---------|-----------|------|
| 0.000 | 1.000 | 完全合规 |
| 0.001 | 0.905 | 极轻微违规 |
| 0.005 | 0.607 | 少量违规 |
| 0.010 | 0.368 | 中度违规 |
| 0.050 | 0.007 | 严重违规 |

### Step 4 — 单动作安全分（加权几何平均）

```python
MSS_i = S_pos^0.5 × S_vel^0.3 × S_acc^0.2
```

权重分配：`位置(0.5) > 速度(0.3) > 加速度(0.2)`

- 权重和为 1.0，保证 MSS ∈ [0, 1]
- 几何平均具有 **短板效应**：任何维度严重违规都拉低总分

### Step 5 — 测试集总分

```python
MSS = (1/N) × Σ MSS_i
```

### 辅助指标：违规率 (Violation Rate)

为了提供更直观的物理意义，同时输出违规率：

```python
Rate = mean( v > 1e-4 )  # 统计违规帧/关节的比例
```

- **含义**：有多少比例的时步-关节对发生了违规。
- **单位**：百分比（%）。

## 输出解读

由于引入了敏感度系数，分数的绝对值会比原始版本低，且对违规更敏感。

| 分数区间 | 等级 | 含义 |
|----------|------|------|
| ≥ 0.90 | 优秀 | 极少违规，安全可靠 |
| 0.70 ~ 0.90 | 良好 | 存在少量轻微违规 |
| 0.50 ~ 0.70 | 警告 | 存在明显违规，需检查 |
| < 0.50 | 危险 | 严重违规，不可部署 |

同时应关注 **违规率**：
- **< 0.1%**：非常安全
- **> 1.0%**：存在系统性风险

## 设计依据

1. **阈值来源于训练代码**。soft_factor、速度clamp、加速度clamp 都是训练策略实际面对的约束。

2. **先聚合后映射 `exp(-s × mean(v))`**。传统 `mean(exp(-f(v)))` 在 99% 合规时分数被"淹没"在 1.0 附近，无法区分模型。改为先算平均违规度再映射，确保少量违规也能在分数上体现。`s=100` 使得 `mean(v)=0.005` 时分数为 `0.61`，具有良好区分度。

3. **几何平均的短板效应**。安全评价中，不能允许某个维度极差被另一个维度的高分掩盖。

4. **权重反映危险等级**。
   - 位置超限 (0.5)：物理硬约束，超过硬限制的关节角度完全不可执行
   - 速度超限 (0.3)：电机能力约束，超速导致追踪延迟和累积偏差
   - 加速度超限 (0.2)：力矩需求约束，超加速可提前规划缓解

---

**版本**: 1.2 (先聚合后映射，解决区分度问题)
**最后更新**: 2026-02-11
**数据来源**:
- `sim2real/assets/g1/g1.xml` — 关节硬限制、力矩限制
- `active_adaptation/envs/mdp/rewards/locomotion.py` — soft_factor, acc clamp
- `active_adaptation/utils/simple_multimotion.py` — 速度 clamp
- `cfg/task/G1/G1_tracking.yaml` — 训练配置
