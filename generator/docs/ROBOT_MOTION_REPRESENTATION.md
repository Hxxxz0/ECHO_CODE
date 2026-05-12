# Robot-Skeleton Motion Representation

## 概述

本文档详细描述了机器人骨架运动数据从**原始 NPZ 文件**到**38 维特征表示**的完整处理流程。该表示用于基于扩散模型的文本到运动生成任务。

---

## 1. 原始数据格式

### 1.1 NPZ 文件结构

每个运动序列存储为一个 `.npz` 文件，位于 `robot_humanml_data/npz/` 目录下。文件包含以下字段：

| 字段名 | 维度 | 说明 |
|--------|------|------|
| `joint_pos` | `(T, 29)` | **关节角度**，29 个关节的角度值（弧度） |
| `body_pos_w` | `(T, 30, 3)` | **刚体世界位置**，30 个刚体在世界坐标系下的 XYZ 位置 |
| `body_quat_w` | `(T, 30, 4)` | **刚体世界姿态**，30 个刚体在世界坐标系下的四元数（**wxyz 格式**） |
| `body_lin_vel_w` | `(T, 30, 3)` | 刚体线速度（存在但**不使用**，我们自己计算速度） |

其中：
- `T` 是运动序列的帧数
- 数据采样频率：**50 FPS**
- 坐标系约定：**Z-up**（XY 为地面，Z 为高度）
- 根节点索引：`root_idx = 0`（通常是骨盆/躯干）

### 1.2 坐标系说明

原始数据使用世界坐标系（右手系，Z-up）：
- **X 轴**：任意水平方向
- **Y 轴**：任意水平方向（垂直于 X）
- **Z 轴**：竖直向上（高度方向）
- **地面**：XY 平面

---

## 2. 数据处理流程

处理函数：`utils/robot_process.py::process_robot_npz()`

### 2.1 Step 1: 落地归一化 (Put on Floor)

**目的**：将角色放置在地面上，消除不同序列的高度偏移。

**操作**：
```python
floor_height = body_pos_w[:, :, 2].min()  # 全序列所有刚体的最小 Z 值
body_pos_w[:, :, 2] -= floor_height        # 所有 Z 坐标减去最小值
```

**结果**：序列中最低点的 Z 坐标变为 0。

---

### 2.2 Step 2: 根节点 XY 归零 (Root XY at Origin)

**目的**：将角色的初始位置平移到原点，消除全局位置偏移。

**操作**：
```python
root_pos_init = body_pos_w[0, root_idx, :]  # 第 0 帧根节点位置
root_xy_init = root_pos_init[[0, 1]]         # 提取 XY 分量
body_pos_w[:, :, 0] -= root_xy_init[0]       # 所有 X 坐标减去初始 X
body_pos_w[:, :, 1] -= root_xy_init[1]       # 所有 Y 坐标减去初始 Y
```

**结果**：第 0 帧根节点的 XY 坐标变为 (0, 0)。

---

### 2.3 Step 3: 朝向归一化 (Face +X Direction)

**目的**：将角色的初始朝向统一旋转到 +X 方向，消除不同序列的朝向差异。

#### 3.1 估计初始朝向

由于机器人身体结构未知，使用启发式方法估计：

```python
positions_init = body_pos_w[0]  # 第 0 帧所有刚体位置 (30, 3)

# 计算横向（左右）和前向的展开范围
y_range = positions_init[:, 1].max() - positions_init[:, 1].min()
x_range = positions_init[:, 0].max() - positions_init[:, 0].min()

# 判断哪个方向是横向（左右）
if y_range > x_range:
    across = np.array([0, 1, 0])  # Y 是横向，X 是前向
else:
    across = np.array([1, 0, 0])  # X 是横向，Y 是前向

# 计算前向：Z-up × across（叉乘）
z_up = np.array([0, 0, 1])
forward_init = np.cross(z_up, across)
forward_init = forward_init / np.linalg.norm(forward_init)
```

#### 3.2 计算旋转四元数

```python
target = np.array([1, 0, 0])  # 目标朝向：+X

# 计算从 forward_init 到 target 的旋转四元数
root_quat_init = qbetween_np(forward_init, target)  # (1, 4)
```

**特殊情况处理**：如果 `forward_init` 与 `target` 反向（点积 < -0.99），会翻转 `across` 以避免 180° 旋转导致的数值不稳定。

#### 3.3 应用旋转

**旋转所有刚体位置**：
```python
# 将旋转四元数广播到所有帧和刚体
root_quat_repeated = np.repeat(root_quat_init, T * 30, axis=0)  # (T*30, 4)
body_pos_w_flat = body_pos_w.reshape(-1, 3)                      # (T*30, 3)

# 使用四元数旋转位置向量
body_pos_w_flat = qrot_np(root_quat_repeated, body_pos_w_flat)
body_pos_w = body_pos_w_flat.reshape(T, 30, 3)
```

**旋转根节点姿态**：
```python
root_quat_original = body_quat_w[:, root_idx, :]  # (T, 4)

# 四元数乘法：新四元数 = 旋转四元数 × 原始四元数
root_quat_aligned = qmul_np(root_quat_init, root_quat_original)  # (T, 4)
```

**结果**：所有帧的位置和姿态都经过旋转，使得角色初始朝向为 +X 方向。

---

### 2.4 Step 4: 提取根节点速度 (Root Velocity)

**目的**：计算根节点在对齐后坐标系下的逐帧位移（速度形式）。

**操作**：
```python
root_pos_aligned = body_pos_w[:, root_idx, :]  # (T, 3) 对齐后的根节点位置

# 计算 XY 平面的逐帧位移
root_vel_xy_global = root_pos_aligned[1:, :2] - root_pos_aligned[:-1, :2]  # (T-1, 2)

# 构造长度为 T 的速度序列
root_vel_xy = np.zeros((T, 2), dtype=np.float32)
if T > 1:
    root_vel_xy[1:] = root_vel_xy_global  # 第 0 帧速度为 0
```

**约定**：`root_vel_xy[t]` 表示从第 `t-1` 帧到第 `t` 帧的位移。

**注意**：
- 这里的速度是在**对齐后的全局坐标系**下计算的（即朝向已经统一为 +X）
- 第 0 帧速度为 0（因为没有前一帧）
- 单位：米/帧（50 FPS 时，1 帧 = 0.02 秒）

---

### 2.5 Step 5: 提取根节点特征

#### 5.1 根节点高度
```python
root_z = body_pos_w[:, root_idx, 2:3]  # (T, 1) 根节点的 Z 坐标
```

#### 5.2 根节点旋转（6D 表示）

**为什么用 6D 而不是四元数？**
- 四元数有 4 维但实际自由度只有 3（SO(3)）
- 四元数不是连续的（存在双重覆盖：q 和 -q 表示同一旋转）
- 6D 表示是连续的，适合神经网络学习

**转换方法**：
```python
root_rot_6d = quat_wxyz_to_6d(root_quat_aligned)  # (T, 6)
```

**6D 表示定义**：取旋转矩阵的前两列（每列 3 维，共 6 维）
```python
def quat_wxyz_to_6d(quat):
    # quat: (T, 4) 格式 [w, x, y, z]
    # 转换为旋转矩阵 R: (T, 3, 3)
    # 取前两列：(T, 3, 2)
    # 展平为：(T, 6)
    rot_mat = quat_to_rotation_matrix(quat)
    rot6d = rot_mat[:, :, :2].reshape(T, 6)
    return rot6d
```

---

### 2.6 Step 6: 拼接 38D 特征

**最终特征**：
```python
features_38d = np.concatenate([
    joint_pos,      # (T, 29) 关节角度
    root_vel_xy,    # (T, 2)  根节点 XY 速度（逐帧位移）
    root_z,         # (T, 1)  根节点高度
    root_rot_6d     # (T, 6)  根节点旋转（6D 表示）
], axis=1)  # (T, 38)
```

**维度分解**：
- **[0:29]**   → 29 维关节角度
- **[29:31]**  → 2 维根节点 XY 速度
- **[31:32]**  → 1 维根节点高度
- **[32:38]**  → 6 维根节点旋转

---

## 3. 最终表示说明

### 3.1 表示形式

每个运动序列表示为：
- **形状**：`(T, 38)`
- **数据类型**：`float32`
- **坐标系**：对齐后的全局坐标系（朝向 +X，根节点初始 XY 在原点，Z 在地面）

### 3.2 特征解释

| 索引范围 | 维度 | 特征名称 | 说明 |
|---------|------|----------|------|
| 0-28 | 29 | `joint_pos` | 关节角度（弧度），直接从原始数据复制 |
| 29-30 | 2 | `root_vel_xy` | 根节点 XY 平面速度（米/帧），在对齐全局系下 |
| 31 | 1 | `root_z` | 根节点高度（米），相对于地面 |
| 32-37 | 6 | `root_rot_6d` | 根节点旋转（6D 连续表示） |

### 3.3 设计理由

#### 为什么使用速度而不是位置？
1. **平移不变性**：速度不受全局位置影响，模型更容易学习
2. **序列长度适应**：速度可以积分到任意长度，位置会无限增长
3. **物理意义**：速度更符合运动控制的直觉

#### 为什么用 6D 旋转？
1. **连续性**：6D 表示是连续的，没有奇异点
2. **冗余度**：6 维表示 3 自由度的旋转，提供了冗余信息
3. **网络友好**：比四元数更容易被神经网络学习

#### 为什么保留关节角度？
- 关节角度直接控制机器人的局部形态，是运动的核心信息
- 关节角度与根节点运动解耦，分别建模更清晰

---

## 4. 数据集统计

### 4.1 数据范围

- **序列长度**：100 - 490 帧（2 秒 - 9.8 秒，50 FPS）
- **最大序列长度**：490 帧（用于训练时的填充/裁剪）
- **最小序列长度**：100 帧（过滤掉过短的序列）

### 4.2 归一化

**训练时**：使用 Z-score 归一化
```python
motion_normalized = (motion - mean) / std
```

- **mean**：`robot_humanml_data/Mean_38d.npy` (shape: 38)
- **std**：`robot_humanml_data/Std_38d.npy` (shape: 38)

**反归一化**（推理时）：
```python
motion_original = motion_normalized * std + mean
```

---

## 5. 从 38D 特征恢复运动

### 5.1 根节点 XY 位置恢复

**方法**：积分速度
```python
def recover_root_xy_from_velocity(root_vel_xy):
    """
    Args:
        root_vel_xy: (T, 2) 逐帧位移
    Returns:
        root_xy_pos: (T, 2) 全局 XY 位置
    """
    T = root_vel_xy.shape[0]
    root_xy_pos = np.zeros((T, 2), dtype=np.float32)
    
    # 第 0 帧在原点
    root_xy_pos[0] = np.array([0, 0])
    
    # 累积求和
    for t in range(1, T):
        root_xy_pos[t] = root_xy_pos[t-1] + root_vel_xy[t]
    
    return root_xy_pos
```

### 5.2 根节点旋转恢复

**方法**：6D 转回四元数
```python
from utils.rotation_utils import rot6d_to_quat_wxyz

root_rot_6d = features_38d[:, 32:38]  # (T, 6)
root_quat = rot6d_to_quat_wxyz(root_rot_6d)  # (T, 4)
```

### 5.3 完整刚体位置恢复

需要结合：
1. 关节角度 → 正向运动学（FK）→ 局部刚体位置
2. 根节点位置（XY + Z）+ 旋转 → 全局变换
3. 应用全局变换到局部刚体位置

---

## 6. 与 HumanML3D 的对比

| 特性 | HumanML3D | 本项目（Robot） |
|------|-----------|-----------------|
| 坐标系 | Y-up | **Z-up** |
| 朝向 | +Z | **+X** |
| FPS | 20 | **50** |
| 关节数 | 22 | **30** |
| 特征维度 | 263 | **38** |
| 速度定义 | 相同 | 相同（对齐全局系下的逐帧位移） |
| 旋转表示 | 6D | 6D |

**核心区别**：
- **坐标系差异**：Z-up vs Y-up，导致所有几何计算的轴不同
- **特征维度差异**：HumanML3D 包含更多局部关节位置和速度，本项目只用关节角度
- **朝向差异**：+X vs +Z，影响前向方向的定义

---

## 7. 代码实现位置

- **主处理函数**：`utils/robot_process.py::process_robot_npz()`
- **数据集类**：`datasets/robot_dataset.py::RobotMotionDataset`
- **旋转工具**：
  - `utils/quaternion.py`：四元数运算（qbetween, qrot, qmul）
  - `utils/rotation_utils.py`：6D 旋转转换

---

## 8. 使用示例

### 8.1 加载和处理单个样本

```python
import numpy as np
from utils.robot_process import process_robot_npz

# 加载 NPZ
npz_data = np.load('robot_humanml_data/npz/000000.npz')

# 处理成 38D 特征
features_38d = process_robot_npz(npz_data, root_idx=0)

print(f"Shape: {features_38d.shape}")  # (T, 38)
print(f"Joint angles: {features_38d[0, :29]}")
print(f"Root vel XY: {features_38d[0, 29:31]}")
print(f"Root Z: {features_38d[0, 31]}")
print(f"Root rot 6D: {features_38d[0, 32:38]}")
```

### 8.2 使用数据集类

```python
from datasets.robot_dataset import RobotMotionDataset

# 创建数据集
dataset = RobotMotionDataset(opt, split='train', mode='train')

# 获取一个样本
caption, motion, length = dataset[0]

print(f"Caption: {caption}")
print(f"Motion shape: {motion.shape}")  # (490, 38) 已归一化和填充
print(f"Actual length: {length}")
```

---

## 9. 注意事项

### 9.1 坐标系一致性
- 所有处理必须使用 **Z-up** 坐标系
- 朝向统一为 **+X** 方向
- 地面是 **XY 平面**（Z=0）

### 9.2 数据质量
- 确保原始 NPZ 文件的 Z 轴确实是竖直方向
- 检查关节角度范围是否合理（弧度制）
- 验证刚体数量是否为 30

### 9.3 速度约定
- 速度是在**对齐后的全局坐标系**下计算的
- 第 0 帧速度为 0
- 速度单位：米/帧（50 FPS）

### 9.4 归一化
- 训练时必须使用相同的 mean/std
- 推理时必须使用训练时的 mean/std 进行归一化
- 生成后必须反归一化

---

## 10. 总结

**原始数据 → 38D 特征的完整流程**：

```
NPZ 文件
  ├─ joint_pos (T, 29)
  ├─ body_pos_w (T, 30, 3)
  └─ body_quat_w (T, 30, 4)
       ↓
  [Step 1] 落地归一化
       ↓
  [Step 2] 根节点 XY 归零
       ↓
  [Step 3] 朝向对齐到 +X
       ↓
  [Step 4] 计算根节点速度
       ↓
  [Step 5] 提取根节点特征 (Z, 旋转 6D)
       ↓
  [Step 6] 拼接 38D 特征
       ↓
38D 特征向量 (T, 38)
  ├─ joint_pos [0:29]   (29D)
  ├─ root_vel_xy [29:31] (2D)
  ├─ root_z [31:32]      (1D)
  └─ root_rot_6d [32:38] (6D)
```

**关键设计思想**：
1. **全局归一化**：消除不同序列的位置/朝向差异
2. **速度表示**：提供平移不变性
3. **连续旋转**：6D 表示适合神经网络
4. **紧凑表示**：38 维远小于 HumanML3D 的 263 维，更高效

---

## 参考资料

- HumanML3D: https://github.com/EricGuo5513/HumanML3D
- 6D Rotation Representation: Zhou et al., "On the Continuity of Rotation Representations in Neural Networks" (CVPR 2019)
- Quaternion Operations: https://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/

---

**文档版本**: v1.0  
**最后更新**: 2026-02-11  
**作者**: StableRofusion 项目组
