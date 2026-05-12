"""Motion Safety Score (MSS) -- offline safety evaluation for generated robot motions.

Evaluates the likelihood that a generated motion trajectory can be safely executed
on the Unitree G1 robot, based on joint position/velocity/acceleration limits.

Output: scalar in [0, 1], higher is safer.
All thresholds are derived from training code -- zero hyperparameters.
"""
import numpy as np
from utils.rotation_utils import rot6d_to_quat_wxyz
from utils.robot_process import recover_root_xy_from_velocity


# Isaac joint order hard limits (29 joints)
# Source: sim2real/assets/g1/g1.xml
# Format: [lower(rad), upper(rad)]
ISAAC_JOINT_LIMITS = np.array([
    # idx  joint_name               lower(rad)   upper(rad)
    [-2.5307, 2.8798],      # 0   left_hip_pitch_joint
    [-2.5307, 2.8798],      # 1   right_hip_pitch_joint
    [-2.618, 2.618],        # 2   waist_yaw_joint
    [-0.5236, 2.9671],      # 3   left_hip_roll_joint
    [-2.9671, 0.5236],      # 4   right_hip_roll_joint
    [-0.52, 0.52],          # 5   waist_roll_joint
    [-2.7576, 2.7576],      # 6   left_hip_yaw_joint
    [-2.7576, 2.7576],      # 7   right_hip_yaw_joint
    [-0.52, 0.52],          # 8   waist_pitch_joint
    [-0.087267, 2.8798],    # 9   left_knee_joint
    [-0.087267, 2.8798],    # 10  right_knee_joint
    [-3.0892, 2.6704],      # 11  left_shoulder_pitch_joint
    [-3.0892, 2.6704],      # 12  right_shoulder_pitch_joint
    [-0.87267, 0.5236],     # 13  left_ankle_pitch_joint
    [-0.87267, 0.5236],     # 14  right_ankle_pitch_joint
    [-1.5882, 2.2515],      # 15  left_shoulder_roll_joint
    [-2.2515, 1.5882],      # 16  right_shoulder_roll_joint
    [-0.2618, 0.2618],      # 17  left_ankle_roll_joint
    [-0.2618, 0.2618],      # 18  right_ankle_roll_joint
    [-2.618, 2.618],        # 19  left_shoulder_yaw_joint
    [-2.618, 2.618],        # 20  right_shoulder_yaw_joint
    [-1.0472, 2.0944],      # 21  left_elbow_joint
    [-1.0472, 2.0944],      # 22  right_elbow_joint
    [-1.97222, 1.97222],    # 23  left_wrist_roll_joint
    [-1.97222, 1.97222],    # 24  right_wrist_roll_joint
    [-1.61443, 1.61443],    # 25  left_wrist_pitch_joint
    [-1.61443, 1.61443],    # 26  right_wrist_pitch_joint
    [-1.61443, 1.61443],    # 27  left_wrist_yaw_joint
    [-1.61443, 1.61443],    # 28  right_wrist_yaw_joint
], dtype=np.float32)

# Safety constraint parameters (derived from training code)
SOFT_FACTOR = 0.9           # locomotion.py:265-275
VEL_LIMIT = 10.0            # rad/s, simple_multimotion.py:115-116
ACC_LIMIT = 100.0           # rad/s², locomotion.py:261

# 几何平均权重（反映危险等级）
WEIGHT_POS = 0.5            # 位置超限最危险
WEIGHT_VEL = 0.3            # 速度超限次之
WEIGHT_ACC = 0.2            # 加速度超限可提前规划缓解

# 敏感度系数：控制违规对分数的惩罚力度
# 公式：S = exp(-s × mean(v))
# s=100 时：mean(v)=0.001 → S=0.90, mean(v)=0.005 → S=0.61, mean(v)=0.01 → S=0.37
MSS_SENSITIVITY = 100.0


def motion_38d_to_isaac_format(features_38d, mean, std):
    """
    将38维特征转换为Isaac格式（用于MSS计算）
    
    Args:
        features_38d: (T, 38) - 归一化的38维特征
        mean: (38,) - 数据集均值
        std: (38,) - 数据集标准差
    
    Returns:
        joint_pos: (T, 29) - 关节角度（弧度，Isaac顺序）
        root_pos: (T, 3) - 根部位置 [x, y, z]
        root_rot: (T, 4) - 根部旋转（wxyz四元数）
    """
    # Step 1: 反归一化
    features_real = features_38d * std + mean
    
    # Step 2: 提取各部分
    joint_pos = features_real[:, :29]           # (T, 29) 关节角度
    root_vel_xy = features_real[:, 29:31]       # (T, 2) 根速度XY
    root_z = features_real[:, 31:32]            # (T, 1) 根高度Z
    root_rot_6d = features_real[:, 32:38]       # (T, 6) 根旋转6D
    
    # Step 3: 转换root_rot 6D -> wxyz四元数
    root_rot = rot6d_to_quat_wxyz(root_rot_6d)  # (T, 4)
    
    # Step 4: 积分root_vel_xy恢复root_pos_xy
    root_xy = recover_root_xy_from_velocity(root_vel_xy, root_rot)  # (T, 2)
    
    # Step 5: 组合root_pos
    root_pos = np.concatenate([root_xy, root_z], axis=1)  # (T, 3)
    
    return joint_pos, root_pos, root_rot


def calculate_motion_safety_score(joint_pos, root_pos=None, root_rot=None, fps=50):
    """
    计算单个动作序列的Motion Safety Score
    
    基于关节位置、速度、加速度的综合安全性评估。
    使用exp(-v²)核函数和几何平均，体现"短板效应"。
    
    Args:
        joint_pos: (T, 29) - 关节角度序列（弧度）
        root_pos: (T, 3) - 根部位置（暂未使用）
        root_rot: (T, 4) - 根部旋转（暂未使用）
        fps: int - 帧率，用于计算速度和加速度
    
    Returns:
        dict: {
            'mss': float - 总分 ∈ [0, 1]
            'pos_score': float - 位置分 ∈ [0, 1]
            'vel_score': float - 速度分 ∈ [0, 1]
            'acc_score': float - 加速度分 ∈ [0, 1]
        }
    """
    T = joint_pos.shape[0]
    
    if T < 3:
        # 序列太短，无法计算加速度
        return {
            'mss': 1.0,
            'pos_score': 1.0,
            'vel_score': 1.0,
            'acc_score': 1.0
        }
    
    # ============ Step 1: 差分求导 ============
    velocity = (joint_pos[1:] - joint_pos[:-1]) * fps      # (T-1, 29)
    acceleration = (velocity[1:] - velocity[:-1]) * fps    # (T-2, 29)
    
    # ============ Step 2: 计算软限制 ============
    lower = ISAAC_JOINT_LIMITS[:, 0]  # (29,)
    upper = ISAAC_JOINT_LIMITS[:, 1]  # (29,)
    
    joint_range = upper - lower
    center = (upper + lower) / 2
    margin = joint_range * (1 - SOFT_FACTOR) / 2  # 约5%范围作为缓冲区
    
    soft_lower = center - joint_range * SOFT_FACTOR / 2
    soft_upper = center + joint_range * SOFT_FACTOR / 2
    
    # ============ Step 3: 计算归一化违规度 ============
    
    # 位置违规度：超出软限制的部分，以margin为单位归一化
    # v_pos[t,j] = max(0, soft_lower[j] - θ[t,j], θ[t,j] - soft_upper[j]) / margin[j]
    v_pos_lower = np.maximum(0, soft_lower - joint_pos) / (margin + 1e-8)  # (T, 29)
    v_pos_upper = np.maximum(0, joint_pos - soft_upper) / (margin + 1e-8)  # (T, 29)
    v_pos = np.maximum(v_pos_lower, v_pos_upper)  # (T, 29)
    
    # 速度违规度：超出10 rad/s的部分
    # v_vel[t,j] = max(0, |velocity[t,j]| / 10.0 - 1.0)
    v_vel = np.maximum(0, np.abs(velocity) / VEL_LIMIT - 1.0)  # (T-1, 29)
    
    # 加速度违规度：超出100 rad/s²的部分
    # v_acc[t,j] = max(0, |acceleration[t,j]| / 100.0 - 1.0)
    v_acc = np.maximum(0, np.abs(acceleration) / ACC_LIMIT - 1.0)  # (T-2, 29)
    
    # ============ Step 4: 计算子分数 ============
    # 核心改动：先算平均违规度，再用 exp 映射到 [0, 1]
    #
    # 旧公式（无区分度）: S = mean(exp(-f(v)))
    #   问题：99% 的 v=0 贡献 1.0，把 1% 的违规"淹没"了
    #
    # 新公式（有区分度）: S = exp(-s × mean(v))
    #   先聚合违规程度（包含 0），再映射到分数
    #   即使只有 1% 违规，其严重程度也不会被稀释
    
    S_pos = np.exp(-MSS_SENSITIVITY * np.mean(v_pos))
    S_vel = np.exp(-MSS_SENSITIVITY * np.mean(v_vel))
    S_acc = np.exp(-MSS_SENSITIVITY * np.mean(v_acc))
    
    # ============ Step 5: 几何平均得到总分 ============
    # MSS = S_pos^0.5 × S_vel^0.3 × S_acc^0.2
    MSS = (S_pos ** WEIGHT_POS) * (S_vel ** WEIGHT_VEL) * (S_acc ** WEIGHT_ACC)
    
    # 违规率（辅助指标）：统计有多少比例的帧/关节发生了违规
    R_pos = np.mean(v_pos > 1e-4)
    R_vel = np.mean(v_vel > 1e-4)
    R_acc = np.mean(v_acc > 1e-4)
    
    return {
        'mss': float(MSS),
        'pos_score': float(S_pos),
        'vel_score': float(S_vel),
        'acc_score': float(S_acc),
        'pos_rate': float(R_pos),  # 违规率
        'vel_rate': float(R_vel),
        'acc_rate': float(R_acc)
    }


def evaluate_mss_batch(motions, m_lens, mean, std, fps=50):
    """
    批量评估Motion Safety Score
    
    Args:
        motions: (B, T, 38) - 批量归一化的38维特征
        m_lens: (B,) - 每个序列的实际长度
        mean: (38,) - 数据集均值
        std: (38,) - 数据集标准差
        fps: int - 帧率
    
    Returns:
        dict: 批次平均的MSS及子分数
    """
    batch_size = motions.shape[0]
    all_mss = []
    all_pos = []
    all_vel = []
    all_acc = []
    all_r_pos = []
    all_r_vel = []
    all_r_acc = []
    
    for i in range(batch_size):
        # 提取有效长度的序列
        motion = motions[i, :m_lens[i], :]  # (T_i, 38)
        
        # 转换为Isaac格式
        joint_pos, root_pos, root_rot = motion_38d_to_isaac_format(motion, mean, std)
        
        # 计算MSS
        scores = calculate_motion_safety_score(joint_pos, root_pos, root_rot, fps)
        
        all_mss.append(scores['mss'])
        all_pos.append(scores['pos_score'])
        all_vel.append(scores['vel_score'])
        all_acc.append(scores['acc_score'])
        all_r_pos.append(scores['pos_rate'])
        all_r_vel.append(scores['vel_rate'])
        all_r_acc.append(scores['acc_rate'])
    
    # 返回批次平均
    return {
        'mss': np.mean(all_mss),
        'pos_score': np.mean(all_pos),
        'vel_score': np.mean(all_vel),
        'acc_score': np.mean(all_acc),
        'pos_rate': np.mean(all_r_pos),
        'vel_rate': np.mean(all_r_vel),
        'acc_rate': np.mean(all_r_acc)
    }


def average_mss_results(mss_scores):
    """
    聚合多个批次的MSS结果
    
    Args:
        mss_scores: List[dict] - 每个批次的MSS字典
    
    Returns:
        dict: 所有批次的平均MSS
    """
    all_mss = [s['mss'] for s in mss_scores]
    all_pos = [s['pos_score'] for s in mss_scores]
    all_vel = [s['vel_score'] for s in mss_scores]
    all_acc = [s['acc_score'] for s in mss_scores]
    all_r_pos = [s.get('pos_rate', 0.0) for s in mss_scores]
    all_r_vel = [s.get('vel_rate', 0.0) for s in mss_scores]
    all_r_acc = [s.get('acc_rate', 0.0) for s in mss_scores]
    
    return {
        'mss': np.mean(all_mss),
        'pos_score': np.mean(all_pos),
        'vel_score': np.mean(all_vel),
        'acc_score': np.mean(all_acc),
        'pos_rate': np.mean(all_r_pos),
        'vel_rate': np.mean(all_r_vel),
        'acc_rate': np.mean(all_r_acc)
    }


if __name__ == "__main__":
    """测试MSS计算"""
    print("Testing Motion Safety Score calculation...")
    
    # 创建测试数据
    T = 100  # 100帧
    joint_pos_test = np.zeros((T, 29), dtype=np.float32)
    
    # 测试1：完全合规的动作（所有关节在中心位置）
    print("\n[Test 1] 完全合规的动作")
    lower = ISAAC_JOINT_LIMITS[:, 0]
    upper = ISAAC_JOINT_LIMITS[:, 1]
    center = (lower + upper) / 2
    joint_pos_test[:] = center[np.newaxis, :]
    
    scores = calculate_motion_safety_score(joint_pos_test, fps=50)
    print(f"  MSS: {scores['mss']:.4f}")
    print(f"  Pos: {scores['pos_score']:.4f} (Rate: {scores['pos_rate']:.4f})")
    print(f"  Vel: {scores['vel_score']:.4f} (Rate: {scores['vel_rate']:.4f})")
    print(f"  Acc: {scores['acc_score']:.4f} (Rate: {scores['acc_rate']:.4f})")
    print(f"  预期：MSS接近1.0（完全安全）")
    
    # 测试2：有轻微超限的动作
    print("\n[Test 2] 轻微超限的动作")
    joint_pos_test[:] = center[np.newaxis, :]
    # 让第0个关节到达软限制边缘
    soft_upper_0 = center[0] + (upper[0] - lower[0]) * SOFT_FACTOR / 2
    joint_pos_test[:, 0] = soft_upper_0
    
    scores = calculate_motion_safety_score(joint_pos_test, fps=50)
    print(f"  MSS: {scores['mss']:.4f}")
    print(f"  Pos: {scores['pos_score']:.4f} (Rate: {scores['pos_rate']:.4f})")
    print(f"  Vel: {scores['vel_score']:.4f} (Rate: {scores['vel_rate']:.4f})")
    print(f"  Acc: {scores['acc_score']:.4f} (Rate: {scores['acc_rate']:.4f})")
    print(f"  预期：MSS略低于1.0（轻微违规）")
    
    # 测试3：速度超限的动作
    print("\n[Test 3] 速度超限的动作")
    joint_pos_test[:] = center[np.newaxis, :]
    # 创建快速变化的动作（速度超限）
    for t in range(T):
        joint_pos_test[t, 0] = center[0] + 0.5 * np.sin(2 * np.pi * t / 10)
    
    scores = calculate_motion_safety_score(joint_pos_test, fps=50)
    print(f"  MSS: {scores['mss']:.4f}")
    print(f"  Pos: {scores['pos_score']:.4f} (Rate: {scores['pos_rate']:.4f})")
    print(f"  Vel: {scores['vel_score']:.4f} (Rate: {scores['vel_rate']:.4f})")
    print(f"  Acc: {scores['acc_score']:.4f} (Rate: {scores['acc_rate']:.4f})")
    print(f"  预期：Vel分数降低（速度违规）")
    
    print("\n✓ MSS计算测试完成！")
