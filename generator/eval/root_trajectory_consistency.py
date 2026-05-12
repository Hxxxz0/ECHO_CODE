"""Root Trajectory Consistency (RTC) -- evaluates root trajectory alignment.

Measures whether the generated root path matches the ground-truth path in shape
and spatial extent, using arc-length parameterization to eliminate speed differences.

Output: scalar in [0, 1], higher is more consistent.
- Shape Score (weight 0.7) + Extent Score (weight 0.3)
- Zero hyperparameters (sigma values based on documented standards)
"""
import sys
import os

# Add project root to path for imports
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from utils.rotation_utils import rot6d_to_quat_wxyz
from utils.robot_process import recover_root_xy_from_velocity


# RTC parameters (slightly relaxed for readability; original doc values: σ_shape=0.2, σ_extent=0.5)
SIGMA_SHAPE = 0.35     # shape score sensitivity (larger = more permissive)
SIGMA_EXTENT = 0.8     # extent score sensitivity (larger = more permissive)
WEIGHT_SHAPE = 0.7     # shape weight
WEIGHT_EXTENT = 0.3    # extent weight
NUM_SAMPLES = 50       # arc-length resample points
EPSILON = 1e-6         # 数值稳定性阈值


def compute_arc_length(trajectory):
    """
    计算轨迹的累积弧长
    
    Args:
        trajectory: (T, 2) - XY 轨迹
    
    Returns:
        arc_lengths: (T,) - 累积弧长，从0开始
        total_length: float - 总弧长
    """
    T = trajectory.shape[0]
    
    if T < 2:
        return np.zeros(T, dtype=np.float32), 0.0
    
    # 计算逐段距离
    segment_lengths = np.linalg.norm(trajectory[1:] - trajectory[:-1], axis=1)  # (T-1,)
    
    # 累积弧长
    arc_lengths = np.zeros(T, dtype=np.float32)
    arc_lengths[1:] = np.cumsum(segment_lengths)
    
    total_length = arc_lengths[-1]
    
    return arc_lengths, total_length


def arc_length_parameterization(trajectory, num_samples=NUM_SAMPLES):
    """
    对轨迹进行弧长参数化和重采样
    
    沿轨迹均匀采样 num_samples 个点（按弧长比例）
    
    Args:
        trajectory: (T, 2) - 原始轨迹
        num_samples: int - 重采样点数
    
    Returns:
        resampled_trajectory: (num_samples, 2) - 重采样后的轨迹
        total_length: float - 总弧长
    """
    T = trajectory.shape[0]
    
    # 处理极短序列
    if T < 3:
        num_samples = min(num_samples, T)
    
    # 计算累积弧长
    arc_lengths, total_length = compute_arc_length(trajectory)
    
    # 处理静态情况（总弧长接近0）
    if total_length < EPSILON:
        # 返回起点重复 num_samples 次
        return np.tile(trajectory[0:1], (num_samples, 1)), 0.0
    
    # 归一化弧长到 [0, 1]
    arc_lengths_norm = arc_lengths / total_length
    
    # 生成均匀采样点（弧长比例）
    sample_fractions = np.linspace(0, 1, num_samples, dtype=np.float32)
    
    # 插值得到重采样轨迹
    resampled_x = np.interp(sample_fractions, arc_lengths_norm, trajectory[:, 0])
    resampled_y = np.interp(sample_fractions, arc_lengths_norm, trajectory[:, 1])
    resampled_trajectory = np.stack([resampled_x, resampled_y], axis=1)  # (num_samples, 2)
    
    return resampled_trajectory, total_length


def calculate_shape_score(traj_gen, traj_gt, sigma_shape=SIGMA_SHAPE, num_samples=NUM_SAMPLES):
    """
    计算 Shape Score - 路径形状一致性
    
    使用弧长参数化消除速度差异，只比较路径形状。
    
    Args:
        traj_gen: (T_gen, 2) - 生成轨迹
        traj_gt: (T_gt, 2) - GT 轨迹
        sigma_shape: float - 敏感度系数
        num_samples: int - 重采样点数
    
    Returns:
        shape_score: float ∈ [0, 1]
    """
    # 弧长参数化重采样
    Q_gen, L_gen = arc_length_parameterization(traj_gen, num_samples)
    Q_gt, L_gt = arc_length_parameterization(traj_gt, num_samples)
    
    # 处理双方都静态的情况
    if L_gen < EPSILON and L_gt < EPSILON:
        return 1.0  # 都不动 = 完美一致
    
    # 逐点计算欧式距离
    distances = np.linalg.norm(Q_gen - Q_gt, axis=1)  # (num_samples,)
    mean_distance = np.mean(distances)
    
    # 归一化（以两条轨迹的平均总弧长为基准）
    avg_length = (L_gen + L_gt) / 2.0
    if avg_length < EPSILON:
        d_norm = 0.0
    else:
        d_norm = mean_distance / avg_length
    
    # 映射为分数: S_shape = exp(-(d_norm / σ)²)
    shape_score = np.exp(-((d_norm / sigma_shape) ** 2))
    
    return float(shape_score)


def calculate_extent_score(traj_gen, traj_gt, sigma_extent=SIGMA_EXTENT):
    """
    计算 Extent Score - 覆盖范围一致性
    
    检查生成轨迹是否走完了应有的距离。
    
    Args:
        traj_gen: (T_gen, 2) - 生成轨迹
        traj_gt: (T_gt, 2) - GT 轨迹
        sigma_extent: float - 敏感度系数
    
    Returns:
        extent_score: float ∈ [0, 1]
    """
    # 计算总弧长
    _, L_gen = compute_arc_length(traj_gen)
    _, L_gt = compute_arc_length(traj_gt)
    
    # 处理双方都静态的情况
    if L_gen < EPSILON and L_gt < EPSILON:
        return 1.0  # 都不动 = 完美一致
    
    # 计算弧长偏差（相对于 GT）
    if L_gt < EPSILON:
        # GT 静态但 Gen 不静态 = 不一致
        delta = 1.0 if L_gen > EPSILON else 0.0
    else:
        delta = abs(L_gen - L_gt) / L_gt
    
    # 映射为分数: S_extent = exp(-(Δ / σ)²)
    extent_score = np.exp(-((delta / sigma_extent) ** 2))
    
    return float(extent_score)


def calculate_rtc_single(traj_gen, traj_gt):
    """
    计算单个样本的 Root Trajectory Consistency
    
    RTC = S_shape^0.7 × S_extent^0.3
    
    Args:
        traj_gen: (T_gen, 2) - 生成轨迹
        traj_gt: (T_gt, 2) - GT 轨迹
    
    Returns:
        dict: {
            'rtc': float - 总分 ∈ [0, 1]
            'shape_score': float - 形状分 ∈ [0, 1]
            'extent_score': float - 覆盖分 ∈ [0, 1]
        }
    """
    # 计算两个子分数
    shape_score = calculate_shape_score(traj_gen, traj_gt)
    extent_score = calculate_extent_score(traj_gen, traj_gt)
    
    # 几何平均（加权）
    rtc = (shape_score ** WEIGHT_SHAPE) * (extent_score ** WEIGHT_EXTENT)
    
    return {
        'rtc': float(rtc),
        'shape_score': float(shape_score),
        'extent_score': float(extent_score)
    }


def extract_root_trajectory_from_38d(features_38d, mean, std):
    """
    从归一化的38维特征提取根节点XY轨迹
    
    38维特征格式：[joint_pos(29), root_vel_xy(2), root_z(1), root_rot_6d(6)]
    
    Args:
        features_38d: (T, 38) - 归一化的38维特征
        mean: (38,) - 数据集均值
        std: (38,) - 数据集标准差
    
    Returns:
        root_trajectory: (T, 2) - 根节点XY轨迹（起点在原点）
    """
    # Step 1: 反归一化
    features_real = features_38d * std + mean
    
    # 使用 raw 函数提取
    return extract_root_trajectory_raw(features_real)


def extract_root_trajectory_raw(features_38d_real):
    """
    从未归一化的38维特征直接提取根节点XY轨迹
    
    Args:
        features_38d_real: (T, 38) - 未归一化的38维特征（原始物理量）
    
    Returns:
        root_trajectory: (T, 2) - 根节点XY轨迹（起点在原点）
    """
    # 提取 root_vel_xy 和 root_rot_6d
    root_vel_xy = features_38d_real[:, 29:31]       # (T, 2)
    root_rot_6d = features_38d_real[:, 32:38]       # (T, 6)
    
    # 转换 root_rot 6D -> wxyz 四元数
    root_rot_quat = rot6d_to_quat_wxyz(root_rot_6d)  # (T, 4)
    
    # 积分 root_vel_xy 恢复根轨迹
    root_trajectory = recover_root_xy_from_velocity(root_vel_xy, root_rot_quat)  # (T, 2)
    
    return root_trajectory


def evaluate_rtc_batch(motions_gen, motions_gt, m_lens_gen, m_lens_gt, mean, std):
    """
    批量评估 Root Trajectory Consistency
    
    Args:
        motions_gen: (B, T, 38) - 生成动作（归一化）
        motions_gt: (B, T, 38) - GT 动作（归一化）
        m_lens_gen: (B,) - 生成动作的实际长度
        m_lens_gt: (B,) - GT 动作的实际长度
        mean: (38,) - 数据集均值
        std: (38,) - 数据集标准差
    
    Returns:
        dict: 批次平均的 RTC 及子分数
    """
    batch_size = motions_gen.shape[0]
    all_rtc = []
    all_shape = []
    all_extent = []
    
    for i in range(batch_size):
        # 提取有效长度的序列
        motion_gen = motions_gen[i, :m_lens_gen[i], :]  # (T_gen, 38)
        motion_gt = motions_gt[i, :m_lens_gt[i], :]     # (T_gt, 38)
        
        # 提取根轨迹
        traj_gen = extract_root_trajectory_from_38d(motion_gen, mean, std)
        traj_gt = extract_root_trajectory_from_38d(motion_gt, mean, std)
        
        # 计算 RTC
        scores = calculate_rtc_single(traj_gen, traj_gt)
        
        all_rtc.append(scores['rtc'])
        all_shape.append(scores['shape_score'])
        all_extent.append(scores['extent_score'])
    
    # 返回批次平均
    return {
        'rtc': np.mean(all_rtc),
        'shape_score': np.mean(all_shape),
        'extent_score': np.mean(all_extent)
    }


def average_rtc_results(rtc_scores):
    """
    聚合多个批次的 RTC 结果
    
    Args:
        rtc_scores: List[dict] - 每个批次的 RTC 字典
    
    Returns:
        dict: 所有批次的平均 RTC
    """
    all_rtc = [s['rtc'] for s in rtc_scores]
    all_shape = [s['shape_score'] for s in rtc_scores]
    all_extent = [s['extent_score'] for s in rtc_scores]
    
    return {
        'rtc': np.mean(all_rtc),
        'shape_score': np.mean(all_shape),
        'extent_score': np.mean(all_extent)
    }


if __name__ == "__main__":
    """测试 RTC 计算"""
    print("Testing Root Trajectory Consistency calculation...")
    
    # 测试1：圆形轨迹（半径略有差异）
    print("\n[Test 1] 圆形轨迹（半径略有差异）")
    T = 100
    theta = np.linspace(0, 2 * np.pi, T)
    traj_gt = np.stack([1.5 * np.cos(theta), 1.5 * np.sin(theta)], axis=1)
    traj_gen = np.stack([1.6 * np.cos(theta), 1.6 * np.sin(theta)], axis=1)
    
    scores = calculate_rtc_single(traj_gen, traj_gt)
    print(f"  RTC: {scores['rtc']:.4f}")
    print(f"  Shape: {scores['shape_score']:.4f}")
    print(f"  Extent: {scores['extent_score']:.4f}")
    print(f"  预期：RTC ≈ 0.99（形状几乎相同，半径略有差异）")
    
    # 测试2：半圆 vs 全圆
    print("\n[Test 2] 半圆 vs 全圆")
    theta_half = np.linspace(0, np.pi, T // 2)
    traj_gen = np.stack([1.5 * np.cos(theta_half), 1.5 * np.sin(theta_half)], axis=1)
    traj_gt = np.stack([1.5 * np.cos(theta), 1.5 * np.sin(theta)], axis=1)
    
    scores = calculate_rtc_single(traj_gen, traj_gt)
    print(f"  RTC: {scores['rtc']:.4f}")
    print(f"  Shape: {scores['shape_score']:.4f}")
    print(f"  Extent: {scores['extent_score']:.4f}")
    print(f"  预期：RTC << 0.4（只走了一半，形状和覆盖都不足）")
    
    # 测试3：直线（不同速度）
    print("\n[Test 3] 直线前进（不同速度）")
    # GT: 3m in 100 frames
    traj_gt = np.stack([np.linspace(0, 3, T), np.zeros(T)], axis=1)
    # Gen: 3m in 150 frames (slower)
    T_slow = 150
    traj_gen = np.stack([np.linspace(0, 3, T_slow), np.zeros(T_slow)], axis=1)
    
    scores = calculate_rtc_single(traj_gen, traj_gt)
    print(f"  RTC: {scores['rtc']:.4f}")
    print(f"  Shape: {scores['shape_score']:.4f}")
    print(f"  Extent: {scores['extent_score']:.4f}")
    print(f"  预期：RTC ≈ 1.0（弧长参数化消除速度差异）")
    
    # 测试4：直线 vs 圆形
    print("\n[Test 4] 直线 vs 圆形")
    traj_gen = np.stack([np.linspace(0, 3, T), np.zeros(T)], axis=1)
    theta = np.linspace(0, 2 * np.pi, T)
    traj_gt = np.stack([1.5 * np.cos(theta), 1.5 * np.sin(theta)], axis=1)
    
    scores = calculate_rtc_single(traj_gen, traj_gt)
    print(f"  RTC: {scores['rtc']:.4f}")
    print(f"  Shape: {scores['shape_score']:.4f}")
    print(f"  Extent: {scores['extent_score']:.4f}")
    print(f"  预期：RTC ≈ 0.0（形状完全不同）")
    
    # 测试5：都静态
    print("\n[Test 5] 双方都静态")
    traj_gen = np.zeros((T, 2))
    traj_gt = np.zeros((T, 2))
    
    scores = calculate_rtc_single(traj_gen, traj_gt)
    print(f"  RTC: {scores['rtc']:.4f}")
    print(f"  Shape: {scores['shape_score']:.4f}")
    print(f"  Extent: {scores['extent_score']:.4f}")
    print(f"  预期：RTC = 1.0（都不动 = 完美一致）")
    
    print("\n✓ RTC 计算测试完成！")
