"""Utilities for reconstructing NPZ files from 38D velocity-based motion representation."""
import numpy as np
import torch
from scipy.signal import savgol_filter
from utils.rotation_utils import rot6d_to_quat_wxyz
from utils.robot_process import recover_root_xy_from_velocity


def compute_linear_velocity(positions, fps=50):
    """Compute linear velocity from positions using forward differences."""
    dt = 1.0 / fps
    velocities = np.zeros_like(positions)
    velocities[:-1] = (positions[1:] - positions[:-1]) / dt
    velocities[-1] = velocities[-2] if len(velocities) > 1 else 0
    return velocities


def smooth_positions_savgol(positions, window_length=5, polyorder=3):
    """Smooth position trajectories using Savitzky-Golay filter."""
    frames, N, dims = positions.shape
    if window_length % 2 == 0:
        window_length += 1
    if frames < window_length:
        return positions
    if polyorder >= window_length:
        polyorder = window_length - 1
    smoothed = np.zeros_like(positions)
    for joint_idx in range(N):
        for dim_idx in range(dims):
            smoothed[:, joint_idx, dim_idx] = savgol_filter(
                positions[:, joint_idx, dim_idx],
                window_length=window_length,
                polyorder=polyorder,
                mode='interp'
            )
    return smoothed


def adaptive_smooth_positions(positions, fps=50):
    """Adaptive smoothing: stronger on initial frames to remove startup jitter.

    Strategy:
    1. First 15 frames use window=11 strong smoothing
    2. Global window=7 moderate smoothing
    """
    frames, N, dims = positions.shape
    smoothed = positions.copy()
    if frames >= 7:
        for joint_idx in range(N):
            for dim_idx in range(dims):
                smoothed[:, joint_idx, dim_idx] = savgol_filter(
                    smoothed[:, joint_idx, dim_idx],
                    window_length=7, polyorder=3, mode='interp'
                )
    if frames >= 15:
        for joint_idx in range(N):
            for dim_idx in range(dims):
                smoothed[:15, joint_idx, dim_idx] = savgol_filter(
                    positions[:15, joint_idx, dim_idx],
                    window_length=11, polyorder=3, mode='interp'
                )
    return smoothed


def force_static_start(positions, static_frames=2, blend_frames=8):
    """Force near-static start to fix large initial velocity.

    First static_frames frames hold the frame-0 position, then linearly
    blend to the original trajectory over blend_frames.
    """
    frames, N, dims = positions.shape
    if frames < static_frames + blend_frames:
        return positions
    start_pos = positions[0].copy()
    for i in range(static_frames):
        positions[i] = start_pos
    target_frame = static_frames + blend_frames
    target_pos = positions[target_frame].copy()
    for i in range(static_frames, target_frame):
        alpha = (i - static_frames) / blend_frames
        positions[i] = start_pos * (1 - alpha) + target_pos * alpha
    return positions


def reshape_generated_motion_38d(motion_38d, fps=50, smooth=False, smooth_window=5,
                                 adaptive_smooth=False, static_start=False,
                                 static_frames=2, blend_frames=8):
    """Reconstruct minimal NPZ from 38D velocity-based representation.

    38D format: [joint_pos(29), root_vel_xy(2), root_z(1), root_rot_6d(6)]

    Integrates root velocity to recover position via quaternion-based
    body-frame transformation.

    Args:
        motion_38d: (T, 38) numpy array
        fps: frame rate (default 50)
        smooth: apply basic Savitzky-Golay smoothing (default False)
        smooth_window: smoothing window size (default 5)
        adaptive_smooth: stronger smoothing on first 15 frames
        static_start: force static start frames
        static_frames: number of static frames (default 2)
        blend_frames: number of blend frames (default 8)

    Returns:
        dict with keys: fps, joint_pos (T,29), root_pos (T,3), root_rot (T,4 wxyz)
    """
    frames = motion_38d.shape[0]

    # Split 38D components
    joint_pos = motion_38d[:, :29]         # (T, 29)
    root_vel_xy = motion_38d[:, 29:31]     # (T, 2) planar velocity in body frame
    root_z = motion_38d[:, 31:32]          # (T, 1) root height
    root_rot_6d = motion_38d[:, 32:38]     # (T, 6) continuous 6D rotation

    root_rot_quat = rot6d_to_quat_wxyz(root_rot_6d)  # (T, 4) wxyz
    root_xy_pos = recover_root_xy_from_velocity(root_vel_xy, root_rot_quat)  # (T, 2)
    root_pos = np.concatenate([root_xy_pos, root_z], axis=1)  # (T, 3)

    # Smoothing
    if smooth or adaptive_smooth or static_start:
        root_pos_reshaped = root_pos[:, np.newaxis, :]  # (T, 1, 3)
        if adaptive_smooth:
            root_pos_reshaped = adaptive_smooth_positions(root_pos_reshaped, fps=fps)
        elif smooth:
            root_pos_reshaped = smooth_positions_savgol(root_pos_reshaped, window_length=smooth_window, polyorder=3)
        if static_start:
            root_pos_reshaped = force_static_start(root_pos_reshaped, static_frames=static_frames, blend_frames=blend_frames)
        root_pos = root_pos_reshaped[:, 0, :]

        # Smooth joint positions
        joint_pos_reshaped = joint_pos[:, :, np.newaxis]
        if adaptive_smooth:
            joint_pos_reshaped = adaptive_smooth_positions(joint_pos_reshaped, fps=fps)
        elif smooth:
            joint_pos_reshaped = smooth_positions_savgol(joint_pos_reshaped, window_length=smooth_window, polyorder=3)
        if static_start:
            joint_pos_reshaped = force_static_start(joint_pos_reshaped, static_frames=static_frames, blend_frames=blend_frames)
        joint_pos = joint_pos_reshaped[:, :, 0]

    return {
        'fps': np.array([fps]),
        'joint_pos': joint_pos.astype(np.float32),
        'root_pos': root_pos.astype(np.float32),
        'root_rot': root_rot_quat.astype(np.float32),
    }


if __name__ == '__main__':
    import sys
    print("Testing 38D NPZ reconstruction...")
    frames, dim = 100, 38
    motion_38d = np.random.randn(frames, dim).astype(np.float32)
    motion_38d[:, 31] = 0.78  # root height ~0.78m
    result = reshape_generated_motion_38d(motion_38d, fps=50)
    print(f"  joint_pos: {result['joint_pos'].shape}")
    print(f"  root_pos:  {result['root_pos'].shape}")
    print(f"  root_rot:  {result['root_rot'].shape}")
    print("OK")
