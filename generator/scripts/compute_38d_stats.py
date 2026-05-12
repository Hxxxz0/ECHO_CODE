"""
Compute 38D feature normalization statistics (Mean, Std).

38D format (velocity-based):
- joint_pos:    29D (joint angles)
- root_vel_xy:   2D (root velocity in XY plane, local frame)
- root_z:        1D (root height)
- root_rot_6d:   6D (root rotation in continuous 6D representation)
Total:          38D

This script processes all NPZ files and computes mean/std for training.
"""
import os
import sys
import glob
import argparse
import multiprocessing as mp
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.robot_process import process_robot_npz


def extract_38d_features(npz_path):
    """
    Extract 38D features from NPZ file using velocity-based representation.
    
    Args:
        npz_path: str - path to NPZ file
    
    Returns:
        features: (T, 38) numpy array - [joint_pos, root_vel_xy, root_z, root_rot_6d]
    """
    data = np.load(npz_path)
    
    # Use the unified processing pipeline (includes alignment and velocity extraction)
    features = process_robot_npz(data, root_idx=0)  # (T, 38)
    
    return features.astype(np.float32)


def _worker_accumulate(npz_path):
    """Worker: load one NPZ and return sum, sumsq, count for 38D features."""
    features = extract_38d_features(npz_path).astype(np.float64)
    feat_sum = features.sum(axis=0)
    feat_sumsq = (features * features).sum(axis=0)
    count = features.shape[0]
    return npz_path, feat_sum, feat_sumsq, count


def main():
    parser = argparse.ArgumentParser(description="Compute 38D normalization stats (velocity-based).")
    parser.add_argument("--data_dir", type=str, default="robot_humanml_data/npz")
    parser.add_argument("--output_dir", type=str, default="robot_humanml_data")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    parser.add_argument("--chunksize", type=int, default=10)
    args = parser.parse_args()

    # Configuration
    data_dir = args.data_dir
    output_dir = args.output_dir
    
    print(f"[INFO] Computing 38D statistics from {data_dir}")
    
    # Find all NPZ files
    npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
    print(f"[INFO] Found {len(npz_files)} NPZ files")
    
    if len(npz_files) == 0:
        print(f"[ERROR] No NPZ files found in {data_dir}!")
        return
    
    # Accumulate sums in parallel (no full concatenation)
    total_sum = np.zeros((38,), dtype=np.float64)
    total_sumsq = np.zeros((38,), dtype=np.float64)
    total_count = 0
    failed_files = []

    workers = max(1, int(args.workers))
    chunksize = max(1, int(args.chunksize))
    print(f"[INFO] Using {workers} workers, chunksize={chunksize}")

    with mp.Pool(processes=workers) as pool:
        for npz_path, feat_sum, feat_sumsq, count in tqdm(
            pool.imap_unordered(_worker_accumulate, npz_files, chunksize=chunksize),
            total=len(npz_files),
            desc="Extracting 38D features",
        ):
            if count == 0:
                failed_files.append((npz_path, "Empty features"))
                continue
            total_sum += feat_sum
            total_sumsq += feat_sumsq
            total_count += count
    
    if len(failed_files) > 0:
        print(f"\n[WARN] Failed to process {len(failed_files)} files")
    
    if total_count == 0:
        print("[ERROR] No valid features extracted!")
        return

    print(f"\n[INFO] Total frames: {total_count:,}")
    print("[INFO] Feature dimension: 38")

    # Compute statistics from sums
    mean = total_sum / total_count
    var = total_sumsq / total_count - mean * mean
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)
    
    # Improved Std handling: prevent division by zero and numerical explosion
    # If Std is too small (dimension barely changes), set it to 1.0 to skip normalization
    small_std_mask = std < 1e-4
    print(f"\n[INFO] Found {np.sum(small_std_mask)} dimensions with tiny Std (< 1e-4). Setting them to 1.0.")
    std[small_std_mask] = 1.0
    
    # Print statistics breakdown
    print("\n" + "="*60)
    print("38D Feature Statistics")
    print("="*60)
    print(f"joint_pos (dims 0-28):")
    print(f"  Mean: min={mean[:29].min():.4f}, max={mean[:29].max():.4f}, avg={mean[:29].mean():.4f}")
    print(f"  Std:  min={std[:29].min():.4f}, max={std[:29].max():.4f}, avg={std[:29].mean():.4f}")
    print(f"\nroot_vel_xy (dims 29-30):")
    print(f"  Mean: {mean[29:31]}")
    print(f"  Std:  {std[29:31]}")
    print(f"\nroot_z (dim 31):")
    print(f"  Mean: {mean[31]:.4f}")
    print(f"  Std:  {std[31]:.4f}")
    print(f"\nroot_rot_6d (dims 32-37):")
    print(f"  Mean: {mean[32:38]}")
    print(f"  Std:  {std[32:38]}")
    print("="*60)
    
    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    mean_path = os.path.join(output_dir, "Mean_38d.npy")
    std_path = os.path.join(output_dir, "Std_38d.npy")
    
    np.save(mean_path, mean)
    np.save(std_path, std)
    
    print(f"\n[SUCCESS] Saved normalization statistics:")
    print(f"  {mean_path} - shape: {mean.shape}")
    print(f"  {std_path} - shape: {std.shape}")
    
    # Verify by loading
    mean_loaded = np.load(mean_path)
    std_loaded = np.load(std_path)
    assert mean_loaded.shape == (38,), f"Mean shape mismatch: {mean_loaded.shape}"
    assert std_loaded.shape == (38,), f"Std shape mismatch: {std_loaded.shape}"
    print(f"\n[INFO] Verification: Files loaded successfully ✓")


if __name__ == "__main__":
    main()

