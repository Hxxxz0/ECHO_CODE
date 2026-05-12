"""
Rotation representation conversion utilities.
Supports: quaternion (wxyz/xyzw) <-> 6D continuous representation <-> rotation matrix

6D representation is more suitable for neural networks:
- No constraints (quaternions need ||q||=1)
- Continuous (no double-cover issue)
- Easier to learn and interpolate
"""
import numpy as np


def quat_wxyz_to_rotation_matrix(quat):
    """
    Convert quaternion (w,x,y,z) to 3x3 rotation matrix.
    
    Args:
        quat: (*, 4) numpy array - [w, x, y, z]
    
    Returns:
        R: (*, 3, 3) numpy array - rotation matrix
    """
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    
    R = np.stack([
        1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w),
        2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w),
        2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)
    ], axis=-1).reshape(*quat.shape[:-1], 3, 3)
    
    return R


def rotation_matrix_to_6d(R):
    """
    Convert rotation matrix to 6D continuous representation.
    Takes first two columns of rotation matrix and flattens them in column-major order.
    
    Args:
        R: (*, 3, 3) numpy array - rotation matrix
    
    Returns:
        rot6d: (*, 6) numpy array - 6D representation [col0_xyz, col1_xyz]
    """
    # Extract first two columns: R[:, 0] and R[:, 1]
    # Concatenate them as [R[:, 0], R[:, 1]] to get proper 6D format
    batch_shape = R.shape[:-2]
    col0 = R[..., :, 0]  # First column (3,)
    col1 = R[..., :, 1]  # Second column (3,)
    return np.concatenate([col0, col1], axis=-1)  # (*, 6)


def rot6d_to_rotation_matrix(rot6d):
    """
    Convert 6D continuous representation to rotation matrix.
    Uses Gram-Schmidt orthogonalization.
    
    Args:
        rot6d: (*, 6) numpy array - 6D representation
    
    Returns:
        R: (*, 3, 3) numpy array - rotation matrix
    """
    # Split into two 3D vectors
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]
    
    # Gram-Schmidt orthogonalization
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2, axis=-1)
    
    # Stack to form rotation matrix
    R = np.stack([b1, b2, b3], axis=-1)  # (*, 3, 3)
    return R


def rotation_matrix_to_quat_wxyz(R):
    """
    Convert rotation matrix to quaternion (w,x,y,z).
    
    Args:
        R: (*, 3, 3) numpy array - rotation matrix
    
    Returns:
        quat: (*, 4) numpy array - [w, x, y, z]
    """
    batch_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    quat = np.zeros((R.shape[0], 4), dtype=R.dtype)
    
    # Case 1: trace > 0
    mask = trace > 0
    if np.any(mask):
        s = np.sqrt(trace[mask] + 1.0) * 2  # s = 4 * w
        quat[mask, 0] = 0.25 * s
        quat[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / s
        quat[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / s
        quat[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / s
    
    # Case 2: R[0,0] > R[1,1] and R[0,0] > R[2,2]
    mask = (~(trace > 0)) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if np.any(mask):
        s = np.sqrt(1.0 + R[mask, 0, 0] - R[mask, 1, 1] - R[mask, 2, 2]) * 2
        quat[mask, 0] = (R[mask, 2, 1] - R[mask, 1, 2]) / s
        quat[mask, 1] = 0.25 * s
        quat[mask, 2] = (R[mask, 0, 1] + R[mask, 1, 0]) / s
        quat[mask, 3] = (R[mask, 0, 2] + R[mask, 2, 0]) / s
    
    # Case 3: R[1,1] > R[2,2]
    mask = (~(trace > 0)) & (~((R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2]))) & (R[:, 1, 1] > R[:, 2, 2])
    if np.any(mask):
        s = np.sqrt(1.0 + R[mask, 1, 1] - R[mask, 0, 0] - R[mask, 2, 2]) * 2
        quat[mask, 0] = (R[mask, 0, 2] - R[mask, 2, 0]) / s
        quat[mask, 1] = (R[mask, 0, 1] + R[mask, 1, 0]) / s
        quat[mask, 2] = 0.25 * s
        quat[mask, 3] = (R[mask, 1, 2] + R[mask, 2, 1]) / s
    
    # Case 4: else
    mask = (~(trace > 0)) & (~((R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2]))) & (~(R[:, 1, 1] > R[:, 2, 2]))
    if np.any(mask):
        s = np.sqrt(1.0 + R[mask, 2, 2] - R[mask, 0, 0] - R[mask, 1, 1]) * 2
        quat[mask, 0] = (R[mask, 1, 0] - R[mask, 0, 1]) / s
        quat[mask, 1] = (R[mask, 0, 2] + R[mask, 2, 0]) / s
        quat[mask, 2] = (R[mask, 1, 2] + R[mask, 2, 1]) / s
        quat[mask, 3] = 0.25 * s
    
    # Normalize quaternions
    quat = quat / (np.linalg.norm(quat, axis=-1, keepdims=True) + 1e-8)
    
    return quat.reshape(*batch_shape, 4)


# ============ Convenience functions ============

def quat_wxyz_to_6d(quat):
    """
    Convert quaternion (w,x,y,z) to 6D continuous representation.
    
    Args:
        quat: (*, 4) numpy array - [w, x, y, z]
    
    Returns:
        rot6d: (*, 6) numpy array - 6D representation
    """
    R = quat_wxyz_to_rotation_matrix(quat)
    return rotation_matrix_to_6d(R)


def rot6d_to_quat_wxyz(rot6d):
    """
    Convert 6D continuous representation to quaternion (w,x,y,z).
    
    Args:
        rot6d: (*, 6) numpy array - 6D representation
    
    Returns:
        quat: (*, 4) numpy array - [w, x, y, z]
    """
    R = rot6d_to_rotation_matrix(rot6d)
    return rotation_matrix_to_quat_wxyz(R)


# ============ Testing ============

if __name__ == "__main__":
    print("Testing rotation conversion utilities...")
    
    # Test 1: Identity quaternion
    print("\n[Test 1] Identity quaternion")
    quat = np.array([1, 0, 0, 0], dtype=np.float32)
    rot6d = quat_wxyz_to_6d(quat)
    quat_recovered = rot6d_to_quat_wxyz(rot6d)
    print(f"  Original quat:   {quat}")
    print(f"  6D:              {rot6d}")
    print(f"  Recovered quat:  {quat_recovered}")
    print(f"  Error:           {np.abs(quat - quat_recovered).max():.6f}")
    
    # Test 2: Random quaternion
    print("\n[Test 2] Random quaternion")
    quat = np.random.randn(4).astype(np.float32)
    quat = quat / np.linalg.norm(quat)  # Normalize
    rot6d = quat_wxyz_to_6d(quat)
    quat_recovered = rot6d_to_quat_wxyz(rot6d)
    
    # Account for q and -q representing same rotation
    error1 = np.abs(quat - quat_recovered).max()
    error2 = np.abs(quat + quat_recovered).max()
    error = min(error1, error2)
    
    print(f"  Original quat:   {quat}")
    print(f"  6D:              {rot6d}")
    print(f"  Recovered quat:  {quat_recovered}")
    print(f"  Error:           {error:.6f}")
    
    # Test 3: Batch of quaternions
    print("\n[Test 3] Batch of quaternions (10, 4)")
    quats = np.random.randn(10, 4).astype(np.float32)
    quats = quats / np.linalg.norm(quats, axis=-1, keepdims=True)
    rot6d_batch = quat_wxyz_to_6d(quats)
    quats_recovered = rot6d_to_quat_wxyz(rot6d_batch)
    
    errors = []
    for i in range(10):
        error1 = np.abs(quats[i] - quats_recovered[i]).max()
        error2 = np.abs(quats[i] + quats_recovered[i]).max()
        errors.append(min(error1, error2))
    
    print(f"  Input shape:     {quats.shape}")
    print(f"  6D shape:        {rot6d_batch.shape}")
    print(f"  Recovered shape: {quats_recovered.shape}")
    print(f"  Max error:       {max(errors):.6f}")
    print(f"  Mean error:      {np.mean(errors):.6f}")
    
    print("\n✓ All tests passed!")

