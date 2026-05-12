import json
import socket
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R

from common.joint_mapper import create_isaac_to_real_mapper
from common.math_utils import (
    _linspace_rows,
    _remove_yaw_keep_rp_wxyz,
    _slerp,
    _yaw_component_wxyz,
    _zero_z,
)
from common.utils import DictToClass, MotionUDPServer
from paths import ASSETS_DIR, REAL_G1_ROOT

def benchmark_onnx(module, sample_input, runs=100, warmup=10, desc=""):
    for _ in range(warmup):
        _ = module(sample_input)

    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = module(sample_input)
        t1 = time.perf_counter()
        ts.append((t1 - t0) * 1000.0)

    mean = statistics.mean(ts)
    stdev = statistics.pstdev(ts)
    p50 = np.percentile(ts, 50)
    p90 = np.percentile(ts, 90)
    p95 = np.percentile(ts, 95)
    p99 = np.percentile(ts, 99)

    print(f"[{desc}] runs={runs}, warmup={warmup}")
    print(f"mean={mean:.3f} ms, stdev={stdev:.3f} ms")
    print(f"p50={p50:.3f} ms, p90={p90:.3f} ms, p95={p95:.3f} ms, p99={p99:.3f} ms")
    return {"mean": mean, "stdev": stdev, "p50": p50, "p90": p90, "p95": p95, "p99": p99}


class ONNXModule:
    def __init__(self, path: str):
        self.ort_session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        meta_path = path.replace(".onnx", ".json")
        with open(meta_path, "r") as f:
            self.meta = json.load(f)
        self.in_keys = [k if isinstance(k, str) else tuple(k) for k in self.meta["in_keys"]]
        self.out_keys = [k if isinstance(k, str) else tuple(k) for k in self.meta["out_keys"]]

    def __call__(self, input: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        args = {
            inp.name: input[key]
            for inp, key in zip(self.ort_session.get_inputs(), self.in_keys)
            if key in input
        }
        outputs = self.ort_session.run(None, args)
        outputs = {k: v for k, v in zip(self.out_keys, outputs)}
        return outputs

# =========================================
# Upright Detector
# =========================================
class UprightDetector:
    """
    Monitors robot IMU and joint angles to detect successful standing up.
    Uses roll/pitch angles AND knee joint angles to determine if robot is truly upright.
    """
    def __init__(self, controller, threshold_deg: float = 15.0, knee_threshold_rad: float = 0.6, consecutive_frames: int = 4):
        """
        Args:
            controller: Robot controller with .quat and .qj_isaac attributes
            threshold_deg: Maximum absolute roll/pitch angle (degrees) to consider upright
            knee_threshold_rad: Maximum knee angle (radians), 0.6 rad ≈ 34°
            consecutive_frames: Number of consecutive frames that must satisfy condition
        """
        self.controller = controller
        self.threshold_rad = np.deg2rad(threshold_deg)
        self.knee_threshold_rad = knee_threshold_rad
        self.consecutive_frames = consecutive_frames
        self.upright_count = 0
        self.is_monitoring = False

        # Isaac joint order indices for knee joints
        self.left_knee_idx = 9   # "left_knee_joint" in ISAAC_JOINT_ORDER
        self.right_knee_idx = 10  # "right_knee_joint" in ISAAC_JOINT_ORDER

    def start_monitoring(self):
        """Start monitoring for upright condition"""
        self.is_monitoring = True
        self.upright_count = 0
        print(f"[UprightDetector] Started monitoring")
        print(f"  - IMU姿态: roll/pitch < ±{np.rad2deg(self.threshold_rad):.1f}°")
        print(f"  - 膝盖角度: < {np.rad2deg(self.knee_threshold_rad):.1f}° (伸直)")
        print(f"  - 连续帧数: {self.consecutive_frames}")

    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        self.upright_count = 0
        print(f"[UprightDetector] Stopped monitoring")

    def check(self) -> bool:
        """
        Check if robot is upright based on IMU quaternion AND knee angles.
        Returns True if upright condition satisfied for consecutive_frames.
        """
        if not self.is_monitoring:
            return False

        # 1. Check IMU orientation (roll and pitch)
        quat = self.controller.quat  # [w, x, y, z] scalar_first
        r = R.from_quat(quat, scalar_first=True)
        roll, pitch, yaw = r.as_euler('xyz')

        imu_ok = (abs(roll) < self.threshold_rad) and (abs(pitch) < self.threshold_rad)

        # 2. Check knee angles (legs should be roughly straight)
        qj = self.controller.qj_isaac
        left_knee = abs(qj[self.left_knee_idx])
        right_knee = abs(qj[self.right_knee_idx])
        knees_ok = (left_knee < self.knee_threshold_rad) or (right_knee < self.knee_threshold_rad)

        is_upright = imu_ok and knees_ok

        # Debug print every 30 frames
        if self.upright_count % 30 == 0:
            print(f"[UprightDetector] 检测中 [{self.upright_count}/{self.consecutive_frames}]:")
            print(f"  IMU: roll={np.rad2deg(roll):.1f}°, pitch={np.rad2deg(pitch):.1f}° {'✓' if imu_ok else '✗'}")
            print(f"  膝盖: 左={np.rad2deg(left_knee):.1f}°, 右={np.rad2deg(right_knee):.1f}° {'✓' if knees_ok else '✗'}")

        if is_upright:
            self.upright_count += 1
            if self.upright_count >= self.consecutive_frames:
                print(f"[UprightDetector] SUCCESS - Robot upright!")
                print(f"  - IMU: roll={np.rad2deg(roll):.1f}°, pitch={np.rad2deg(pitch):.1f}°")
                print(f"  - 膝盖: 左={np.rad2deg(left_knee):.1f}°, 右={np.rad2deg(right_knee):.1f}°")
                return True
        else:
            if self.upright_count > 0:
                reason = []
                if not imu_ok:
                    reason.append(f"IMU姿态(roll={np.rad2deg(roll):.1f}°, pitch={np.rad2deg(pitch):.1f}°)")
                if not knees_ok:
                    reason.append(f"膝盖弯曲(左={np.rad2deg(left_knee):.1f}°, 右={np.rad2deg(right_knee):.1f}°)")
                print(f"[UprightDetector] 检测中断: {', '.join(reason)}")
            self.upright_count = 0

        return False


# =========================================
# Policy Base
# =========================================
class Policy:
    FADE_OUT_DURATION = 2.0  # s

    def __init__(self, name: str, policy_cfg: DictToClass, controller):
        self.name = name
        self.controller = controller

        self.config = policy_cfg

        # Resolve policy path relative to repo root for robustness
        p = Path(policy_cfg.policy_path)
        self.policy_path = str(p if p.is_absolute() else (REAL_G1_ROOT / p))
        self.action_joint_names = list(policy_cfg.action_joint_names)
        self.action_scale_isaac = np.array(policy_cfg.action_scale, dtype=np.float32)
        self.alpha = float(policy_cfg.action_alpha)
        self.lowstate_alpha = float(policy_cfg.lowstate_alpha)
        self.action_clip = float(policy_cfg.action_clip)

        if hasattr(policy_cfg, "kps_real"):
            self.kps_real = np.array(policy_cfg.kps_real, dtype=np.float32)
        if hasattr(policy_cfg, "kds_real"):
            self.kds_real = np.array(policy_cfg.kds_real, dtype=np.float32)

        assert len(self.action_joint_names) == len(self.action_scale_isaac), (
            f"[{self.name}] action_joint_names ({len(self.action_joint_names)}) "
            f"!= action_scale ({len(self.action_scale_isaac)})"
        )

        self.module = ONNXModule(self.policy_path)

        self.mapper_action = create_isaac_to_real_mapper(
            self.action_joint_names,
            self.controller.config.real_joint_names
        )
        map_info = self.mapper_action.get_mapping_info()
        print(f"[Policy:{self.name}] Action mapping: {map_info['mapped_joints']}/{map_info['from_space_size']} mapped")
        if map_info['unmapped_from_joints']:
            print(f"[Policy:{self.name}] Unmapped policy action joints: {map_info['unmapped_from_joints']}")
        if map_info['unmapped_to_joints']:
            print(f"[Policy:{self.name}] Unmapped Real joints: {map_info['unmapped_to_joints']}")

        self.policy_input: Optional[Dict[str, np.ndarray]] = None
        self.applied_action_isaac = np.zeros(len(self.action_joint_names), dtype=np.float32)
        self.last_action = np.zeros(len(self.action_joint_names), dtype=np.float32)

        self._fading_deadline: Optional[float] = None
        self._active: bool = False

        self.obs_modules = []
        self.num_obs = 0
        self._build_obs_modules()

        self.policy_input = {
            "policy": np.zeros((1, self.num_obs), dtype=np.float32),
            "is_init": np.ones((1,), dtype=bool)
        }
        benchmark_onnx(self.module, self.policy_input, runs=100, warmup=200, desc="model@cuda")

    # -------- lifecycle ----------
    def fade_in(self):
        self.reset()
        self._active = True
        self._fading_deadline = None
        print(f"[Policy:{self.name}] fade_in()")

    def fade_out(self) -> float:
        self._fading_deadline = time.monotonic() + self.FADE_OUT_DURATION
        print(f"[Policy:{self.name}] fade_out() - continue until {self._fading_deadline:.3f}")
        return self._fading_deadline

    def is_fading(self) -> bool:
        return self._fading_deadline is not None

    def fading_done(self) -> bool:
        return self._fading_deadline is not None and time.monotonic() >= self._fading_deadline

    def deactivate(self):
        self._active = False
        self._fading_deadline = None
        print(f"[Policy:{self.name}] deactivated")

    # -------- abstract hooks ----------
    def _build_obs_modules(self):
        raise NotImplementedError

    def _reset_obs_modules(self):
        for m in self.obs_modules:
            if hasattr(m, "reset") and callable(m.reset):
                m.reset()

    def update_obs(self):
        obs_list = []
        for m in self.obs_modules:
            m.update()
            obs_list.append(m.compute())
        if self.policy_input is None:
            self.policy_input = {
                "policy": np.zeros((1, self.num_obs), dtype=np.float32),
                "is_init": np.ones((1,), dtype=bool),
            }
        else:
            self.policy_input["policy"][0, :] = np.concatenate(obs_list, axis=0)

    def compute_action(self) -> np.ndarray:
        try:
            out = self.module(self.policy_input)
        except Exception as e:
            print(f"[Policy:{self.name}] ONNX forward failed: {e}")
            return np.zeros(self.controller.dof_size_real, dtype=np.float32)

        if ("next", "adapt_hx") in out:
            self.policy_input["adapt_hx"][:] = out["next", "adapt_hx"]
        self.policy_input["is_init"][:] = False

        action_isaac = out["action"].copy()[0].astype(np.float32).clip( -self.action_clip, self.action_clip)
        self.last_action[:] = action_isaac
        self.applied_action_isaac[:] = action_isaac * self.action_scale_isaac

        action_real = self.mapper_action.map_action_from_to(self.applied_action_isaac)
        return action_real

    def reset(self):
        self.policy_input = None
        self.applied_action_isaac[:] = 0.0
        self.last_action[:] = 0.0
        self._reset_obs_modules()

# =========================================
# Policy Subclasses
# =========================================
def remap_joint_array_by_names(
    data: np.ndarray,
    source_joint_names: List[str],
    target_joint_names: List[str],
) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D joint array [T, J], got shape={data.shape}")
    if data.shape[1] != len(source_joint_names):
        raise ValueError(
            f"Joint dim mismatch: data has {data.shape[1]} dims, "
            f"but source_joint_names has {len(source_joint_names)} names."
        )

    name_to_idx = {name: i for i, name in enumerate(source_joint_names)}
    remap = np.zeros((data.shape[0], len(target_joint_names)), dtype=np.float32)
    for i, name in enumerate(target_joint_names):
        j = name_to_idx.get(name, None)
        if j is not None:
            remap[:, i] = data[:, j]
    return remap

class TrackingPolicyRaw(Policy):
    def __init__(self, name: str, policy_cfg: DictToClass, controller):
        # ---- Config ---------------------------------------------------------
        self.body_name = "torso_link"
        self.transition_steps = int(getattr(policy_cfg, "transition_steps", 100))
        self.compliance_flag_value = float(getattr(policy_cfg, "compliance_flag_value", 0.0))
        self.udp_enable = bool(getattr(policy_cfg, "udp_enable", True))
        self.udp_host = str(getattr(policy_cfg, "udp_host", "127.0.0.1"))
        self.udp_port = int(getattr(policy_cfg, "udp_port", 28562))
        self.status_port = int(getattr(policy_cfg, "status_port", 28563))
        self.dataset_joint_names = list(getattr(policy_cfg, "dataset_joint_names", []))
        if len(self.dataset_joint_names) == 0:
            raise ValueError(
                "[TrackingPolicyRaw] dataset_joint_names must be provided in tracking.yaml."
            )
        self.obs_joint_names = controller.config.isaac_joint_names_state

        # ---- Load motions; keep all root data (no yaw split) ----------------
        self.motions: Dict[str, Dict[str, np.ndarray]] = {}
        for m in policy_cfg.motions:
            mc = DictToClass(m)
            motion_name = mc.name
            mp = Path(mc.path)
            path = str(mp if mp.is_absolute() else (REAL_G1_ROOT / mp))
            t0, t1 = int(mc.start), int(mc.end)

            data = np.load(path, allow_pickle=True)
            if not isinstance(data, np.lib.npyio.NpzFile):
                raise ValueError(f"[TrackingPolicyRaw] Only .npz is supported: {path}")

            joint_pos = data["dof_pos"][t0:t1].astype(np.float32)
            root_pos = data["root_pos"][t0:t1].astype(np.float32)
            root_rot_xyzw = data["root_rot"][t0:t1].astype(np.float32)
            root_quat = np.concatenate([root_rot_xyzw[:, 3:4], root_rot_xyzw[:, :3]], axis=-1)

            joint_names = data.get("joint_names", None)
            if joint_names is None:
                raise ValueError(
                    f"[TrackingPolicyRaw] Motion '{motion_name}' is missing 'joint_names' in npz. "
                    "Please export joint_names with the dataset."
                )
            source_joint_names = []
            for n in joint_names.tolist():
                if isinstance(n, (bytes, np.bytes_)):
                    source_joint_names.append(n.decode("utf-8"))
                else:
                    source_joint_names.append(str(n))
            joint_pos = remap_joint_array_by_names(joint_pos, source_joint_names, self.obs_joint_names)

            self.motions[motion_name] = {
                "joint_pos": joint_pos,  # (T,J)
                "root_quat": root_quat,  # (T,4) wxyz
                "root_pos": root_pos,    # (T,3)
            }

        # ---- One-frame motion clips (config provided) ----------------------
        for m in policy_cfg.motion_clips:
            mc = DictToClass(m)
            motion_name = mc.name
            joint_pos_1 = np.asarray(mc.joint_pos, dtype=np.float32).reshape(1, -1)
            if joint_pos_1.shape[1] != len(self.dataset_joint_names):
                raise ValueError(
                    f"[TrackingPolicyRaw] Motion clip '{motion_name}' dim={joint_pos_1.shape[1]} "
                    f"does not match dataset_joint_names size={len(self.dataset_joint_names)}."
                )
            source_joint_names = self.dataset_joint_names
            joint_pos_1 = remap_joint_array_by_names(joint_pos_1, source_joint_names, self.obs_joint_names)
            root_quat_1 = np.asarray(mc.root_quat, dtype=np.float32).reshape(1, 4)
            root_pos_1 = np.asarray(mc.root_pos, dtype=np.float32).reshape(1, 3)

            self.motions[motion_name] = {
                "joint_pos": joint_pos_1,  # (1,J)
                "root_quat": root_quat_1,  # (1,4)
                "root_pos": root_pos_1,    # (1,3)
            }

        assert "default" in self.motions, "[TrackingPolicyRaw] motions must include a 'default' clip (length==1)."

        # ---- Reference stream ----------------------------------------------
        self.ref_joint_pos: Optional[np.ndarray] = None  # (T_ref, J)
        self.ref_root_quat: Optional[np.ndarray] = None  # (T_ref, 4)
        self.ref_root_pos: Optional[np.ndarray] = None   # (T_ref, 3)

        # ---- Playback state ------------------------------------------------
        self.ref_idx: int = 0
        self.ref_len: int = 0
        self.current_name: str = "default"
        self.current_done: bool = True  # boot: default done

        # ---- Misc ----------------------------------------------------------
        self.n_joints = len(self.obs_joint_names)

        # Optional UDP selector
        self._udp_server: Optional[MotionUDPServer] = None
        if self.udp_enable:
            try:
                self._udp_server = MotionUDPServer(self.udp_host, self.udp_port)
                self._udp_server.start()
            except Exception as e:
                print(f"[TrackingPolicyRaw] Failed to start UDP server: {e}")

        # Upright detector for "up" command (IMU姿态 + 膝盖角度 + 连续帧)
        self._upright_detector = UprightDetector(
            controller,
            threshold_deg=15.0,
            knee_threshold_rad=0.6,  # ≈34°，站立时膝盖可能稍微弯曲
            consecutive_frames=4
        )

        # Status feedback port
        self.status_port = int(getattr(policy_cfg, "status_port", 28563))

        super().__init__(name, policy_cfg, controller)
        self.init_count = 0
        self._fallen_during_recovery_count = 0
        self._recovery_retry_count = 0
        self._last_recovery_name = None

    def fade_in(self):
        super().fade_in()
        self._start_motion_from_current("default")

    def fade_out(self) -> float:
        self._start_motion_from_current("default")
        return super().fade_out()

    def deactivate(self):
        if self._udp_server is not None:
            self._udp_server.stop()
        self.ref_root_pos = None
        self.ref_root_quat = None
        self.ref_joint_pos = None
        super().deactivate()

    def _build_obs_modules(self):
        from observation import (
            TrackingCommandObsRaw,
            TargetRootZObs,
            TargetJointPosObs,
            TargetProjectedGravityBObs,
            RootAngVelB,
            ProjectedGravityB,
            JointPos,
            PrevActions,
            BootIndicator,
            ComplianceFlagObs,
        )
        self.obs_modules = [
            BootIndicator(),
            TrackingCommandObsRaw(self.controller, self),
            ComplianceFlagObs(self),
            TargetJointPosObs(self),
            TargetRootZObs(self),
            TargetProjectedGravityBObs(self),
            RootAngVelB(self.controller),
            ProjectedGravityB(self.controller),
            JointPos(self.controller, pos_steps=[0, 1, 2, 3, 4, 8]),
            PrevActions(self, steps=3),
        ]
        self.num_obs = sum(m.size for m in self.obs_modules)

    def request_motion(self, name: str) -> bool:
        if name not in self.motions:
            print(f"[TrackingPolicyRaw] Unknown motion '{name}'")
            return False
        if name == "default":
            # default can always be requested
            self._start_motion_from_current(name)
            return True
        elif self.current_name == "default" and self.current_done:
            self._start_motion_from_current(name)
            return True
        else:
            # Allow interruption of current motion
            print(f"[TrackingPolicyRaw] Interrupting '{self.current_name}' to start '{name}'")
            self._start_motion_from_current(name)
            return True

    def _send_motion_complete_notification(self):
        """通过UDP发送动作完成通知到text_to_motion.py"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(b"MOTION_COMPLETE", ("127.0.0.1", self.status_port))
            sock.close()
            print(f"[TrackingPolicyRaw] Status → MOTION_COMPLETE")
        except Exception:
            pass

    def _send_upright_success_notification(self):
        """通过UDP发送站起成功通知到text_to_motion.py"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(b"UPRIGHT_SUCCESS", ("127.0.0.1", self.status_port))
            sock.close()
            print("[TrackingPolicyRaw] Status → UPRIGHT_SUCCESS")
        except Exception as e:
            print(f"[TrackingPolicyRaw] Failed to send UPRIGHT_SUCCESS: {e}")

    def _load_generated_motion(self, filename: str) -> bool:
        """Dynamically load a generated NPZ file and register it as a motion."""
        generated_dir = REAL_G1_ROOT / "assets" / "data" / "generated"
        filepath = generated_dir / f"{filename}.npz"
        if not filepath.exists():
            filepath = REAL_G1_ROOT / "assets" / "data" / f"{filename}.npz"
            if not filepath.exists():
                print(f"[TrackingPolicyRaw] Generated file not found: {filename}")
                return False
        try:
            data = np.load(str(filepath), allow_pickle=True)
            if "dof_pos" in data:
                joint_pos = data["dof_pos"].astype(np.float32)
            elif "joint_pos" in data:
                joint_pos = data["joint_pos"].astype(np.float32)
            else:
                print(f"[TrackingPolicyRaw] NPZ must contain 'dof_pos' or 'joint_pos'")
                return False
            root_pos = data["root_pos"].astype(np.float32)
            root_rot_xyzw = data["root_rot"].astype(np.float32)
            root_quat = np.concatenate([root_rot_xyzw[:, 3:4], root_rot_xyzw[:, :3]], axis=-1)

            joint_names = data.get("joint_names", None)
            if joint_names is not None:
                source_joint_names = []
                for n in joint_names.tolist():
                    if isinstance(n, (bytes, np.bytes_)):
                        source_joint_names.append(n.decode("utf-8"))
                    else:
                        source_joint_names.append(str(n))
                joint_pos = remap_joint_array_by_names(joint_pos, source_joint_names, self.obs_joint_names)

            self.motions[filename] = {
                "joint_pos": joint_pos,
                "root_quat": root_quat,
                "root_pos": root_pos,
            }
            print(f"[TrackingPolicyRaw] Loaded generated motion '{filename}' | frames={joint_pos.shape[0]}")
            return True
        except Exception as e:
            print(f"[TrackingPolicyRaw] Failed to load generated motion '{filename}': {e}")
            import traceback
            traceback.print_exc()
            return False

    def _find_closest_recovery_state(self, motion_names: List[str]) -> Tuple[str, int]:
        """两阶段匹配：先按投影重力筛选姿态相近的候选帧，再从中选关节最接近的"""
        # 使用真实机器人 IMU 姿态
        real_quat = self.controller.quat.copy()  # wxyz, 真实IMU
        curr_q = R.from_quat(real_quat, scalar_first=True)
        curr_proj_grav = curr_q.inv().apply([0, 0, -1])  # (3,)

        curr_joints = self.controller.qj_isaac.copy().astype(np.float32)

        # 收集所有候选帧: (name, frame_idx, grav_dist, joint_dist)
        all_candidates = []

        for name in motion_names:
            if name not in self.motions:
                if not self._load_generated_motion(name):
                    continue
            if name not in self.motions:
                continue

            m = self.motions[name]
            T = m["joint_pos"].shape[0]

            # 每帧的投影重力（体坐标系）
            m_quats = R.from_quat(m["root_quat"], scalar_first=True)
            m_proj_grav = m_quats.inv().apply(np.tile([0, 0, -1], (T, 1)))  # (T, 3)

            # 重力方向距离
            grav_dist = np.sum((m_proj_grav - curr_proj_grav) ** 2, axis=1)

            # 关节距离
            joint_dist = np.sum((m["joint_pos"] - curr_joints) ** 2, axis=1)

            for i in range(T):
                all_candidates.append((name, i, grav_dist[i], joint_dist[i]))

        if len(all_candidates) == 0:
            if len(motion_names) > 0:
                return motion_names[0], 0
            return None, 0

        # 第一阶段：按重力距离排序
        all_candidates.sort(key=lambda x: x[2])
        best_grav_dist = all_candidates[0][2]
        
        # 不要固定的 20%，而是动态阈值：只保留跟最优重力匹配非常接近的帧
        # 允许的重力误差余量设置为 0.1（平方差），以确保姿态几乎一致
        grav_threshold = best_grav_dist + 0.1
        shortlist = [c for c in all_candidates if c[2] <= grav_threshold]
        
        # 至少保证有10个备选，如果连10个都凑不齐，硬塞到前10
        if len(shortlist) < 10:
            shortlist = all_candidates[:min(10, len(all_candidates))]

        print(f"  [筛选] 总候选帧={len(all_candidates)}, "
              f"重力最优={best_grav_dist:.3f}, "
              f"进入复选池={len(shortlist)} (阈值={grav_threshold:.3f})")

        # 第二阶段：从极度严格的重力候选中，按关节距离选最优
        shortlist.sort(key=lambda x: x[3])
        best_name, best_frame, best_grav, best_joint = shortlist[0]

        print(f"  → 选择: '{best_name}' frame {best_frame} "
              f"(grav={best_grav:.3f}, joints={best_joint:.3f})")
        return best_name, best_frame

    def update_obs(self):
        if self._udp_server is not None:
            for cmd in self._udp_server.pop_all():
                if cmd.startswith("LOAD:"):
                    filename = cmd[len("LOAD:"):].strip()
                    if filename not in self.motions:
                        self._load_generated_motion(filename)
                    if filename in self.motions:
                        self.request_motion(filename)
                    else:
                        print(f"[TrackingPolicyRaw] Cannot load '{filename}', skipping.")
                elif cmd.startswith("RECOVER:"):
                    names = cmd[len("RECOVER:"):].strip().split(";")
                    names = [n.strip() for n in names if n.strip()]
                    if len(names) > 0:
                        # 根据重试次数选择不同策略
                        if self._recovery_retry_count < 5:
                            # 前5次：用匹配算法
                            print(f"[TrackingPolicyRaw] Recovery attempt {self._recovery_retry_count}: "
                                  f"matching closest frame from {names}")
                            best_name, best_frame = self._find_closest_recovery_state(names)
                        else:
                            # 第6次起：轮换动作，从头开始
                            idx = self._recovery_retry_count % len(names)
                            best_name = names[idx]
                            # 确保动作已加载
                            if best_name not in self.motions:
                                self._load_generated_motion(best_name)
                            best_frame = 0
                            print(f"[TrackingPolicyRaw] Recovery attempt {self._recovery_retry_count}: "
                                  f"force '{best_name}' from frame 0 (cycling)")
                        
                        self._recovery_retry_count += 1
                        self._last_recovery_name = best_name
                        
                        if best_name is not None and best_name in self.motions:
                            self._start_motion_from_current(best_name, start_frame=best_frame)
                        else:
                            print("[TrackingPolicyRaw] Failed to find recovery state.")
                elif cmd == "START_UPRIGHT_MONITORING":
                    self._upright_detector.start_monitoring()
                elif cmd == "default":
                    self.request_motion("default")
                else:
                    self.request_motion(cmd)

        # 检查站起状态（如果正在监测）
        if self._upright_detector.is_monitoring:
            if self._upright_detector.check():
                self._send_upright_success_notification()
                self._upright_detector.stop_monitoring()
                self._fallen_during_recovery_count = 0
                self._recovery_retry_count = 0  # 成功站起，重置重试计数
                self._last_recovery_name = None
            else:
                # 站起模式中：检查跟踪目标是否已站立但机器人仍趴着
                # grace_frames 取 (transition+150) 和 70%动作长度 的较大值
                # 确保机器人有足够时间执行完翻身爬起
                grace_frames = max(self.transition_steps + 150,
                                   int(self.ref_len * 0.7))
                if (self.ref_root_quat is not None
                        and self.ref_idx >= grace_frames
                        and self.ref_idx < self.ref_len):
                    ref_q = R.from_quat(self.ref_root_quat[self.ref_idx], scalar_first=True)
                    ref_roll, ref_pitch, _ = ref_q.as_euler('xyz')
                    ref_is_upright = abs(ref_roll) < np.deg2rad(25) and abs(ref_pitch) < np.deg2rad(25)

                    if ref_is_upright:
                        curr_q = R.from_quat(self.controller.quat, scalar_first=True)
                        curr_roll, curr_pitch, _ = curr_q.as_euler('xyz')
                        robot_fallen = abs(curr_roll) > np.deg2rad(35) or abs(curr_pitch) > np.deg2rad(35)

                        if robot_fallen:
                            self._fallen_during_recovery_count += 1
                            if self._fallen_during_recovery_count >= 100:  # ~2秒@50Hz
                                print(f"[UprightDetector] 跟踪目标已站立但机器人仍趴着 "
                                      f"(roll={np.rad2deg(curr_roll):.1f}°, pitch={np.rad2deg(curr_pitch):.1f}°)")
                                print(f"  → 中断当前动作，触发重试")
                                self._fallen_during_recovery_count = 0
                                self.current_done = True
                                self._upright_detector.stop_monitoring()
                                self._send_motion_complete_notification()
                        else:
                            self._fallen_during_recovery_count = 0
                    else:
                        self._fallen_during_recovery_count = 0

        # 更新 ref_idx
        if self.ref_len > 0 and self.ref_idx < self.ref_len - 1:
            self.ref_idx += 1
            if self.ref_idx == self.ref_len - 1:
                self.current_done = True
                if self.current_name != "default":
                    if self._upright_detector.is_monitoring:
                        print("[UprightDetector] 恢复动作播放完毕但未站立 - 触发重试")
                        self._upright_detector.stop_monitoring()
                    self._send_motion_complete_notification()

        super().update_obs()


    def _read_current_state(self) -> Dict[str, np.ndarray]:
        q_policy = self.controller.qj_isaac.copy().astype(np.float32)

        if self.ref_root_pos is not None:
            root_pos = self.ref_root_pos[self.ref_idx]
            root_quat = self.ref_root_quat[self.ref_idx]
        else:
            root_pos = np.array([0.0, 0.0, 0.78], dtype=np.float32)
            root_quat = self.controller.quat.copy()
        return {
            "joint_pos": q_policy,
            "root_pos": root_pos,
            "root_quat": root_quat,
        }

    def _align_motion_to_current(
        self,
        motion: Dict[str, np.ndarray],
        curr: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        p0 = motion["root_pos"][0]
        q0_yaw = _yaw_component_wxyz(motion["root_quat"][0])
        pc = curr["root_pos"]
        qc_yaw = _yaw_component_wxyz(curr["root_quat"])

        R0 = R.from_quat(q0_yaw, scalar_first=True)
        Rc = R.from_quat(qc_yaw, scalar_first=True)
        R_delta = Rc * R0.inv()

        root_pos_aligned = R_delta.apply(motion["root_pos"] - p0) + pc
        root_pos_aligned[:, 2] = motion["root_pos"][:, 2]  # keep original z

        root_quat_all = R.from_quat(motion["root_quat"], scalar_first=True)
        root_quat_aligned = (R_delta * root_quat_all).as_quat(scalar_first=True)

        return {
            "joint_pos": motion["joint_pos"].astype(np.float32).copy(),
            "root_quat": root_quat_aligned.astype(np.float32),
            "root_pos": root_pos_aligned.astype(np.float32),
        }

    def _build_transition_prefix(
        self,
        curr: Dict[str, np.ndarray],
        tgt_first: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        T = int(self.transition_steps)
        if T <= 0:
            raise ValueError("[TrackingPolicyRaw] transition_steps must be > 0")

        joints_tr = _linspace_rows(curr["joint_pos"], tgt_first["joint_pos"], T)
        root_pos_tr = _linspace_rows(curr["root_pos"], tgt_first["root_pos"], T)
        root_quat_tr = _slerp(curr["root_quat"], tgt_first["root_quat"], T)

        return {
            "joint_pos": joints_tr,
            "root_quat": root_quat_tr,
            "root_pos": root_pos_tr,
        }

    def _start_motion_from_current(self, name: str, start_frame: int = 0):
        assert name in self.motions
        curr = self._read_current_state()

        m = self.motions[name]
        aligned_motion = self._align_motion_to_current(m, curr)

        max_frame = aligned_motion["joint_pos"].shape[0] - 1
        if start_frame > max_frame:
            start_frame = max_frame
        if start_frame < 0:
            start_frame = 0

        tgt_first = {
            "joint_pos": aligned_motion["joint_pos"][start_frame],
            "root_quat": aligned_motion["root_quat"][start_frame],
            "root_pos": aligned_motion["root_pos"][start_frame],
        }

        trans_motion = self._build_transition_prefix(curr, tgt_first)

        self.ref_joint_pos = np.concatenate([trans_motion["joint_pos"], aligned_motion["joint_pos"][start_frame:]], axis=0)
        self.ref_root_quat = np.concatenate([trans_motion["root_quat"], aligned_motion["root_quat"][start_frame:]], axis=0)
        self.ref_root_pos = np.concatenate([trans_motion["root_pos"], aligned_motion["root_pos"][start_frame:]], axis=0)

        self.ref_idx = 0
        self.ref_len = int(self.ref_joint_pos.shape[0])
        self.current_name = name
        self.current_done = (self.ref_len <= 1)

        print(f"[TrackingPolicyRaw] Start motion '{name}' from frame {start_frame} | ref_len={self.ref_len}, transition={self.transition_steps}")
