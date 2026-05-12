#!/usr/bin/env python3
"""
文本生成动作交互工具
通过WebSocket连接到StableMoFusion服务器，将文本描述转换为机器人动作
"""

import asyncio
import argparse
import socket
import json
import io
import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from scipy.spatial.transform import Rotation as R
import yaml

# 导入路径配置
from paths import REAL_G1_ROOT

# 从convert_simple_to_deploy.py复用的配置
# Isaac关节顺序（左右交替）
ISAAC_JOINT_ORDER = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]

# MT关节顺序（部署格式要求，按部位分组）
MT_JOINT_ORDER = [
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 
    'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 
    'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 
    'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 
    'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint'
]

# 创建Isaac到MT的映射
def create_isaac_to_mt_mapping():
    """创建从Isaac顺序到MT顺序的索引映射"""
    mapping = []
    for mt_joint in MT_JOINT_ORDER:
        if mt_joint in ISAAC_JOINT_ORDER:
            mapping.append(ISAAC_JOINT_ORDER.index(mt_joint))
        else:
            raise ValueError(f"MT关节 '{mt_joint}' 在Isaac顺序中未找到!")
    return np.array(mapping)

ISAAC_TO_MT_MAP = create_isaac_to_mt_mapping()

BANNER = """\
=== 文本生成动作工具 ===
  输入文本描述 → 生成并加载动作
  - 动作执行中可随时输入新文本切换动作
  - 动作完成后自动切换到default姿态

命令:
  <文本描述>     - 生成新动作
  up             - 站起并自动恢复（摔倒后使用）
  default        - 手动回到默认姿态
  kf list        - 显示本地Kungfu动作列表
  kf <名称或编号> - 加载指定的Kungfu动作并转换执行
  last           - 重新加载上一个生成的动作
  list           - 显示所有已生成的动作
  clear          - 清理旧的生成文件
  status         - 显示当前动作状态
  tunnel         - 显示SSH隧道配置帮助
  q/quit         - 退出
========================
"""


class MotionStatusListener(threading.Thread):
    """监听来自policy.py的动作完成和站起成功通知"""
    
    def __init__(self, port: int = 28563):
        super().__init__(daemon=True)
        self.port = port
        self._sock = None
        self._running = True
        self._motion_complete_callback = None
        self._upright_success_callback = None
        
    def set_callback(self, callback):
        """设置动作完成回调函数"""
        self._motion_complete_callback = callback
    
    def set_upright_callback(self, callback):
        """设置站起成功回调函数"""
        self._upright_success_callback = callback
    
    def run(self):
        """监听线程主循环"""
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.bind(("127.0.0.1", self.port))
            self._sock.settimeout(1.0)
            print(f"[StatusListener] 监听动作状态 udp://127.0.0.1:{self.port}")
        except Exception as e:
            print(f"[StatusListener] 无法绑定端口 {self.port}: {e}")
            return
        
        while self._running:
            try:
                data, _ = self._sock.recvfrom(1024)
                msg = data.decode("utf-8", errors="ignore").strip()
                if msg == "MOTION_COMPLETE" and self._motion_complete_callback:
                    self._motion_complete_callback()
                elif msg == "UPRIGHT_SUCCESS" and self._upright_success_callback:
                    self._upright_success_callback()
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    print(f"[StatusListener] 错误: {e}")
    
    def stop(self):
        """停止监听"""
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except:
                pass


class TextToMotionClient:
    """文本生成动作客户端"""
    
    def __init__(self, config_path: str = "config/tracking.yaml"):
        # 加载配置
        config_path_obj = Path(config_path)
        if not config_path_obj.is_absolute():
            config_path_obj = REAL_G1_ROOT / config_path_obj
        
        with open(str(config_path_obj), 'r') as f:
            cfg = yaml.safe_load(f)
        
        self.config = cfg.get('text_to_motion', {})
        
        # WebSocket配置
        self.ws_host = self.config.get('ws_host', '127.0.0.1')
        self.ws_port = self.config.get('ws_port', 8000)
        self.ws_uri = f"ws://{self.ws_host}:{self.ws_port}/ws"
        
        # 远程服务器配置（用于提示）
        self.remote_server = self.config.get('remote_server', {})
        
        # 生成参数
        self.default_motion_length = self.config.get('default_motion_length', 4.0)
        self.default_inference_steps = self.config.get('default_inference_steps', 10)
        self.adaptive_smooth = self.config.get('adaptive_smooth', True)
        self.static_start = self.config.get('static_start', True)
        self.static_frames = self.config.get('static_frames', 2)
        self.blend_frames = self.config.get('blend_frames', 8)
        
        # 自动default切换
        self.auto_default = self.config.get('auto_default_on_complete', True)
        
        # UDP配置
        self.udp_host = "127.0.0.1"
        self.udp_port = 28562
        self.status_port = 28563
        
        # 文件管理
        self.generated_dir = REAL_G1_ROOT / "assets/data/generated"
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        
        self.kungfu_dir = REAL_G1_ROOT / "assets/data/kungfu"
        self.kungfu_dir.mkdir(parents=True, exist_ok=True)
        
        # 状态
        self.last_generated = None
        self.current_status = "空闲"
        self._is_up_mode = False  # 是否在站起模式
        
        # 状态监听器
        self.status_listener = MotionStatusListener(self.status_port)
        self.status_listener.set_callback(self._on_motion_complete)
        self.status_listener.start()
        
        # UDP套接字
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    def _on_motion_complete(self):
        """动作完成回调"""
        print(f"\n[完成] 动作执行完毕")
        
        # 如果在站起模式但动作完成时仍未站起，显示警告并重试
        if self._is_up_mode:
            print("[警告] 站起动作已完成，但机器人仍未检测到直立姿态")
            print("       正在自动根据当前姿态重试站立恢复指令...")
            
            def _retry_recover():
                print("       准备阶段：先发送default短暂放松关节(0.5秒)...")
                self._send_udp_command("default")
                time.sleep(0.5)
                self._send_udp_command("RECOVER:fallAndGetUp1_subject1;fallAndGetUp2_subject2")
                self._send_udp_command("START_UPRIGHT_MONITORING")
                
            threading.Thread(target=_retry_recover, daemon=True).start()
        elif self.auto_default:
            print("[自动切换] 发送default命令...")
            self._send_udp_command("default")
            self.current_status = "空闲"
    
    def _on_upright_success(self):
        """站起成功回调"""
        print(f"\n[成功] 检测到机器人已站起！")
        print("[自动切换] 发送default命令...")
        self._send_udp_command("default")
        self._is_up_mode = False
        self.current_status = "空闲"
    
    def _send_udp_command(self, command: str) -> bool:
        """发送UDP命令到deploy.py"""
        try:
            self.udp_sock.sendto(command.encode("utf-8"), (self.udp_host, self.udp_port))
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] 发送命令 '{command}' 到 udp://{self.udp_host}:{self.udp_port}")
            return True
        except Exception as e:
            print(f"[错误] UDP发送失败: {e}")
            return False
    
    def convert_38d_to_deploy(self, npz_bytes: bytes) -> Dict[str, np.ndarray]:
        """
        将38D格式NPZ转换为部署格式
        38D格式（服务器返回，Isaac顺序）:
          - fps: (1,) int32
          - joint_pos: (T, 29) float32 [Isaac顺序：左右交替]
          - root_pos: (T, 3) float32
          - root_rot: (T, 4) float32 [w,x,y,z]
        
        部署格式（policy.py需要）:
          - fps: float32
          - dof_pos: (T, 29) float32 [MT顺序：按部位分组]
          - root_pos: (T, 3) float32
          - root_rot: (T, 4) float32 [x,y,z,w]
          - joint_names: array of strings
        """
        # 解析NPZ
        data = np.load(io.BytesIO(npz_bytes))
        
        # 检查是否已经是部署格式（比如从generated拷回来的，包含了 dof_pos）
        if 'dof_pos' in data:
            print(f"  [转换] 检测到已经是部署目标格式，直接加载 ✓")
            fps = int(data['fps'].item() if isinstance(data['fps'], np.ndarray) else data['fps'])
            return {
                'fps': np.float32(fps),
                'dof_pos': data['dof_pos'].astype(np.float32),
                'root_pos': data['root_pos'].astype(np.float32),
                'root_rot': data['root_rot'].astype(np.float32),
                'joint_names': data['joint_names'] if 'joint_names' in data else np.array(MT_JOINT_ORDER, dtype='<U26')
            }
        
        # 提取数据，兼容 text2motion格式(root_pos) 和 Isaac格式(body_pos_w)
        fps = int(data['fps'].item() if isinstance(data['fps'], np.ndarray) else data['fps'])
        joint_pos_isaac = data['joint_pos'].astype(np.float32)  # Isaac顺序
        
        if 'root_pos' in data:
            root_pos = data['root_pos'].astype(np.float32)
            root_rot_wxyz = data['root_rot'].astype(np.float32)  # [w,x,y,z]
        elif 'body_pos_w' in data:
            # 提取 pelvis (index 0)
            root_pos = data['body_pos_w'][:, 0].astype(np.float32)
            root_rot_wxyz = data['body_quat_w'][:, 0].astype(np.float32)  # [w,x,y,z]
        else:
            raise KeyError("NPZ文件中找不到 root_pos 或 body_pos_w 键")
        
        print(f"  [转换] 帧数: {joint_pos_isaac.shape[0]}, FPS: {fps}")
        
        # 验证旋转格式：如果是Isaac生成的数据，可能不一定是 wxyz，这里统一当作wxyz转换?
        # 注意：无论是text2motion还是Isaac原始输出，四元数通常都是 wxyz。
        # 转换root_rot为xyzw格式 (供policy使用)
        root_rot_xyzw = np.concatenate([root_rot_wxyz[:, 1:4], root_rot_wxyz[:, 0:1]], axis=-1)
        
        # 验证quaternion
        try:
            R.from_quat(root_rot_xyzw[0])
            print(f"  [转换] root_rot格式: wxyz → xyzw ✓")
        except Exception as e:
            raise ValueError(f"root_rot转换失败: {e}")
        
        # 关键修正：从Isaac顺序重排到MT顺序
        dof_pos = joint_pos_isaac[:, ISAAC_TO_MT_MAP]
        print(f"  [转换] 关节顺序: Isaac → MT (重映射) ✓")
        
        # 构建部署格式
        deploy_data = {
            'fps': np.float32(fps),
            'dof_pos': dof_pos,
            'root_pos': root_pos,
            'root_rot': root_rot_xyzw,
            'joint_names': np.array(MT_JOINT_ORDER, dtype='<U26'),
        }
        
        return deploy_data
    
    async def generate_motion(self, text: str, motion_length: Optional[float] = None,
                             inference_steps: Optional[int] = None) -> Optional[str]:
        """
        生成动作并保存
        返回生成的文件名（不含路径和扩展名）
        """
        if motion_length is None:
            motion_length = self.default_motion_length
        if inference_steps is None:
            inference_steps = self.default_inference_steps
        
        print(f"\n[生成中] 文本: '{text}'")
        print(f"          长度: {motion_length}s, 推理步数: {inference_steps}")
        self.current_status = "生成中"
        
        try:
            # 动态导入websockets（避免启动时就报错）
            try:
                import websockets
            except ImportError:
                print("[错误] 缺少 websockets 库，请运行: pip install websockets")
                return None
            
            # 连接WebSocket
            try:
                ws = await websockets.connect(self.ws_uri, max_size=50*1024*1024, open_timeout=10)
            except Exception as e:
                print(f"[错误] WebSocket连接失败: {e}")
                print("\n提示: 请确保SSH隧道已建立")
                self._print_tunnel_help()
                return None
            
            # 准备请求
            request = {
                "text": text,
                "motion_length": motion_length,
                "num_inference_steps": inference_steps,
                "seed": int(time.time()) % 10000,  # 使用时间戳作为种子
                "adaptive_smooth": self.adaptive_smooth,
                "static_start": self.static_start,
                "static_frames": self.static_frames,
                "blend_frames": self.blend_frames,
            }
            
            # 发送请求
            await ws.send(json.dumps(request))
            print("[生成中] 等待服务器响应...")
            
            # 接收响应
            response = await ws.recv()
            await ws.close()
            
            # 检查是否为错误
            if isinstance(response, str):
                error = json.loads(response)
                print(f"[错误] 服务器返回错误: {error.get('error', 'Unknown error')}")
                return None
            
            # 解析NPZ
            print("[转换中] 解析动作数据...")
            deploy_data = self.convert_38d_to_deploy(response)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gen_{timestamp}"
            filepath = self.generated_dir / f"{filename}.npz"
            
            # 保存文件
            np.savez(str(filepath), **deploy_data)
            print(f"[保存] {filepath}")
            
            # 记录
            self.last_generated = filename
            self.current_status = "加载中"
            
            return filename
            
        except Exception as e:
            print(f"[错误] 生成失败: {e}")
            import traceback
            traceback.print_exc()
            self.current_status = "错误"
            return None
    
    def load_motion(self, filename: str) -> bool:
        """加载动作到policy"""
        print(f"\n[加载中] {filename}")
        success = self._send_udp_command(f"LOAD:{filename}")
        if success:
            self.current_status = "执行中"
        return success
    
    def list_kungfu_motions(self):
        """列出所有本地Kungfu动作"""
        if not self.kungfu_dir.exists():
            print("\nKungfu目录不存在!")
            return
            
        files = sorted(self.kungfu_dir.glob("*.npz"))
        if not files:
            print(f"\n暂无Kungfu动作 ({self.kungfu_dir})")
            return
        
        print(f"\n=== 本地Kungfu动作 ({len(files)}个) ===")
        for i, f in enumerate(files, 1):
            print(f"  [{i}] {f.stem}")
            
    def process_kungfu_motion(self, user_input: str) -> Optional[str]:
        """处理并在本地转换Kungfu动作"""
        if not self.kungfu_dir.exists():
            print("[错误] Kungfu目录不存在")
            return None
            
        files = sorted(self.kungfu_dir.glob("*.npz"))
        if not files:
            print("[错误] 没有找到Kungfu动作文件")
            return None
            
        target_file = None
        # 如果输入是数字，按索引加载
        if user_input.isdigit():
            idx = int(user_input) - 1
            if 0 <= idx < len(files):
                target_file = files[idx]
        else:
            # 尝试精准匹配
            for f in files:
                if f.stem.lower() == user_input.lower():
                    target_file = f
                    break
            # 如果没找到完全匹配，则部分匹配
            if target_file is None:
                for f in files:
                    if user_input.lower() in f.stem.lower():
                        target_file = f
                        break
                    
        if target_file is None:
            print(f"[错误] 未找到匹配的Kungfu动作: '{user_input}'")
            return None
            
        print(f"\n[处理中] 本地Kungfu动作: '{target_file.name}'")
        self.current_status = "处理中"
        
        try:
            with open(target_file, 'rb') as f:
                npz_bytes = f.read()
                
            print("[转换中] 解析Kungfu动作数据...")
            deploy_data = self.convert_38d_to_deploy(npz_bytes)
            
            # 生成文件名 (前缀带kf)
            output_name = f"kf_{target_file.stem}"
            output_path = self.generated_dir / f"{output_name}.npz"
            
            # 保存转换后的文件到 generated 目录
            np.savez(str(output_path), **deploy_data)
            print(f"[保存] {output_path}")
            
            self.last_generated = output_name
            self.current_status = "加载中"
            
            return output_name
            
        except Exception as e:
            print(f"[错误] Kungfu动作处理失败: {e}")
            import traceback
            traceback.print_exc()
            self.current_status = "错误"
            return None

    def list_generated_motions(self):
        """列出所有已生成的动作"""
        files = sorted(self.generated_dir.glob("gen_*.npz"))
        if not files:
            print("\n暂无生成的动作")
            return
        
        print(f"\n=== 已生成的动作 ({len(files)}个) ===")
        for i, f in enumerate(files, 1):
            # 解析时间戳
            name = f.stem
            try:
                ts = name.replace("gen_", "")
                dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
                time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                time_str = "未知时间"
            
            print(f"  {i}. {name} ({time_str})")
        print("=" * 40)
    
    def clear_old_motions(self, keep_last: int = 10):
        """清理旧的生成文件"""
        files = sorted(self.generated_dir.glob("gen_*.npz"))
        if len(files) <= keep_last:
            print(f"\n已生成动作数量 ({len(files)}) 未超过保留数量 ({keep_last})")
            return
        
        to_delete = files[:-keep_last]
        print(f"\n[清理] 删除 {len(to_delete)} 个旧文件...")
        for f in to_delete:
            try:
                f.unlink()
                print(f"  删除: {f.name}")
            except Exception as e:
                print(f"  删除失败: {f.name} - {e}")
        print(f"保留最近 {keep_last} 个文件")
    
    def show_status(self):
        """显示当前状态"""
        print(f"\n=== 当前状态 ===")
        print(f"状态: {self.current_status}")
        print(f"上次生成: {self.last_generated or '无'}")
        print(f"自动default: {'开启' if self.auto_default else '关闭'}")
        print(f"WebSocket: {self.ws_uri}")
        print(f"UDP控制: {self.udp_host}:{self.udp_port}")
        print(f"状态监听: 127.0.0.1:{self.status_port}")
        print("=" * 20)
    
    def _print_tunnel_help(self):
        """打印SSH隧道帮助"""
        if not self.remote_server:
            return
        
        host = self.remote_server.get('host', '')
        port = self.remote_server.get('port', 22)
        user = self.remote_server.get('user', 'root')
        alias = self.remote_server.get('ssh_alias', '')
        
        print("\n=== SSH隧道配置 ===")
        print("建立隧道命令:")
        if alias:
            print(f"  ssh -L 8000:127.0.0.1:8000 {alias}")
        if host:
            print(f"  ssh -L 8000:127.0.0.1:8000 {user}@{host} -p {port}")
        print("\n验证隧道:")
        print("  curl http://127.0.0.1:8000/")
        print("=" * 25)
    
    def stop(self):
        """停止客户端"""
        self.status_listener.stop()
        self.udp_sock.close()


async def interactive_loop(client: TextToMotionClient):
    """交互式主循环"""
    print(BANNER)
    
    # 检查WebSocket连接
    print("[启动] 检查WebSocket服务器连接...")
    try:
        import websockets
        ws = await websockets.connect(client.ws_uri, open_timeout=5)
        await ws.close()
        print("[成功] WebSocket服务器连接正常")
    except ImportError:
        print("[警告] 缺少 websockets 库")
        print("       请运行: pip install websockets")
    except Exception as e:
        print(f"[警告] WebSocket连接失败: {e}")
        client._print_tunnel_help()
    
    print("\n准备就绪！请输入文本描述或命令...\n")
    
    while True:
        try:
            # 获取用户输入
            user_input = input(f"[{client.current_status}] > ").strip()
            
            if not user_input:
                continue
            
            # 处理命令
            cmd_lower = user_input.lower()
            
            if cmd_lower in ('q', 'quit', 'exit'):
                print("退出...")
                break
            
            elif cmd_lower == 'default':
                client._send_udp_command('default')
                client.current_status = "空闲"
                client._is_up_mode = False
            
            elif cmd_lower == 'up':
                print("[站起] 准备阶段：发送default短暂放松关节(0.5秒)...")
                client._send_udp_command('default')
                await asyncio.sleep(0.5)

                print("[站起] 开始站起流程...")
                client._is_up_mode = True
                client.current_status = "站起中"
                # 设置站起成功回调
                client.status_listener.set_upright_callback(client._on_upright_success)
                # 发送加载站起动作命令（底层会自动插值过渡）
                client._send_udp_command("RECOVER:fallAndGetUp1_subject1;fallAndGetUp2_subject2")
                # 发送开始监测命令
                client._send_udp_command("START_UPRIGHT_MONITORING")
            
            elif cmd_lower == 'last':
                if client.last_generated:
                    client.load_motion(client.last_generated)
                else:
                    print("还没有生成过动作")
            
            elif cmd_lower == 'list':
                client.list_generated_motions()
            
            elif cmd_lower == 'clear':
                client.clear_old_motions(keep_last=10)
            
            elif cmd_lower == 'status':
                client.show_status()
            
            elif cmd_lower == 'tunnel':
                client._print_tunnel_help()
            
            elif cmd_lower == 'kf list':
                client.list_kungfu_motions()
                
            elif cmd_lower.startswith('kf '):
                kf_target = user_input[3:].strip()
                if kf_target:
                    filename = client.process_kungfu_motion(kf_target)
                    if filename:
                        client.load_motion(filename)
                else:
                    print("用法: kf <名称或编号>")
            
            elif cmd_lower.startswith('?') or cmd_lower == 'help':
                print(BANNER)
            
            else:
                # 当作文本描述处理
                filename = await client.generate_motion(user_input)
                if filename:
                    client.load_motion(filename)
        
        except KeyboardInterrupt:
            print("\n\nCtrl-C 退出...")
            break
        except EOFError:
            print("\nEOF 退出...")
            break
        except Exception as e:
            print(f"[错误] {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="文本生成动作交互工具",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/tracking.yaml",
                       help="配置文件路径")
    args = parser.parse_args()
    
    try:
        client = TextToMotionClient(args.config)
        asyncio.run(interactive_loop(client))
    except KeyboardInterrupt:
        print("\n退出")
    finally:
        try:
            client.stop()
        except:
            pass


if __name__ == "__main__":
    main()

