import socket
import threading
from collections import deque
from typing import Dict, List, Optional, Tuple

import linuxfd
import select


class DictToClass:
    def __init__(self, data_dict):
        for key, value in data_dict.items():
            setattr(self, key, value)


class Timer(object):
    '''Timer class for accurate loop rate control
    This class does not use Python's built-in thread timing control
    or management. Only use this class on Linux platforms.
    '''
    def __init__(self, interval: float) -> None:
        self.__epl, self.__tfd = self.__create_timerfd(interval)

    @staticmethod
    def __create_timerfd(interval: float):
        '''Produces a timerfd file descriptor from the kernel
        '''
        tfd = linuxfd.timerfd(rtc=True, nonBlocking=True)
        tfd.settime(interval, interval)
        epl = select.epoll()
        epl.register(tfd.fileno(), select.EPOLLIN)
        return epl, tfd

    def sleep(self) -> None:
        '''Blocks the thread holding this func until the next time point
        '''
        events = self.__epl.poll(-1)
        for fd, event in events:
            if fd == self.__tfd.fileno() and event & select.EPOLLIN:
                self.__tfd.read()

# =========================================
# Tiny non-blocking UDP command server
# =========================================
class MotionUDPServer(threading.Thread):
    """Very small UDP server; each datagram is a motion name string."""
    def __init__(self, host: str = "127.0.0.1", port: int = 28562):
        super().__init__(daemon=True)
        self._host = host
        self._port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((self._host, self._port))
        self._sock.settimeout(0.2)
        self._q = deque()
        self._lock = threading.Lock()
        self._running = True
        print(f"[MotionUDPServer] Listening on udp://{host}:{port}")

    def run(self):
        while self._running:
            try:
                data, _ = self._sock.recvfrom(1024)
                name = data.decode("utf-8", errors="ignore").strip()
                if name:
                    with self._lock:
                        self._q.append(name)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[MotionUDPServer] Error: {e}")

    def stop(self):
        self._running = False
        try:
            self._sock.close()
        except Exception:
            pass

    def pop_all(self) -> List[str]:
        with self._lock:
            items = list(self._q)
            self._q.clear()
        return items

joint_names_29 = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint", "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]

joint_names_23 = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint", "waist_yaw_joint", "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint"]

body_names_29 = ["pelvis", "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link", "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link", "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link", "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link", "torso_link", "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link", "left_elbow_link", "left_wrist_roll_link", "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_roll_link", "head_mimic", "left_hand_mimic", "right_hand_mimic"]

body_names_23 = ["world", "pelvis", "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link", "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link", "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link", "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link", "torso_link", "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link", "left_elbow_link", "left_wrist_roll_rubber_hand", "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_roll_rubber_hand", "head_mimic", "left_hand_mimic", "right_hand_mimic"]
