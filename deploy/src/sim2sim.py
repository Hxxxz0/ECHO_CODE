import os
import sys

import sys
import time
import argparse
import yaml
import struct
import threading
import signal
from pathlib import Path
from multiprocessing import Value

import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path

from sshkeyboard import listen_keyboard, stop_listening
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.utils.crc import CRC

from common.utils import DictToClass, Timer
from common.remote_controller import KeyMap
from common.joint_mapper import create_real_to_mujoco_mapper

from paths import ASSETS_DIR, to_assets_path

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

Keyboard2Joystick = {
    'a': 'A',
    's': 'start',
    'x': 'select',
    'u': 'up',
    'd': 'down'
}


class Sim2sim:
    def __init__(self, args, config):
        self.args = args
        self.config = config

        self.pub_freq = 200
        self.pub_dt = 1. / self.pub_freq
        self.low_level_freq = 500
        self.low_level_dt = 1. / self.low_level_freq

        # Initialize model and data
        # Resolve XML path (absolute or under assets)
        xml_candidate = Path(args.xml_path)
        if not xml_candidate.is_absolute():
            # try relative to ASSETS_DIR
            under_assets = ASSETS_DIR / xml_candidate
            model_path = os.path.abspath(str(under_assets if under_assets.exists() else xml_candidate))
        else:
            model_path = str(xml_candidate)
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.low_level_dt
        self.data = mujoco.MjData(self.model)

        self.ctrl_lower = self.model.actuator_ctrlrange[:, 0]
        self.ctrl_upper = self.model.actuator_ctrlrange[:, 1]
        # Initialize joint mapper for Real -> Mujoco conversion
        self.real_to_mujoco_mapper = create_real_to_mujoco_mapper(
            self.config.real_joint_names,
            self.config.mujoco_joint_names
        )
        mapping_info = self.real_to_mujoco_mapper.get_mapping_info()
        print(f"Real->Mujoco mapping: {mapping_info['mapped_joints']}/{mapping_info['from_space_size']} joints mapped")
        if mapping_info['unmapped_from_joints']:
            print(f"Unmapped Real joints: {mapping_info['unmapped_from_joints']}")
        if mapping_info['unmapped_to_joints']:
            print(f"Unmapped Mujoco joints: {mapping_info['unmapped_to_joints']}")

        # Set initial position
        self.data.qpos[7:] = self.real_to_mujoco_mapper.map_state_to_from(self.config.default_qpos_real)
        self.data.qvel[:] = 0.
        mujoco.mj_forward(self.model, self.data)

        # Low level commands (in Real space)
        self.__ptargets_real = np.zeros(len(self.config.real_joint_names))
        self.__kp_real = np.zeros(len(self.config.real_joint_names))
        self.__kd_real = np.zeros(len(self.config.real_joint_names))

        self.low_cmd = None
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.state_pub = ChannelPublisher(self.config.lowstate_topic, LowStateHG)
        self.state_pub.Init()
        self.state_pub_thread = threading.Thread(target=self.state_pub_handler, daemon=False)
        self.cmd_sub = ChannelSubscriber(self.config.lowcmd_topic, LowCmdHG)
        self.cmd_sub.Init(self.cmd_sub_handler)
        self.simulate_joystick_thread = threading.Thread(
            target=listen_keyboard,
            kwargs={"on_press": self.on_press, "on_release": self.on_release},
            daemon=False
        )
        self.is_alive = True
        self.policy_queried = False
        self.loop_count = Value('i', 0)

        self.render_gui = bool(getattr(self.config, "render_gui", False))

        # MuJoCo viewer
        self.viewer = None
        self.renderer = None
        self._viewer_tick = 0
        self.viewer_decim = max(1, self.low_level_freq // 30)  # Default 30 fps for viewer

        self.p_loop_rate = None

        # Interrupt handler for graceful exit
        signal.signal(signal.SIGINT, self.close)

    def count_loop_rate(self, loop_count):
        count_loop_timer = Timer(1.)
        while self.is_alive:
            print(f'Loop rate: {loop_count.value} Hz')
            loop_count.value = 0
            count_loop_timer.sleep()

    def on_press(self, key):
        print(f'Key pressed: {key}')
        joystick_btn = Keyboard2Joystick.get(key, None)
        if joystick_btn is None:
            return
        joystick_idx = getattr(KeyMap, joystick_btn, None)
        if joystick_idx is None:
            return
        self.low_state.wireless_remote[0] = joystick_idx
        self.state_pub.Write(self.low_state)

    def on_release(self, key):
        time.sleep(0.1)
        self.low_state.wireless_remote[0] = 0
        self.state_pub.Write(self.low_state)

    def cmd_sub_handler(self, msg):
        self.low_cmd = msg
        self.policy_queried |= self.low_cmd.reserve[0]
        for i in range(len(self.config.real_joint_names)):
            self.__ptargets_real[i] = msg.motor_cmd[i].q
            self.__kp_real[i] = msg.motor_cmd[i].kp
            self.__kd_real[i] = msg.motor_cmd[i].kd

    def state_pub_handler(self):
        timer = Timer(self.pub_dt)
        while self.is_alive:
            low_state = self.low_state
            joint_qpos_mujoco = self.data.qpos[7:]  # Skip base pose
            joint_qvel_mujoco = self.data.qvel[6:]  # Skip base velocity
            joint_torque_mujoco = self.data.ctrl.copy()

            joint_qpos_real = self.real_to_mujoco_mapper.map_state_to_from(joint_qpos_mujoco)
            joint_qvel_real = self.real_to_mujoco_mapper.map_state_to_from(joint_qvel_mujoco)
            joint_torque_real = self.real_to_mujoco_mapper.map_state_to_from(joint_torque_mujoco)

            for i in range(len(self.config.real_joint_names)):
                low_state.motor_state[i].q = joint_qpos_real[i]
                low_state.motor_state[i].dq = joint_qvel_real[i]
                low_state.motor_state[i].tau_est = joint_torque_real[i]

            low_state.imu_state.quaternion = self.data.qpos[3:7].copy() # Mujoco is wxyz
            low_state.imu_state.gyroscope = self.data.qvel[3:6].copy()
            low_state.tick = 1
            low_state.crc = CRC().Crc(low_state)
            self.state_pub.Write(low_state)

            timer.sleep()

    def wait_for_high_cmd(self):
        print(f'Waiting for high level controller...')
        while self.low_cmd is None:
            continue
        print(f'Connected to high level')
        print(f'Press "s" to move to default pose')
        running_zero_cmd = True
        while running_zero_cmd:
            running_zero_cmd = self.low_state.wireless_remote[0] != KeyMap.start

    def simulate_gantry(self):
        print(
            f'''Moving to default pose...\n'''
            f'''Press "a" after the robot is in default pose to being control loop'''
        )
        timer = Timer(self.low_level_dt)
        while True:
            ptargets_mujoco = self.real_to_mujoco_mapper.map_action_from_to(self.__ptargets_real)
            # gantry pose
            self.data.qpos[:7] = [0, 0, 2, 1.0, 0.0, 0.0, 0.0]
            self.data.qpos[7:] = ptargets_mujoco
            mujoco.mj_forward(self.model, self.data)

            if not self._viewer_sync():
                break

            running_default_pos = self.low_state.wireless_remote[0] != KeyMap.A and self.low_state.wireless_remote[0] != KeyMap.B and self.low_state.wireless_remote[0] != KeyMap.X
            if not running_default_pos:
                break
            timer.sleep()

    def simulate_control(self):
        print(f'Running control loop...')
        self.data.qpos[2] = 0.78
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # Neutral orientation
        mujoco.mj_forward(self.model, self.data)

        timer = Timer(self.low_level_dt)
        time_start = time.time()
        loop_count = self.loop_count

        while self.is_alive:
            if not self.policy_queried:
                timer.sleep()
                time_start = time.time()
                continue

            ptargets_mujoco = self.real_to_mujoco_mapper.map_action_from_to(self.__ptargets_real)
            kp_mujoco = self.real_to_mujoco_mapper.map_action_from_to(self.__kp_real)
            kd_mujoco = self.real_to_mujoco_mapper.map_action_from_to(self.__kd_real)

            qpos_mujoco = self.data.qpos[7:]
            qvel_mujoco = self.data.qvel[6:]
            ctrl = kp_mujoco * (ptargets_mujoco - qpos_mujoco) + kd_mujoco * (0 - qvel_mujoco)
            ctrl = np.clip(ctrl, self.ctrl_lower, self.ctrl_upper)
            self.data.ctrl[:] = ctrl

            seconds = loop_count.value * self.low_level_dt
            
            # Limit external forces applied via viewer to 30N
            self._limit_external_forces(max_force=30.0)
            
            mujoco.mj_step(self.model, self.data)

            if not self._viewer_sync():
                break

            if loop_count.value % 200 == 0:
                seconds_real = time.time() - time_start
                print(f'Time: {seconds:.2f}, Time real: {seconds_real:.2f}, Height: {self.data.qpos[2]:.2f}')

            loop_count.value += 1
            timer.sleep()

        self.close()

    def _limit_external_forces(self, max_force=30.0):
        for i in range(self.model.nbody):
            # Get the force vector (first 3 components)
            force = self.data.xfrc_applied[i, :3]
            force_magnitude = np.linalg.norm(force)
            
            # If force exceeds max_force, clip it
            if force_magnitude > max_force:
                self.data.xfrc_applied[i, :3] = force * (max_force / force_magnitude)

    def _viewer_sync(self) -> bool:
        if self.viewer is None:
            return True
        if not self.viewer.is_running():
            self.is_alive = False
            return False
        self._viewer_tick += 1
        if (self._viewer_tick % self.viewer_decim) == 0:
            self.viewer.sync()
        return True

    def run(self):
        self.state_pub_thread.start()
        self.simulate_joystick_thread.start()

        if self.render_gui:
            self.renderer = mujoco.Renderer(self.model)

            with mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
            ) as viewer:
                self.viewer = viewer
                try:
                    self.wait_for_high_cmd()
                    self.simulate_gantry()
                    self.simulate_control()
                finally:
                    self.viewer = None
        else:
            self.wait_for_high_cmd()
            self.simulate_gantry()
            self.simulate_control()

    def close(self, *args):
        if not self.is_alive:
            return
        self.is_alive = False

        self.on_press('x')
        stop_listening()
        if self.p_loop_rate is not None:
            self.p_loop_rate.terminate()

        self.state_pub_thread.join()
        self.simulate_joystick_thread.join()
        sys.exit(0)


if __name__ == "__main__":
    try:
        import multiprocessing as mp
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method('spawn', force=True)
    except Exception:
        pass

    parser = argparse.ArgumentParser()
    default_xml = ASSETS_DIR / "g1" / "g1.xml"
    parser.add_argument("--xml_path", type=str, default=str(default_xml))
    args = parser.parse_args()

    config_path = Path(ASSETS_DIR).parents[0] / "config" / "controller.yaml"
    config = DictToClass(yaml.load(open(str(config_path), 'r'), Loader=yaml.FullLoader))
    ChannelFactoryInitialize(0, 'lo') #"lo" means local network.

    Sim2sim(args, config).run()
