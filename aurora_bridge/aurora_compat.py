"""
aurora_compat.py —— 让 pip 版 fourier_aurora_client 0.1.8 正确连接 GR-1P (gr1t2) v1.2.1 的 AuroraCore
=====================================================================================================
在 GR-1P v1.2.1 镜像里 `pip install fourier_aurora_client==0.1.8` 后直接用，会遇到四个兼容问题：

1. 消息定义不一致：0.1.8 自带的 fourier_msgs 里 VelocityCmd 多了 `duration` 字段，
   与 AuroraCore v1.2.1 的定义不同 → DDS 类型不匹配，速度指令根本发不到机器人
   （现象：进入行走状态后发 set_velocity，机器人原地不动）。
   解决：改用镜像自带、与 AuroraCore 同版本编译的消息库
   /opt/fftai/fourier_dds_msgs/lib/python3.10/site-packages/fourier_msgs。
2. v1.2.1 不发布 `robot_motor_cfg_group_state`，而 client 初始化要求所有订阅都匹配。
3. v1.2.1 没有 policy_change_service（client 初始化会等它匹配）。
4. client 初始化要求所有话题都已匹配，但只等 5×0.5 s；容器里 DDS 发现较慢，经常超时失败。
   解决：等待放宽到 5×2 s，指令类话题不参与“全部匹配”检查（匹配后照常收发）。

用法：用 `from aurora_bridge.aurora_compat import AuroraClient` 代替 `from fourier_aurora_client import AuroraClient`，
其余 API 与官方一致。需要在镜像环境（/env.sh 的 LD_LIBRARY_PATH/PYTHONPATH）下运行。
"""
import os
import sys
import time as _time
import types

IMAGE_MSGS = "/opt/fftai/fourier_dds_msgs/lib/python3.10/site-packages"

# ---------- 1. 用镜像自带的消息定义 ----------
if os.path.isdir(os.path.join(IMAGE_MSGS, "fourier_msgs")):
    _pkg = types.ModuleType("fourier_msgs")
    _pkg.__path__ = [os.path.join(IMAGE_MSGS, "fourier_msgs")]
    sys.modules["fourier_msgs"] = _pkg
    # 镜像里的 PolicyService 在 fourier_srvs 包下；client 只在策略切换服务里用到它
    _srv = types.ModuleType("fourier_msgs.srv")
    _srv.__path__ = []
    sys.modules["fourier_msgs.srv"] = _srv
    if IMAGE_MSGS not in sys.path:
        sys.path.insert(0, IMAGE_MSGS)
    try:
        import fourier_srvs.srv.PolicyService as _ps
    except Exception:          # noqa: BLE001
        _ps = types.ModuleType("fourier_msgs.srv.PolicyService")
    sys.modules["fourier_msgs.srv.PolicyService"] = _ps
    _srv.PolicyService = _ps

from fourier_aurora_client import dds_interface as _ddsi  # noqa: E402

# ---------- 2/4. 可选话题不参与“全部匹配”检查 ----------
OPTIONAL_TOPICS = {"robot_motor_cfg_group_state"}
OPTIONAL_PUB_TOPICS = {"velocity_cmd", "robot_motor_cfg_group_cmd", "robot_stand_pose_cmd",
                       "robot_control_group_cmd", "upper_body_fsm_state_change_cmd", "move_command_data"}


class _AlwaysMatched:
    @property
    def is_matched(self):
        return True

    @is_matched.setter
    def is_matched(self, v):
        self.real_matched = v       # 真实匹配状态仍然记录，便于调试


def _mark_optional(obj):
    obj.__class__ = type("Optional" + obj.__class__.__name__, (_AlwaysMatched, obj.__class__), {})
    obj.real_matched = False


_orig_sub = _ddsi.DDSInterface.create_subscriber
_orig_pub = _ddsi.DDSInterface.create_publisher


def _create_subscriber(self, topic_name, *a, **kw):
    sub = _orig_sub(self, topic_name, *a, **kw)
    if topic_name in OPTIONAL_TOPICS:
        _mark_optional(sub)
    return sub


def _create_publisher(self, pub_sub_type, topic_name, *a, **kw):
    pub = _orig_pub(self, pub_sub_type, topic_name, *a, **kw)
    if topic_name in OPTIONAL_PUB_TOPICS:
        _mark_optional(pub)
    return pub


_ddsi.DDSInterface.create_subscriber = _create_subscriber
_ddsi.DDSInterface.create_publisher = _create_publisher

import fourier_aurora_client.client as _cl  # noqa: E402
from fourier_aurora_client import AuroraClient  # noqa: E402,F401


class _PatientTime:
    """初始化时把 DDS 发现的等待从 5×0.5s 放宽到 5×2s。"""
    def __getattr__(self, k):
        return getattr(_time, k)

    @staticmethod
    def sleep(t):
        _time.sleep(t * 4 if t == 0.5 else t)


_orig_get_instance = AuroraClient.get_instance.__func__


def _get_instance(cls, *a, **kw):
    _cl.time = _PatientTime()
    try:
        inst = _orig_get_instance(cls, *a, **kw)
    finally:
        _cl.time = _time
    if inst is None:
        raise RuntimeError("AuroraClient 初始化失败：请确认 AuroraCore 已启动且 domain_id 一致")
    return inst


AuroraClient.get_instance = classmethod(_get_instance)


# ---------- v1.2.1 没有 policy_change_service（GR-1 只有一个行走策略），跳过 ----------
def _init_service_clients(self):
    self._service_clients_list = []


AuroraClient._init_service_clients = _init_service_clients


# ---------- v1.2.1 的 VelocityCmd 没有 duration 字段 ----------
def _set_velocity(self, vx, vy, yaw, duration=None):
    msg = _cl.AuroraCmd.VelocityCmd()
    msg.vx(float(vx)); msg.vy(float(vy)); msg.yaw(float(yaw))
    self._velocity_publisher.publish(msg)


AuroraClient.set_velocity = _set_velocity
