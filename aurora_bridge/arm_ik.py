"""
aurora_bridge/arm_ik.py —— GR1T2 右臂逆运动学（给 PdStand 下的“操作”用）
=========================================================================
Aurora 只接收关节指令（set_group_cmd），末端笛卡尔目标要自己算。做法：
  1. 在客户端进程里加载同一个 GR1T2 MJCF（只做运动学，不仿真）；
  2. 用 Aurora 发布的当前状态填 qpos：骨盆位姿（true_data 位置 + IMU 姿态）、腿/腰/右臂关节；
  3. 对右臂 7 个关节做阻尼最小二乘 (DLS) 迭代，让 right_end_effector_link 到达目标点。
腰、腿保持当前值（腰的角度由我们在 PdStand 里单独指定），所以解出来的是“在当前站姿下”的臂角。
因为手臂动起来以后 PdStand 会微调全身姿态，所以调用方应该“解 → 走一段 → 用新状态再解一次”。
"""
import numpy as np

import mujoco

LEG = ["hip_roll", "hip_yaw", "hip_pitch", "knee_pitch", "ankle_pitch", "ankle_roll"]
ARM = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch", "wrist_yaw", "wrist_roll",
       "wrist_pitch"]
GROUP_JOINTS = {
    "left_leg": [f"left_{j}_joint" for j in LEG],
    "right_leg": [f"right_{j}_joint" for j in LEG],
    "waist": ["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"],
    "left_manipulator": [f"left_{j}_joint" for j in ARM],
    "right_manipulator": [f"right_{j}_joint" for j in ARM],
}
IMU_OFFSET = np.array([0.0, 0.0, -0.0596])     # imu_link 相对 base_link（只有平移）


class RightArmIK:
    def __init__(self, xml="/usr/local/resources/GR1T2/mjcf/main.xml"):
        self.m = mujoco.MjModel.from_xml_path(xml)
        self.d = mujoco.MjData(self.m)
        self.ee = self.m.body("right_end_effector_link").id
        self.qadr, self.dadr = {}, {}
        for names in GROUP_JOINTS.values():
            for n in names:
                j = self.m.joint(n).id
                self.qadr[n], self.dadr[n] = self.m.jnt_qposadr[j], self.m.jnt_dofadr[j]
        self.arm = GROUP_JOINTS["right_manipulator"]
        self.waist = ["waist_yaw_joint", "waist_pitch_joint"]
        rng = lambda names: np.array([self.m.jnt_range[self.m.joint(n).id] for n in names])  # noqa: E731
        self.arm_rng = rng(self.arm)
        # 腰参与 IK 时收窄范围：前倾 ≤0.35 rad、转 ±0.4 rad（PdStand 实测 0.3 rad + 1 kg 负载稳定）
        self.waist_rng = np.array([[-0.4, 0.4], [0.0, 0.35]])

    def sync(self, client, pose_source="truth"):
        """用 Aurora 的当前状态更新运动学模型。"""
        d = self.d
        qw = np.array(client.get_base_data("quat_wxyz"), float)
        R = np.zeros(9); mujoco.mju_quat2Mat(R, qw); R = R.reshape(3, 3)
        p = np.array(client.get_true_data("pos_W") if pose_source == "truth" else client.get_base_data("pos_W"))
        d.qpos[0:3] = p - R @ IMU_OFFSET
        d.qpos[3:7] = qw
        for g, names in GROUP_JOINTS.items():
            try:
                q = client.get_group_state(g, "position")
            except KeyError:
                continue
            for n, v in zip(names, q):
                d.qpos[self.qadr[n]] = v
        mujoco.mj_kinematics(self.m, d)

    def ee_pos(self):
        return self.d.xpos[self.ee].copy()

    def solve(self, target, iters=100, damping=0.05, tol=0.005, use_waist=True, w_posture=0.02,
              quat_target=None, w_rot=0.15, up_axis=None, q_nominal=None):
        """返回 (右臂 7 关节角, 腰 [yaw, pitch, roll], 剩余位置误差 m)。
        use_waist：腰 yaw/pitch 也参与（范围见 waist_rng），能多够 10 cm 左右。
        quat_target：可选，末端姿态目标（wxyz，世界系）。端着物体时保持抓取那一刻的手部姿态，物体就不会歪
        （没有手指，抓取辅助是把物体刚性固定在手上）。w_rot：姿态误差的权重（1 rad ≈ w_rot 米）。
        up_axis：可选，末端坐标系里的一个方向，要求它保持竖直向上（只约束 2 个自由度，绕竖直轴转不管）。
        端着物体时用这个更合适：抓取那一刻取 up_axis = R_ee^T·[0,0,1]，之后物体就一直是正的。"""
        m, d = self.m, self.d
        names = self.arm + (self.waist if use_waist else [])
        idx_q = [self.qadr[n] for n in names]
        idx_v = [self.dadr[n] for n in names]
        lo = np.concatenate([self.arm_rng[:, 0]] + ([self.waist_rng[:, 0]] if use_waist else []))
        hi = np.concatenate([self.arm_rng[:, 1]] + ([self.waist_rng[:, 1]] if use_waist else []))
        q = np.clip(np.array(d.qpos[idx_q], float), lo, hi)
        # 零空间姿态偏好：默认“保持当前姿态”；给 q_nominal（右臂 7 + 腰 yaw/pitch）则往这个“好用的姿态”靠，
        # 避免一步步增量求解时漂进关节限位、卡在局部极小
        q_pref = q.copy() if q_nominal is None else np.clip(np.asarray(q_nominal, float)[:len(names)], lo, hi)
        n = len(names)
        W = np.ones(n)
        if use_waist:
            W[7:] = 0.3          # 腰的“代价”更高：能用手臂解决就少动腰
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        err = np.inf
        use_rot = quat_target is not None
        use_up = up_axis is not None and not use_rot
        k = 6 if use_rot else (5 if use_up else 3)
        zw = np.array([0.0, 0.0, 1.0])
        P = np.array([[1.0, 0, 0], [0, 1.0, 0]])       # 只取水平两轴的角速度（绕竖直轴自由）
        for _ in range(iters):
            d.qpos[idx_q] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            e = np.asarray(target) - d.xpos[self.ee]
            err = float(np.linalg.norm(e))
            if use_rot:
                er = np.zeros(3)
                mujoco.mju_subQuat(er, np.asarray(quat_target, float), d.xquat[self.ee])   # 末端坐标系下的旋转误差
                er = d.xmat[self.ee].reshape(3, 3) @ er                                     # → 世界系（与 jacr 一致）
                if err < tol and np.linalg.norm(er) < 0.05:
                    break
                e = np.concatenate([e, w_rot * er])
            elif use_up:
                a_w = d.xmat[self.ee].reshape(3, 3) @ np.asarray(up_axis, float)
                er = P @ np.cross(a_w, zw)                 # 让 a_w 转向竖直所需的角速度（水平分量）
                if err < tol and np.linalg.norm(er) < 0.03:
                    break
                e = np.concatenate([e, w_rot * er])
            elif err < tol:
                break
            mujoco.mj_jacBody(m, d, jacp, jacr, self.ee)
            if use_rot:
                Jf = np.vstack([jacp, w_rot * jacr])
            elif use_up:
                Jf = np.vstack([jacp, w_rot * (P @ jacr)])
            else:
                Jf = jacp
            J = Jf[:, idx_v] * W
            dq = W * (J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(k), e))
            N = np.eye(n) - np.linalg.pinv(J) @ J
            dq += N @ (w_posture * (q_pref - q))
            q = np.clip(q + np.clip(dq, -0.2, 0.2), lo, hi)
        waist = [q[7], q[8], 0.0] if use_waist else list(d.qpos[[self.qadr[w] for w in GROUP_JOINTS["waist"]]])
        return q[:7], waist, err
