"""
aurora_bridge/sim_server.py —— 无界面的 Aurora 仿真端：MuJoCo 物理 <-> LCM <-> AuroraCore
=========================================================================================
官方 SDK 的 sim/mujoco/viewer.py 需要 GLFW 窗口；这个文件做同样的事但不需要显示器，并加了：
  * 虚拟吊绳（elastic band）：开机时吊住骨盆；'lower' 像龙门吊一样慢慢放到脚完全承重再摘掉
  * 'reset[:x,y,yaw]'：把机器人放回吊绳上的初始姿态（可指定出生点）
  * 'payload:<kg>'：在右手末端加竖直向下的力（模拟拿着物体）
  * 'place:<物体名>:x,y[,z[,yaw]]' / 'place:tray:x,y'：摆放物体 / 托盘（操作评估用）
  * 'grasp:<物体名>' / 'drop'：抓取辅助（模型没有手指：物体刚性跟随右手末端，并把重量加在手上）
  * LCM 'sim_objects'（10 Hz, JSON）：物体与托盘的真值位置，演示脚本拿它当“感知结果"
  * --scene tabletop：在 GR1T2 模型里加入项目的桌子 / 托盘 / 物体（尺寸来自 config.py）
  * --truth-log：记录真值（位姿、速度、躯干俯仰、右手位置、质心…）供分析
  * --qpos-log：按 30 Hz 记录完整 qpos（+ 'label:<文字>' 发来的阶段标签），
    用 scripts/render_aurora_video.py 离线渲染成视频（无界面服务器上也能出视频）
控制通道：LCM 'sim_ctrl'（见 _on_ctrl）。必须在 Aurora 镜像里运行（mujoco 2.3.6 + lcm）。

PD 规则与官方一致：ctrl = kp*(q_cmd - q) + kd*(qd_cmd - qd) + tau_ff。
"""
import argparse, os, sys, threading, time
import numpy as np, mujoco, lcm
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'lcm_msgs'))
import simulator_pub_msg, simulator_sub_msg  # noqa: E402

DEFAULT_XML = '/usr/local/resources/GR1T2/mjcf/main.xml'


def build_tabletop_xml(main_xml=DEFAULT_XML):
    """在 main.xml 旁边生成 scene_tabletop.xml（必须同目录：网格路径相对顶层文件解析）。"""
    sys.path.insert(0, os.path.dirname(HERE))
    from config import SceneCfg
    sc = SceneCfg()
    cx, cy = sc.table_pos; hx, hy = sc.table_half; h, t = sc.table_height, sc.table_thick
    legs = "".join(
        f'<geom type="box" size="0.025 0.025 {(h - t) / 2:.3f}" pos="{sx * (hx - 0.05):.3f} {sy * (hy - 0.05):.3f} {(h - t) / 2:.3f}" rgba="0.45 0.3 0.2 1" conaffinity="3"/>'
        for sx in (-1, 1) for sy in (-1, 1))
    objs = []
    for i, (name, (shape, size, rgba)) in enumerate(sc.objects.items()):
        sz = " ".join(f"{v:.3f}" for v in size)
        z = h + (size[2] if shape == "box" else size[1]) + 0.002
        objs.append(f'<body name="{name}" pos="{cx - 0.3 + 0.3 * i:.3f} {cy:.3f} {z:.3f}"><freejoint/>'
                    f'<geom type="{shape}" size="{sz}" rgba="{" ".join(map(str, rgba))}" mass="0.15" friction="1.2 0.02 0.001" conaffinity="3"/></body>')
    th = sc.tray_half
    d = os.path.dirname(main_xml)
    # 原模型里有一个 39 维的 keyframe，加了自由物体后维度不匹配 → 复制一份去掉 keyframe 的机器人文件
    robot = open(os.path.join(d, 'gr1t2.xml')).read()
    i, j = robot.find('<keyframe>'), robot.find('</keyframe>')
    if i >= 0:
        robot = robot[:i] + robot[j + len('</keyframe>'):]
    with open(os.path.join(d, 'gr1t2_nokey.xml'), 'w') as f:
        f.write(robot)
    main = open(main_xml).read().replace('<include file="gr1t2.xml"/>', '<include file="gr1t2_nokey.xml"/>')
    xml = main.replace('</mujoco>', f"""
  <worldbody>
    <body name="table" pos="{cx} {cy} 0">
      <geom name="table_top" type="box" size="{hx} {hy} {t / 2}" pos="0 0 {h - t / 2}" rgba="0.65 0.5 0.35 1" conaffinity="3"/>
      {legs}
    </body>
    <body name="tray" pos="{cx + 0.45} {cy + 0.1} {h + th[2]}">
      <geom type="box" size="{th[0]} {th[1]} {th[2]}" rgba="{" ".join(map(str, sc.tray_rgba))}" conaffinity="3"/>
    </body>
    {"".join(objs)}
  </worldbody>
</mujoco>
""")
    out = os.path.join(os.path.dirname(main_xml), 'scene_tabletop.xml')
    with open(out, 'w') as f:
        f.write(xml)
    return out

class AuroraSim:
    def __init__(self, xml, realtime=1.0, log=None, spawn=(0.0, 0.0, 0.0)):
        self.m = mujoco.MjModel.from_xml_path(xml)
        self.d = mujoco.MjData(self.m)
        self.spawn = tuple(spawn)
        self._place(self.spawn)
        mujoco.mj_forward(self.m, self.d)
        self.lock = threading.Lock()
        self.lc = lcm.LCM()
        self.cmd = None
        self.realtime = realtime
        self.truth_log = log is not None
        self.qpos_log, self._ql_last, self.label = None, -1.0, ''
        if log:
            self._tl = open(log, 'w', buffering=1)
            self._tl_last = 0.0
        m = self.m
        self.nu = m.nu
        sid = lambda n: m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, n)]
        self.s_acc = sid('baselink-baseAcc')
        self.lc.subscribe('msg_controller_to_simulator', self._on_cmd)
        self.lc.subscribe('sim_ctrl', self._on_ctrl)
        self.band_xy = np.array(self.d.qpos[0:2])
        self.running = True
        self.step_hooks = []     # callables(m, d) executed every physics step (inside lock)
        # 抓取辅助：GR1T2 仿真模型没有手指，抓住 = 物体刚性跟随右手末端（并在末端加物体重量）
        self.held = None
        self._nocollide = {}
        sys.path.insert(0, os.path.dirname(HERE))
        from config import SceneCfg
        self._sc = SceneCfg()
        self._half = {n: (s[1][2] if s[0] == 'box' else s[1][1]) for n, s in self._sc.objects.items()}
        self.obj_names = [m.body(i).name for i in range(m.nbody)
                          if m.body_jntnum[i] == 1 and m.jnt_type[m.body_jntadr[i]] == mujoco.mjtJoint.mjJNT_FREE
                          and m.body(i).name != 'base_link']
        self._obj_t = 0.0
        # 虚拟吊绳（elastic band）：仿真刚启动时机器人还没进入站立控制，没有它会直接摔倒。
        # 吊住骨盆，等 PdStand 接管后再松开（release_band()）。
        self.band = True
        self.band_gain = 1.0
        self.base_bid = m.body('base_link').id
        self.band_z = None
        self.ree_bid = m.body('right_end_effector_link').id
        self.torso_bid = m.body('torso_link').id
        self.lf_bid = m.body('left_foot_roll_link').id
        self.rf_bid = m.body('right_foot_roll_link').id
        threading.Thread(target=self._lcm_loop, daemon=True).start()

    def _on_cmd(self, ch, data):
        self.cmd = simulator_sub_msg.simulator_sub_msg.decode(data)

    def _on_ctrl(self, ch, data):
        """简单的仿真控制通道（LCM 'sim_ctrl'）：b'release' 松开吊绳，b'band' 重新吊住。"""
        cmd = data.decode()
        if cmd == 'release':
            threading.Thread(target=self.release_band, daemon=True).start()
        elif cmd.startswith('reset'):
            # 'reset[:x,y,yaw[:keep]]'：机器人回到出生点重新吊起来；带 ':keep' 时桌上物体保持原样（连续任务用）
            parts = cmd.split(':')
            sp = tuple(float(v) for v in parts[1].split(',')) if len(parts) > 1 and parts[1] else self.spawn
            keep = len(parts) > 2 and parts[2] == 'keep'
            with self.lock:
                t_keep = self.d.time            # 保持仿真时钟连续（实时步进与日志都依赖它）
                saved = {}
                if keep:
                    for n in self.obj_names:
                        j = self.m.body_jntadr[self.m.body(n).id]
                        qa = self.m.jnt_qposadr[j]
                        saved[n] = self.d.qpos[qa:qa + 7].copy()
                mujoco.mj_resetData(self.m, self.d)
                for n, q in saved.items():
                    j = self.m.body_jntadr[self.m.body(n).id]
                    qa = self.m.jnt_qposadr[j]
                    self.d.qpos[qa:qa + 7] = q
                self.d.time = t_keep
                self.held = None
                for name in list(self._nocollide):
                    for g, ct, ca in self._nocollide.pop(name):
                        self.m.geom_contype[g] = ct
                        self.m.geom_conaffinity[g] = ca
                self._place(sp)
                mujoco.mj_forward(self.m, self.d)
                self.band_z = None
                self.band_xy = np.array(self.d.qpos[0:2])
                self.band = True
        elif cmd == 'lower':
            threading.Thread(target=self.lower_band, daemon=True).start()
        elif cmd.startswith('fade'):
            # 'fade:<seconds>'：吊绳刚度在给定时间内线性减到 0（机器人已在地面上主动平衡时用）
            T = float(cmd.split(':')[1]) if ':' in cmd else 2.0
            threading.Thread(target=self.fade_band, args=(T,), daemon=True).start()
        elif cmd.startswith('payload'):
            # 'payload:<kg>'：在右手末端施加竖直向下 m·g（模拟手里拿着物体）
            kg = float(cmd.split(':')[1])
            with self.lock:
                self.d.xfrc_applied[self.ree_bid, :] = 0
                self.d.xfrc_applied[self.ree_bid, 2] = -9.81 * kg
        elif cmd.startswith('place:'):
            # 'place:<物体>:x,y[,z[,yaw]]'：把物体瞬移到桌面上某点（z 省略 = 放在桌面上）；'place:tray:x,y' 移动托盘
            _, name, vals = cmd.split(':')
            v = [float(t) for t in vals.split(',')]
            with self.lock:
                self.place(name, v)
        elif cmd.startswith('grasp:'):
            self.grasp(cmd.split(':')[1])
        elif cmd.startswith('label:'):  # 视频里的阶段字幕
            self.label = cmd[6:]
        elif cmd == 'drop':          # 松手（'release' 是松吊绳）
            self.drop()
        elif cmd == 'band':
            with self.lock:
                self.band_z = self.d.xpos[self.base_bid][2] + 0.02
                self.band_xy = np.array(self.d.xpos[self.base_bid][0:2])
                self.band = True

    def _place(self, spawn):
        """把机器人的浮动基放到 (x, y, yaw)，高度保持模型默认（吊绳会接住）。"""
        x, y, yaw = spawn
        self.d.qpos[0] += x; self.d.qpos[1] += y
        self.d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

    def place(self, name, v):
        m, d = self.m, self.d
        if name == 'tray':
            b = m.body('tray').id
            m.body_pos[b][:2] = v[:2]
            mujoco.mj_forward(m, d)
            if self.qpos_log:
                # 托盘是固定物体，位置不在 qpos 里：单独记一行，离线渲染视频时按时间把托盘挪过去
                p = m.body_pos[b]
                self.qpos_log.write(f"#tray\t{d.time:.3f}\t{p[0]:.4f},{p[1]:.4f},{p[2]:.4f}\n")
            return
        b = m.body(name).id
        j = m.body_jntadr[b]
        qa, va = m.jnt_qposadr[j], m.jnt_dofadr[j]
        g = m.body_geomadr[b]
        half = m.geom_size[g][2] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else m.geom_size[g][1]
        z = v[2] if len(v) > 2 else 0.80 + half + 0.002
        yaw = v[3] if len(v) > 3 else 0.0
        d.qpos[qa:qa + 3] = [v[0], v[1], z]
        d.qpos[qa + 3:qa + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.qvel[va:va + 6] = 0
        mujoco.mj_forward(m, d)

    def grasp(self, name, max_dist=0.10):
        """若物体离右手末端 < max_dist，则“吸住”：记录相对位姿，之后每步把物体放到手上。"""
        with self.lock:
            if name not in self.obj_names:
                print(f'[sim] grasp: 没有物体 {name}', flush=True); return
            b = self.m.body(name).id; e = self.ree_bid
            d = np.linalg.norm(self.d.xpos[b] - self.d.xpos[e])
            if d > max_dist:
                print(f'[sim] grasp {name} 失败：离手 {d:.3f} m', flush=True); return
            Re = self.d.xmat[e].reshape(3, 3)
            self.held = (name, Re.T @ (self.d.xpos[b] - self.d.xpos[e]), Re.T @ self.d.xmat[b].reshape(3, 3),
                         self.m.body_subtreemass[b])
            self.d.xfrc_applied[e, :] = 0
            self.d.xfrc_applied[e, 2] = -9.81 * self.held[3]     # 物体重量压在手上
            # 抓着的时候关掉物体和机器人的碰撞：物体每步被“摆”到手上，再和手/前臂碰撞会产生巨大接触力，
            # 实测会让仿真数值发散（MuJoCo 报 QACC NaN 并自动重置）
            self._nocollide[name] = [(g, int(self.m.geom_contype[g]), int(self.m.geom_conaffinity[g]))
                                     for g in range(self.m.body_geomadr[b], self.m.body_geomadr[b] + self.m.body_geomnum[b])]
            # 碰撞位：机器人 contype/conaffinity = 1/1；桌子、托盘、其他物体的 conaffinity 含 bit 1（=3）。
            # 抓着的物体设成 2/2：不和机器人碰（它每步被“摆”到手上，再和手臂碰会数值发散），
            # 但仍然和桌子/托盘/其他物体碰 —— 否则它会穿进托盘上已经放好的物体，松手后再恢复碰撞时
            # 两个物体是重叠的，会被巨大的接触力弹飞（实测：放第二、三个物体时前一个物体凭空“消失”）
            for g, _, _ in self._nocollide[name]:
                self.m.geom_contype[g] = 2
                self.m.geom_conaffinity[g] = 2
            print(f'[sim] grasped {name} (d={d:.3f})', flush=True)

    def drop(self):
        with self.lock:
            if self.held:
                print(f'[sim] dropped {self.held[0]}', flush=True)
                # 松手：仍然只和桌子/托盘/地面/其他物体碰，离手够远后再恢复和机器人的碰撞
                for g, _, _ in self._nocollide.get(self.held[0], []):
                    self.m.geom_contype[g] = 2
                    self.m.geom_conaffinity[g] = 2
            self.held = None
            self.d.xfrc_applied[self.ree_bid, :] = 0

    def _support_z(self, xy):
        sc = self._sc
        tx, ty = sc.table_pos; hx, hy = sc.table_half
        z = 0.0
        if abs(xy[0] - tx) <= hx and abs(xy[1] - ty) <= hy:
            z = sc.table_height
            try:
                tp = self.d.xpos[self.m.body('tray').id]; th = sc.tray_half
                if abs(xy[0] - tp[0]) <= th[0] and abs(xy[1] - tp[1]) <= th[1]:
                    z = sc.table_height + 2 * th[2]
            except KeyError:
                pass
        return z

    def _restore_collisions(self):
        # 松手后，等物体和机器人**所有**碰撞体（手、前臂、上臂、躯干……）都分开了再恢复和机器人的碰撞。
        # 旧版只看“离手腕 >13 cm”，但物体可能还贴着前臂 —— 一恢复碰撞两者是重叠的，物体被弹飞几米，
        # 还会撞倒托盘上别的物体（实测）。这里用包围球保守判断。
        m, d = self.m, self.d
        if not hasattr(self, "_robot_gids"):
            root = m.body('base_link').id
            def under(b):
                while b > 0:
                    if b == root:
                        return True
                    b = m.body_parentid[b]
                return False
            self._robot_gids = np.array([g for g in range(m.ngeom)
                                         if under(m.geom_bodyid[g]) and (m.geom_contype[g] or m.geom_conaffinity[g])])
        for name in list(self._nocollide):
            if self.held and self.held[0] == name:
                continue
            gids = [g for g, _, _ in self._nocollide[name]]
            clear = True
            for g in gids:
                dist = np.linalg.norm(d.geom_xpos[self._robot_gids] - d.geom_xpos[g], axis=1)
                if np.any(dist < m.geom_rbound[self._robot_gids] + m.geom_rbound[g] + 0.01):
                    clear = False
                    break
            if clear:
                for g, ct, ca in self._nocollide.pop(name):
                    m.geom_contype[g] = ct
                    m.geom_conaffinity[g] = ca

    def _hold_hook(self):
        if self._nocollide:
            self._restore_collisions()
        if not self.held:
            return
        name, p_rel, R_rel, _ = self.held
        e = self.ree_bid; j = self.m.body(name).jntadr[0]
        Re = self.d.xmat[e].reshape(3, 3)
        qa = self.m.jnt_qposadr[j]; va = self.m.jnt_dofadr[j]
        self.d.qpos[qa:qa + 3] = self.d.xpos[e] + Re @ p_rel
        q = np.zeros(4); mujoco.mju_mat2Quat(q, (Re @ R_rel).flatten())
        self.d.qpos[qa + 3:qa + 7] = q
        # 抓着的物体是“摆”上去的（无限刚），不能让它穿进桌面/托盘：夹住高度（物体半高按竖直放近似）
        p = self.d.qpos[qa:qa + 3]
        self.d.qpos[qa + 2] = max(p[2], self._support_z(p[:2]) + self._half[name])
        self.d.qvel[va:va + 6] = 0

    def publish_objects(self):
        """LCM 'sim_objects'：桌上物体/托盘的真值位置（JSON），给演示脚本当“感知结果”用。"""
        import json
        out = {n: self.d.xpos[self.m.body(n).id].round(4).tolist() for n in self.obj_names}
        try:
            out['tray'] = self.d.xpos[self.m.body('tray').id].round(4).tolist()
        except KeyError:
            pass
        out['held'] = self.held[0] if self.held else None
        out['quat'] = {n: self.d.xquat[self.m.body(n).id].round(4).tolist() for n in self.obj_names}
        self.lc.publish('sim_objects', json.dumps(out).encode())

    def _lcm_loop(self):
        while self.running:
            self.lc.handle_timeout(50)

    def publish_state(self):
        m, d = self.m, self.d
        msg = simulator_pub_msg.simulator_pub_msg()
        msg.timestamp = int(time.time() * 1000.0)
        msg.motorNum = m.nu
        sd = d.sensordata
        msg.q = sd[0:m.nu]; msg.qDot = sd[m.nu:2 * m.nu]
        i = self.s_acc
        msg.acc = sd[i:i + 3]; i += 3
        msg.omega = sd[i:i + 3]; i += 3
        msg.quat = sd[i:i + 4]; i += 4
        msg.lf_touch = sd[i]; i += 1
        msg.rf_touch = sd[i]; i += 1
        msg.pos = sd[i:i + 3]; i += 3
        msg.vel = sd[i:i + 3]
        self.lc.publish('msg_simulator_to_controller', msg.encode())

    def apply_control(self):
        m, d, c = self.m, self.d, self.cmd
        if c is None or c.started == 0:
            d.ctrl[:self.nu] = 0
        else:
            q = d.sensordata[0:m.nu]; qd = d.sensordata[m.nu:2 * m.nu]
            d.ctrl[:self.nu] = (np.asarray(c.q_kp) * (np.asarray(c.qCmd) - q)
                                + np.asarray(c.q_kd) * (np.asarray(c.qDotCmd) - qd) + np.asarray(c.torqCmd))

    def step(self):
        with self.lock:
            self.publish_state()
            self.apply_control()
            if self.band:
                self._apply_band()
            self._hold_hook()
            for h in self.step_hooks:
                h(self.m, self.d)
            mujoco.mj_step(self.m, self.d)

    def _apply_band(self):
        d, b = self.d, self.base_bid
        p = d.xpos[b]; v = d.cvel[b]   # cvel = [ang(3), lin(3)] in world frame at com
        if self.band_z is None:
            self.band_z = p[2]
        f = np.zeros(6)
        mg = self.m.body_subtreemass[b] * 9.81
        f[0:2] = -1500 * (p[0:2] - self.band_xy) - 300 * v[3:5]
        f[2] = mg + 6000 * (self.band_z - p[2]) - 600 * v[5]
        f[3:6] = -150 * v[0:3]
        R = d.xmat[b].reshape(3, 3)
        f[3:6] += 2000 * np.cross(R[:, 2], [0, 0, 1])     # keep upright
        f *= self.band_gain                                # fade_band() 逐渐减小
        d.xfrc_applied[b, 0:3] = f[0:3]
        d.xfrc_applied[b, 3:6] = f[3:6]

    def lower_band(self, speed=0.02, timeout=8.0):
        """像龙门吊一样慢慢放下：吊点以 speed m/s 下降，直到吊绳几乎不受力（< 5% 体重）
        并保持 0.3 s，再摘掉吊绳。避免“还吊着一半体重时突然松手”造成的冲击。"""
        if not self.band:
            return
        mg = self.m.body_subtreemass[self.base_bid] * 9.81
        t0 = time.time(); slack_since = None
        while time.time() - t0 < timeout:
            self.band_z -= speed * 0.01
            fz = self.d.xfrc_applied[self.base_bid, 2]
            if fz < 0.05 * mg:
                slack_since = slack_since or time.time()
                if time.time() - slack_since > 0.3:
                    break
            else:
                slack_since = None
            time.sleep(0.01)
        self.band = False
        with self.lock:
            self.d.xfrc_applied[self.base_bid] = 0
        print(f'[sim] band removed after {time.time() - t0:.1f}s', flush=True)

    def fade_band(self, T):
        n = max(1, int(T / 0.01))
        for i in range(n):
            self.band_gain = 1.0 - (i + 1) / n
            time.sleep(0.01)
        self.band = False
        self.band_gain = 1.0
        with self.lock:
            self.d.xfrc_applied[self.base_bid] = 0

    def release_band(self, lower_first=True):
        """先把吊点缓慢降到脚着地，再松开。"""
        if not self.band:
            return
        if lower_first:
            for _ in range(150):
                self.band_z -= 0.0004
                time.sleep(0.01)
        self.band = False
        with self.lock:
            self.d.xfrc_applied[self.base_bid] = 0

    def _truth_log(self):
        """CSV 列：wall_t, sim_t, x, y, z, yaw, vW(3), vB(3), roll, pitch, torso_pitch,
        右手末端 xyz(3), 质心 xy(2), 双脚中点 xy(2)"""
        d, b = self.d, self.base_bid
        R = d.xmat[b].reshape(3, 3)
        vw = d.sensordata[self.s_acc + 15:self.s_acc + 18]
        yaw = np.arctan2(R[1, 0], R[0, 0])
        pitch = np.arcsin(-np.clip(R[2, 0], -1, 1)); roll = np.arctan2(R[2, 1], R[2, 2])
        Rt = d.xmat[self.torso_bid].reshape(3, 3)
        tpitch = np.arcsin(-np.clip(Rt[2, 0], -1, 1))
        feet = 0.5 * (d.xpos[self.lf_bid][:2] + d.xpos[self.rf_bid][:2])
        self._tl.write(f"{time.time():.3f},{d.time:.3f}," + ",".join(f"{x:.4f}" for x in
                       [*d.xpos[b], yaw, *vw, *(R.T @ vw), roll, pitch, tpitch,
                        *d.xpos[self.ree_bid], *d.subtree_com[b][:2], *feet,
                        *self._cmd_dbg(13)]) + "\n")

    def _cmd_dbg(self, j):
        c = self.cmd
        if c is None:
            return [0, 0, 0, 0, self.d.sensordata[j]]
        return [c.qCmd[j], c.q_kp[j], c.torqCmd[j], c.q_kd[j], self.d.sensordata[j]]

    def run(self, duration=None):
        t0 = time.perf_counter(); s0 = self.d.time
        lag_report = time.perf_counter()
        while self.running:
            target = (time.perf_counter() - t0) * self.realtime
            n = 0
            while self.d.time - s0 < target and n < 20:
                t_before = self.d.time
                self.step(); n += 1
                if self.d.time < t_before:
                    # MuJoCo 数值发散时会自动 mj_resetData（时间归零）：重新对齐时钟，否则仿真会“快进追赶”几分钟
                    print(f'[sim] 仿真数值发散被 MuJoCo 自动重置（t={t_before:.2f}s），时钟重新对齐', flush=True)
                    t0 = time.perf_counter(); s0 = self.d.time; target = 0.0
                    self.held = None; self.d.xfrc_applied[:] = 0
                    break
            if n == 0:
                time.sleep(0.0003)
            if duration and self.d.time - s0 > duration:
                break
            if self.obj_names and self.d.time - self._obj_t >= 0.05:
                self._obj_t = self.d.time
                with self.lock:
                    self.publish_objects()
            if self.qpos_log and self.d.time - self._ql_last >= 1 / 30:
                self._ql_last = self.d.time
                with self.lock:
                    self.qpos_log.write(f"{self.d.time:.3f}\t{self.label}\t"
                                        + ",".join(f"{v:.5f}" for v in self.d.qpos) + "\n")
            if self.truth_log and self.d.time - self._tl_last >= 0.02:
                self._tl_last = self.d.time
                with self.lock:
                    self._truth_log()
            if time.perf_counter() - lag_report > 5:
                lag_report = time.perf_counter()
                lag = target - (self.d.time - s0)
                if lag > 0.05:
                    print(f'[sim] lag {lag*1000:.0f} ms', flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--xml', default=DEFAULT_XML)
    ap.add_argument('--scene', default='plain', choices=['plain', 'tabletop'])
    ap.add_argument('--spawn', default='0,0,0', help='x,y,yaw 出生点')
    ap.add_argument('--realtime', type=float, default=1.0)
    ap.add_argument('--duration', type=float, default=None)
    ap.add_argument('--truth-log', default=None)
    ap.add_argument('--qpos-log', default=None, help='记录 qpos 轨迹（离线渲染视频用）')
    a = ap.parse_args()
    xml = build_tabletop_xml(a.xml) if a.scene == 'tabletop' else a.xml
    s = AuroraSim(xml, a.realtime, log=a.truth_log, spawn=[float(v) for v in a.spawn.split(',')])
    if a.qpos_log:
        s.qpos_log = open(a.qpos_log, 'w', buffering=1)
        s.qpos_log.write(f"# xml={xml}\n")
    print('sim nu', s.nu, 'timestep', s.m.opt.timestep, flush=True)
    s.run(a.duration)
