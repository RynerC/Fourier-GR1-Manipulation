"""
sim/gr1t2_manip_scene.py —— 给“快速操作训练环境”生成 GR1T2 + 桌子场景的 MJCF
==============================================================================
和 Aurora 仿真（aurora_bridge/sim_server.py）用同一个官方 GR1T2 模型（assets/gr1t2，取自 SDK 镜像
/usr/local/resources/GR1T2），但做了这些改动，让它能比实时快几十倍：

  * 骨盆 base_link 改成 mocap 物体（不再有浮动基 + 双腿平衡）：
    PdStand 下骨盆位姿几乎完全由站姿 (Δz, Δpitch, Δyaw) 决定（scripts/aurora_manip_sysid.py 实测：
    z = 0.836 + Δz，pitch = Δpitch，yaw = Δyaw，误差 <2 mm），所以每步直接按站姿摆骨盆。
  * 腿、头、左臂：位置伺服到固定姿态（只为画面好看），不参与碰撞。
  * 腰 + 右臂：位置伺服，增益取 Aurora pd_stand_task/config.yaml 的 kp/kd，
    再加部分重力补偿（gravcomp），让稳态下垂和 Aurora 实测一致（关节误差约 0.01 rad）。
  * 碰撞分组：bit0 桌子/托盘/地面，bit1 物体，bit2 机器人（手臂、躯干）；机器人自身不互撞。
  * 物体与右手末端之间预置 weld 约束（抓取辅助，和 sim_server 的 grasp 一样：没有手指）。
"""
import os
import re

from config import ROOT, SceneCfg

ASSET = os.path.join(ROOT, "assets", "gr1t2")
ROBOT_XML = os.path.join(ASSET, "mjcf", "gr1t2.xml")
OUT_XML = os.path.join(ASSET, "mjcf", "manip_scene.xml")

ARM = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch", "wrist_yaw", "wrist_roll", "wrist_pitch"]
RIGHT_ARM = [f"right_{j}_joint" for j in ARM]
LEFT_ARM = [f"left_{j}_joint" for j in ARM]
WAIST = ["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"]
HEAD = ["head_roll_joint", "head_pitch_joint", "head_yaw_joint"]
LEG = ["hip_roll", "hip_yaw", "hip_pitch", "knee_pitch", "ankle_pitch", "ankle_roll"]
LEGS = [f"{s}_{j}_joint" for s in ("left", "right") for j in LEG]
# PdStand 站立时腿的典型关节角（Aurora 仿真里读到的；只用于画面）
LEG_STAND = {"hip_pitch": -0.29, "knee_pitch": 0.58, "ankle_pitch": -0.29}

# Aurora pd_stand_task/config.yaml：kp / kd。腰 yaw 例外（100 → 500）：Aurora 全身控制器有力矩前馈，实测伸手时
# 腰 yaw 静差只有 0.002 rad；纯 PD kp=100 会被手臂拽偏 0.05 rad。
KP = {**dict(zip(WAIST, [500, 250, 200])), **dict(zip(HEAD, [100, 100, 100])),
      **dict(zip(RIGHT_ARM, [250, 150, 100, 100, 30, 30, 30])), **dict(zip(LEFT_ARM, [250, 150, 100, 100, 30, 30, 30]))}
KD = {**dict(zip(WAIST, [10, 10, 10])), **dict(zip(HEAD, [8, 8, 8])),
      **dict(zip(RIGHT_ARM, [10, 10, 0.5, 0.1, 0.5, 0.5, 0.5])), **dict(zip(LEFT_ARM, [10, 10, 0.5, 0.1, 0.5, 0.5, 0.5]))}
for j in LEGS:
    KP[j], KD[j] = 300.0, 10.0

ROBOT_COLLIDE = ("torso_link", "right_upper_arm_yaw_link", "right_hand_yaw_link", "head_yaw_link",
                 "left_upper_arm_yaw_link", "left_hand_yaw_link")


def build(path=OUT_XML, sc: SceneCfg = None, gravcomp=0.8):
    sc = sc or SceneCfg()
    x = open(ROBOT_XML, encoding="utf-8").read()
    x = re.sub(r"<keyframe>.*?</keyframe>", "", x, flags=re.S)
    x = x.replace('<include file="gr1t2_actuator_sensor.xml"/>', "")
    x = re.sub(r'<joint name="base_free_joint"[^>]*/>', "", x)
    x, n = re.subn(r'<body name="base_link"([^>]*)>', r'<body name="base_link"\1 mocap="true">', x, count=1)
    assert n == 1
    x = re.sub(r"<option[^>]*>", '<option timestep="0.005" integrator="implicitfast" solver="Newton" '
               'cone="pyramidal" gravity="0 0 -9.81">', x, count=1)
    x = x.replace('<flag contact="enable" energy="enable"/>', "")
    x = x.replace("solref='0.001 2'", "solref='0.005 1'")
    x = x.replace('file="../meshes/', 'file="../meshes/')
    # 碰撞：只保留躯干、手臂、头的碰撞体（contype 4，只和桌子/物体碰）；腿和骨盆不碰撞
    def fix_geom(m):
        g = m.group(0)
        name = re.search(r'name="([^"]+)\.STL"', g).group(1)
        if name in ROBOT_COLLIDE:
            return g.replace("class='collision'", "class='collision' contype=\"4\" conaffinity=\"3\"")
        return g.replace("class='collision'", "class='collision' contype=\"0\" conaffinity=\"0\"")
    x = re.sub(r"<geom name=\"[^\"]+\.STL\"[^>]*class='collision'[^>]*/>", fix_geom, x)
    # 重力补偿：上身（腰以上）所有物体
    upper = ["waist_yaw_link", "waist_pitch_link", "waist_roll_link", "torso_link", "head_roll_link", "head_pitch_link",
             "head_yaw_link"] + [f"{s}_{l}" for s in ("left", "right") for l in (
                 "upper_arm_pitch_link", "upper_arm_roll_link", "upper_arm_yaw_link", "lower_arm_pitch_link",
                 "hand_yaw_link", "hand_roll_link", "hand_pitch_link")]
    for b in upper:
        x = re.sub(rf'(<body name="{b}"[^>]*?)(\s*/?>)', rf'\1 gravcomp="{gravcomp}"\2', x, count=1)

    joints = WAIST + HEAD + RIGHT_ARM + LEFT_ARM + LEGS
    acts = "\n".join(
        f'    <general name="{j}" joint="{j}" gaintype="fixed" biastype="affine" gainprm="{KP[j]}" '
        f'biasprm="0 -{KP[j]} -{KD[j]}"/>' for j in joints)
    cx, cy = sc.table_pos; hx, hy = sc.table_half; h, t = sc.table_height, sc.table_thick
    th = sc.tray_half
    objs, welds = [], []
    for i, (name, (shape, size, rgba)) in enumerate(sc.objects.items()):
        sz = " ".join(f"{v:.3f}" for v in size)
        z = h + (size[2] if shape == "box" else size[1]) + 0.002
        objs.append(f'<body name="{name}" pos="{cx - 0.3 + 0.3 * i:.3f} {cy:.3f} {z:.3f}"><freejoint name="{name}_free"/>'
                    f'<geom name="{name}_geom" type="{shape}" size="{sz}" rgba="{" ".join(map(str, rgba))}" mass="0.15" '
                    f'friction="1.2 0.02 0.001" contype="2" conaffinity="7" condim="4"/></body>')
        welds.append(f'<weld name="grasp_{name}" body1="right_end_effector_link" body2="{name}" active="false" '
                     f'solref="0.01 1"/>')
    scene = f"""
  <actuator>
{acts}
  </actuator>
  <equality>
    {"".join(welds)}
  </equality>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
             markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 1" type="plane" material="groundplane" contype="1" conaffinity="6"/>
    <body name="table" pos="{cx} {cy} 0">
      <geom name="table_top" type="box" size="{hx} {hy} {t / 2}" pos="0 0 {h - t / 2}" rgba="0.65 0.5 0.35 1"
            contype="1" conaffinity="6"/>
    </body>
    <body name="tray" mocap="true" pos="{cx + 0.45} {cy + 0.1} {h + th[2]}">
      <geom name="tray_geom" type="box" size="{th[0]} {th[1]} {th[2]}" rgba="{" ".join(map(str, sc.tray_rgba))}"
            contype="1" conaffinity="6"/>
    </body>
    <body name="goal_marker" mocap="true" pos="0 0 -1">
      <geom type="sphere" size="0.015" rgba="1 1 0 0.5" contype="0" conaffinity="0"/>
    </body>
    {"".join(objs)}
  </worldbody>
</mujoco>"""
    x = x[:x.rindex("</mujoco>")] + scene
    with open(path, "w", encoding="utf-8") as f:
        f.write(x)
    return path


if __name__ == "__main__":
    import mujoco
    p = build()
    m = mujoco.MjModel.from_xml_path(p)
    print(p, "nq", m.nq, "nu", m.nu, "nbody", m.nbody)
