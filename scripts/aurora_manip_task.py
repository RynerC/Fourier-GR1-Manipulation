"""
scripts/aurora_manip_task.py —— Aurora 全物理里的“连续任务”：一个场景、一次开机，依次把物体放到托盘上
=====================================================================================================
和 scripts/aurora_manip_eval.py 的区别：
  aurora_manip_eval.py 是子任务的“单元测试”—— 每回合重新开机、按快速环境重新摆场景；
  “放置”模式为了单独测放置，开局直接把物体放进手里（不是真的抓过来的）。
  本脚本是完整流程，状态一直连续：桌上物体只在开局摆一次，之后谁也不去动它；
  每个物体都是 “走到抓取站位 → PdStand 下用操作策略抓 → （托盘够不着就）端着走到放置站位 → 用操作策略放”，
  手里的物体就是刚才真正抓起来的那一个。下一个物体从上一个任务结束时的状态（机器人位置、站姿、
  桌上其他物体的位置、托盘上已有的物体）接着做。

在 Aurora 镜像里：
  bash aurora_bridge/start_stack.sh --scene tabletop --spawn=-1.6,0,0 --qpos-log /tmp/q.log
  python3 scripts/aurora_manip_task.py                                    # 默认：BC 操作策略 + PPO 导航
  python3 scripts/aurora_manip_task.py --manip scripted --layout default  # 脚本老师；物体保持场景文件里的初始位置
  python3 scripts/aurora_manip_task.py --layout keep --resume             # 接着上一次运行结束时的状态继续（物体、机器人都不动）
不加 --seed 时 seed = 1：每次 --layout random 都摆成同一个初始场景（会把上次的结果覆盖掉）；要接着做用 --layout keep。
  MUJOCO_GL=egl python3 scripts/render_aurora_video.py /tmp/q.log --speed 2 --out videos/manip_task.mp4
"""
import argparse
import json
import os
import sys
import time

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from aurora_bridge.arm_ik import RightArmIK  # noqa: E402
from aurora_bridge.aurora_compat import AuroraClient  # noqa: E402
from aurora_bridge.locomotion import ARM_TUCK, ARM_ZERO, AuroraLoco, AuroraNavRunner, ForwardFirstFollower  # noqa: E402
from aurora_bridge.manip_runtime import AuroraManipRuntime, SimObjects  # noqa: E402
from aurora_bridge.np_policy import NumpyPolicy  # noqa: E402
from config import RobotCfg, SceneCfg  # noqa: E402
from planning.geometry import TableGeometry  # noqa: E402
from sim.manip_aurora_env import TRAY_CORNER, AuroraManipSkill, AuroraManipWorld, ScriptedManip, carry_point_local  # noqa: E402

MODE_NAME = {0: "抓放", 1: "抓起端持", 2: "放置"}


class _EnvLike:
    def __init__(self, world, skill):
        self.world, self.skill = world, skill


def random_layout(geo, sc, rng, names):
    """开局摆一次：物体和托盘互相离开 ≥14 cm，且每个点都至少能从一条桌边够到。"""
    tx, ty = sc.table_pos
    hx, hy = sc.table_half
    pts = {}
    for n in names + ["tray"]:
        for _ in range(500):
            m = sc.tray_half[0] + 0.01 if n == "tray" else 0.07
            xy = np.array([tx + rng.uniform(-hx + m, hx - m), ty + rng.uniform(-hy + m, hy - m)])
            far = all(np.linalg.norm(xy - p) > (sc.tray_half[0] + 0.10 if "tray" in (n, k) else 0.14) for k, p in pts.items())
            reach = any(geo.stand_pose_for(xy, s)[1] for s, *_ in geo.sides())
            if far and reach:
                pts[n] = xy
                break
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manip", default=os.path.join(ROOT, "checkpoints", "manip_aurora_bc", "bc.npz"),
                    help="操作策略 .npz，或 scripted = 脚本老师")
    ap.add_argument("--nav", default=os.path.join(ROOT, "checkpoints", "nav_aurora", "best.npz"),
                    help="导航策略 .npz，或 follower = 几何跟踪器")
    ap.add_argument("--objects", default="red_block,green_can,blue_box", help="依次处理的物体")
    ap.add_argument("--layout", default="random", choices=["random", "default", "keep"],
                    help="random：开局按 --seed 随机摆一次（之后不再动）；default：场景文件里的初始位置；"
                         "keep：桌上物体保持上一次运行结束时的样子（接着上次做）")
    ap.add_argument("--resume", action="store_true",
                    help="机器人也不重新开机：如果它还站着（上一次运行结束时的位置），直接从那里接着做；"
                         "没站着才吊起来重新开机（物体照样保持）。一般和 --layout keep 一起用")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--spawn", default="-1.6,0.0,0.0")
    ap.add_argument("--max-steps", type=int, default=300, help="每个操作子任务最多步数（×0.05 s）")
    ap.add_argument("--retries", type=int, default=1, help="抓取 / 放置失败时，从当前状态重试几次")
    ap.add_argument("--out", default=None)
    ap.add_argument("--trace", action="store_true", help="操作时每 0.5 s 打印一次手/物体/目标")
    ap.add_argument("--domain-id", type=int, default=123)
    a = ap.parse_args()

    import lcm
    lc_sub, lc_pub = lcm.LCM(), lcm.LCM()
    sim_ctrl = lambda s: lc_pub.publish("sim_ctrl", s.encode())  # noqa: E731
    objs = SimObjects(lc_sub)
    client = AuroraClient.get_instance(domain_id=a.domain_id, robot_name="gr1t2")
    sc, geo = SceneCfg(), TableGeometry(SceneCfg(), RobotCfg())
    loco = AuroraLoco(client, sim_ctrl=sim_ctrl)
    nav = ForwardFirstFollower(geo) if a.nav == "follower" else NumpyPolicy(a.nav)
    runner = AuroraNavRunner(loco, geo, nav)
    ik = RightArmIK(os.environ.get("GR1T2_XML", "/usr/local/resources/GR1T2/mjcf/main.xml"))
    rt = AuroraManipRuntime(client, loco, ik, objs, sim_ctrl, sc)
    manip = None if a.manip == "scripted" else NumpyPolicy(a.manip)
    fw = AuroraManipWorld(seed=0)                    # 只用它的运动学算“端持点”（站位坐标系，与站位无关）
    fw.reset((0.0, 0.0, 0.0), randomize=False)
    carry_local = carry_point_local(fw)
    log = loco.log
    t_start = time.time()

    # ---------------- 开机（只开一次）+ 摆场景（只摆一次） ----------------
    objs.wait()
    standing = False
    if a.resume:
        time.sleep(0.5)
        z = float(client.get_true_data("pos_W")[2])
        standing = z > 0.7 and loco.stable_level() >= 15
    if standing:
        # 接着上次：机器人还站在上次结束的位置（PdStand）。重新接管：站姿 0、腰回正、手臂保持当前姿态
        log(f"[task] 接着上次运行：机器人站在 {np.round(loco.pose(), 2)}，不重新开机")
        client.set_velocity_source(0)
        client.set_fsm_state(AuroraLoco.FSM_PDSTAND)
        loco.mode = "stand"
        loco.stand_pose[:] = 0
        hold = {"waist": [0, 0, 0]}
        try:
            hold["right_manipulator"] = client.get_group_state("right_manipulator", "position")
        except KeyError:
            pass
        loco.ramp_upper(hold, T=0.5)
        if objs.data.get("held"):
            sim_ctrl("drop")
    else:
        keep = ":keep" if a.layout == "keep" else ""
        client.set_velocity_source(0); client.set_fsm_state(0); time.sleep(0.5)
        sim_ctrl("drop")
        sim_ctrl("reset:" + a.spawn + keep); time.sleep(1.0)
        sim_ctrl("label:开机：PdStand → 预热 → 慢放吊绳")
        loco.startup()
    names = [n for n in a.objects.split(",") if n]
    if a.layout == "random":
        lay = random_layout(geo, sc, np.random.default_rng(a.seed), list(sc.objects))
        for n, xy in lay.items():
            sim_ctrl(f"place:{n}:{xy[0]:.4f},{xy[1]:.4f}")
        loco.wait(0.5)
    rt.refresh()
    log("[task] 初始场景：" + "，".join(f"{n} {np.round(rt.obj_pos(n)[:2], 2)}" for n in sc.objects)
        + f"，托盘 {np.round(rt.tray_pos()[:2], 2)}")

    def tray_slot(target):
        """托盘上放哪儿：托盘 24×24 cm，3 个物体放在四个角（离中心各 6 cm，相邻两角相距 12 cm）。
        先排除离已有物体不到 8 cm 的角，再选离机器人最远的角：先放远处、后放近处，
        这样往远处放的时候前臂不会从已经放好的物体上方压过去（前臂离托盘只有几厘米，会把它碰倒）。"""
        rt.refresh()
        tp, th = rt.tray_pos(), np.array(sc.tray_half)
        on = [rt.obj_pos(n)[:2] for n in sc.objects if n != target
              and np.all(np.abs(rt.obj_pos(n)[:2] - tp[:2]) < th[:2] + 0.03) and rt.obj_pos(n)[2] > sc.table_height]
        corners = [tp[:2] + np.array([sx, sy]) * TRAY_CORNER for sx in (-1, 1) for sy in (-1, 1)]
        tx, ty = sc.table_pos
        hx, hy = sc.table_half
        me3 = loco.pose()
        from planning.geometry import StandPose
        here = StandPose(me3[0], me3[1], me3[2], "")
        on_table = lambda c: abs(c[0] - tx) < hx - 0.05 and abs(c[1] - ty) < hy - 0.05  # noqa: E731  别悬在桌沿外
        free = [c for c in corners + [tp[:2].copy()] if on_table(c) and all(np.linalg.norm(c - p) > 0.09 for p in on)]
        reach = [c for c in free if geo.reachable_from(here, c)]
        if reach:          # 空着、够得着：选离机器人最远的（先放远处、后放近处，前臂不会压过已放好的物体）
            best = max(reach, key=lambda c: np.linalg.norm(c - me3[:2]))
        elif free:         # 空着但不在“标准可达范围”里：选离机器人最近的那个（手臂一般还能够到）
            best = min(free, key=lambda c: np.linalg.norm(c - me3[:2]))
        else:              # 托盘上都占满了：选离已有物体最远的点
            best = max(corners + [tp[:2].copy()], key=lambda c: min(np.linalg.norm(c - p) for p in on))
            log(f"    [提示] 托盘上没有空位了，放在离已有物体最远处")
        g = AuroraManipSkill(rt).tray_goal(name=target)
        g[:2] = best
        log(f"    托盘上已有 {len(on)} 个物体，放到 {np.round(best - tp[:2], 3)}（相对托盘中心）")
        return g

    def run_manip(mode, target, tag):
        """在当前站位、当前状态下跑一个操作子任务。返回 (成功, info)。"""
        loco.stand_mode()
        loco.upper_hold.setdefault("waist", np.zeros(3))
        xy, yaw = rt.begin()
        sk = AuroraManipSkill(rt, obs_noise=False)
        if mode == 1:
            sk.set_task(target, goal_local=carry_local, release=False)
        else:
            sk.set_task(target, goal_world=tray_slot(target), release=True)
        sk.begin()
        teacher = ScriptedManip(_EnvLike(rt, sk)) if manip is None else None
        if teacher:
            teacher.reset()
        sim_ctrl(f"label:{tag}：{MODE_NAME[mode]} {target}（操作策略，PdStand）")
        info = {"success": False}
        for t in range(a.max_steps):
            o = sk.obs()
            act = teacher() if teacher else manip(o)
            sk.act(act)
            _, succ, dropped, info = sk.evaluate()
            if a.trace and t % 10 == 0:
                log(f"      t={t * 0.05:4.1f}s 手 {np.round(rt.to_local(rt.ee_pos()), 3)} 物体 {np.round(rt.to_local(rt.obj_pos(target)), 3)} "
                    f"目标 {np.round(rt.to_local(sk.goal_world), 3)} grip {rt.grip:.2f} held {rt.held} 动作 {np.round(act, 2)}")
            if succ or dropped:
                break
        log(f"    {MODE_NAME[mode]}：成功={info['success']} 用时 {(t + 1) * 0.05:.1f} s d_goal {info.get('d_goal', 0) * 100:.1f} cm"
            + ("  掉了!" if info.get("dropped") else ""))
        return bool(info["success"]), info

    def retract(hold_obj):
        """操作结束：站姿回 0；手里有物体就收成端持姿态，没有就放下手臂。"""
        arm = ARM_TUCK if hold_obj else ARM_ZERO
        loco.ramp_upper({"right_manipulator": arm, "waist": [0, 0, 0]}, T=1.5)
        loco.ramp_stand_pose((0, 0, 0))
        rt.stand = np.zeros(3)

    results = []
    for k, target in enumerate(names):
        rt.refresh()
        tag = f"{k + 1}/{len(names)}"
        obj, tray = rt.obj_pos(target), rt.tray_pos()
        rec = {"object": target, "picked": False, "placed": False}
        th_ = np.array(sc.tray_half)
        if np.all(np.abs(obj[:2] - tray[:2]) < th_[:2]) and abs(obj[2] - (tray[2] + th_[2] + rt.obj_half_h[target])) < 0.015:
            log(f"[task {tag}] {target} 已经在托盘上（上一次放的），跳过")
            rec.update(picked=True, placed=True, skipped="already_on_tray")
            results.append(rec)
            continue
        if obj[2] < sc.table_height - 0.05:
            log(f"[task {tag}] {target} 已经不在桌上（{np.round(obj, 2)}），跳过")
            results.append(rec)
            continue
        here = loco.pose()
        pick = geo.best_stand_pose(obj[:2], here[:2])
        if pick is None:
            log(f"[task {tag}] {target} 从哪条桌边都够不到，跳过")
            results.append(rec)
            continue
        direct = geo.reachable_from(pick, tray[:2])
        log(f"[task {tag}] {target} @ {np.round(obj[:2], 2)}：抓取站位 {pick.side} 侧 ({pick.x:+.2f},{pick.y:+.2f})，"
            f"托盘{'可以直接够到' if direct else '够不到 → 端着走过去'}")
        # 1) 走到抓取站位（手里没东西）
        sim_ctrl(f"label:{tag}：走到 {target} 的抓取站位（{pick.side} 侧）")
        loco.upper_hold.pop("right_manipulator", None)
        r = runner.go_to(pick.as_array(), timeout=90)
        log(f"    导航：到达={r['reached']} 误差 {r['pos_err'] * 100:.1f} cm，倒退 {r['back_dist']:.2f} m")
        # 2) 抓（托盘够得着就直接抓放，否则抓起端持）
        mode = 0 if direct else 1
        ok = False
        for attempt in range(1 + a.retries):
            ok, info = run_manip(mode, target, tag)
            rt.refresh()
            if ok or info.get("dropped"):
                break
            if mode == 0 and rt.held == target:      # 抓住了但没放好：接着在原地放（不重新抓）
                ok, info = run_manip(2, target, tag)
                break
        rec["picked"] = bool(ok or rt.held == target)
        if mode == 0:
            rec["placed"] = ok
            retract(False)
            results.append(rec)
            continue
        if rt.held != target:
            log(f"    没抓起来，放弃 {target}")
            retract(False)
            results.append(rec)
            continue
        # 3) 端着物体走到放置站位（手里就是刚抓起来的那个物体）
        retract(True)
        tray = rt.tray_pos()
        place = geo.best_stand_pose(tray[:2], loco.pose()[:2])
        sim_ctrl(f"label:{tag}：端着 {target} 走到托盘旁（{place.side} 侧）")
        loco.upper_hold["right_manipulator"] = np.array(ARM_TUCK, float)
        r = runner.go_to(place.as_array(), timeout=120)
        rt.refresh()
        log(f"    导航（端着）：到达={r['reached']} 误差 {r['pos_err'] * 100:.1f} cm，物体{'还在手上' if rt.held == target else '掉了!'}")
        if rt.held != target:
            results.append(rec)
            continue
        # 4) 放
        for attempt in range(1 + a.retries):
            ok, info = run_manip(2, target, tag)
            rt.refresh()
            if ok or rt.held != target:
                break
        rec["placed"] = ok
        retract(False)
        results.append(rec)

    # ---------------- 最终状态 ----------------
    loco.wait(1.0)
    rt.refresh()
    tp, th = rt.tray_pos(), np.array(sc.tray_half)
    final = {}
    for n in sc.objects:
        p = rt.obj_pos(n)
        # 在托盘上 = 中心在托盘面内（留 1 cm 余量）、底面正好压在托盘上（高度误差 <1.5 cm）、没倒
        z_ok = abs(p[2] - (tp[2] + th[2] + rt.obj_half_h[n])) < 0.015
        final[n] = {"pos": p.round(3).tolist(), "offset_cm": ((p[:2] - tp[:2]) * 100).round(1).tolist(),
                    "on_tray": bool(np.all(np.abs(p[:2] - tp[:2]) < th[:2] - 0.01) and z_ok and rt.obj_tilt(n) < 10),
                    "tilt": rt.obj_tilt(n)}
    z = float(client.get_true_data("pos_W")[2])
    n_on = sum(final[n]["on_tray"] for n in names)
    sim_ctrl(f"label:完成：{n_on}/{len(names)} 个物体在托盘上")
    loco.wait(2.0)
    summ = {"manip": a.manip, "nav": a.nav, "on_tray": n_on, "n": len(names), "fell": z < 0.6,
            "time_s": time.time() - t_start, "tasks": results, "final": final}
    log("[task] 结果：" + "，".join(f"{n} {'在托盘上' if final[n]['on_tray'] else '不在托盘上'}" for n in names)
        + f"；机器人{'摔倒了' if z < 0.6 else '站着'}，总用时 {summ['time_s']:.0f} s")
    print("SUMMARY", json.dumps({k: v for k, v in summ.items() if k != "final"}, ensure_ascii=False), flush=True)
    if a.out:
        json.dump(summ, open(a.out, "w"), ensure_ascii=False, indent=1)
    os._exit(0)


if __name__ == "__main__":
    main()
