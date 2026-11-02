"""
planning/geometry.py —— 桌子几何：站位计算、可达性判断、绕桌最短路径
======================================================================
这些是“经典方法”（几何 + 图搜索），不需要学习：
  * 站位 (standing pose)：机器人站在桌子某条边外侧、面朝桌子的位置。
  * 可达性：物体离这条桌边的距离 ≤ 手臂可达深度，且在右手的横向工作范围内。
  * 绕桌最短路径（测地距离）：把桌子按身体半径“膨胀”成障碍矩形，
    最短路径只可能经过膨胀矩形的 4 个角 → 在 {起点, 4 个角, 终点} 上做
    可见图 (visibility graph) + Dijkstra。

测地距离有两个用途：
  1. 规划器估计“去哪一侧更近”；
  2. 导航 PPO 的奖励塑形（势函数）。如果用直线距离做奖励，桌子挡在中间时
     机器人会被“往前冲”的奖励困在桌边（局部最优）；用绕桌的距离就没有这个陷阱。
"""
import heapq
from dataclasses import dataclass

import numpy as np

from config import RobotCfg, SceneCfg

# 手臂可达范围（机器人坐标系，由 IK 扫描实验得到）
REACH_DEPTH = 0.33          # 物体离“所站桌边”的最大距离
LAT_MIN, LAT_MAX = -0.30, 0.06   # 物体相对骨盆的横向偏移范围（右手：负值=右侧）
LAT_PREF = -0.12            # 首选横向偏移：物体在右前方最顺手


@dataclass
class StandPose:
    x: float
    y: float
    yaw: float
    side: str               # 站在哪一侧：+x / -x / +y / -y

    def as_array(self):
        return np.array([self.x, self.y, self.yaw])


class TableGeometry:
    def __init__(self, scene: SceneCfg = None, robot: RobotCfg = None, body_radius=0.17,
                 clearance=0.05):
        sc = scene or SceneCfg()
        rc = robot or RobotCfg()
        self.c = np.array(sc.table_pos, dtype=float)
        self.half = np.array(sc.table_half, dtype=float)
        self.standoff = rc.standoff
        self.inflate = body_radius + clearance      # 障碍膨胀量 = 身体半径 + 安全余量

    # ------------------------------------------------------------------
    # 站位与可达性
    # ------------------------------------------------------------------
    def sides(self):
        """每条桌边：(名字, 外法向量 n, 沿边方向 t, 边的半长)。"""
        hx, hy = self.half
        return [("+x", np.array([1.0, 0]), np.array([0, 1.0]), hy),
                ("-x", np.array([-1.0, 0]), np.array([0, -1.0]), hy),
                ("+y", np.array([0, 1.0]), np.array([-1.0, 0]), hx),
                ("-y", np.array([0, -1.0]), np.array([1.0, 0]), hx)]

    def stand_pose_for(self, target_xy, side):
        """针对某条边，计算抓取 target_xy 的最佳站位，并返回 (站位, 是否可达)。"""
        for name, n, t, half_len in self.sides():
            if name != side:
                continue
            half_n = self.half[0] if name.endswith("x") else self.half[1]
            edge_c = self.c + n * half_n                 # 该边中点
            rel = np.asarray(target_xy) - edge_c
            depth = -rel @ n                            # 物体离这条边多远（向桌内为正）
            along = rel @ t
            # 机器人面朝 -n；机器人左方向 = 朝向逆时针转 90°
            yaw = np.arctan2(-n[1], -n[0])
            left = np.array([-np.sin(yaw), np.cos(yaw)])
            # 让物体位于机器人右前方 LAT_PREF 处：物体横向 = (obj - pelvis)·left
            # pelvis 在边外 standoff，横向位置 s 沿 t：lateral = (along - s)*(t·left)
            tl = t @ left
            s = along - LAT_PREF / tl
            s = float(np.clip(s, -half_len - 0.15, half_len + 0.15))
            pelvis = edge_c + n * self.standoff + t * s
            lateral = (along - s) * tl
            ok = (0.0 <= depth <= REACH_DEPTH) and (LAT_MIN <= lateral <= LAT_MAX)
            return StandPose(pelvis[0], pelvis[1], yaw, name), ok, depth
        raise ValueError(side)

    def reachable_from(self, pose: StandPose, target_xy):
        """已经站在 pose 时，target_xy 能不能直接够到？"""
        c, s = np.cos(pose.yaw), np.sin(pose.yaw)
        d = np.asarray(target_xy) - np.array([pose.x, pose.y])
        fwd, lat = c * d[0] + s * d[1], -s * d[0] + c * d[1]
        return (self.standoff - 0.02 <= fwd <= self.standoff + REACH_DEPTH) and \
            (LAT_MIN <= lat <= LAT_MAX)

    def best_stand_pose(self, target_xy, robot_xy):
        """在所有“可达”的桌边中，选绕桌路程最短的那一个。"""
        cands = []
        for name, *_ in self.sides():
            pose, ok, depth = self.stand_pose_for(target_xy, name)
            if ok:
                cost = self.geodesic(robot_xy, pose.as_array()[:2])
                cands.append((cost, pose))
        if not cands:
            return None
        return min(cands, key=lambda c: c[0])[1]

    # ------------------------------------------------------------------
    # 绕桌最短路径
    # ------------------------------------------------------------------
    def corners(self):
        h = self.half + self.inflate + 1e-3
        return [self.c + np.array([sx * h[0], sy * h[1]]) for sx in (-1, 1) for sy in (-1, 1)]

    def _blocked(self, p, q):
        """线段 pq 是否穿过膨胀后的桌子矩形内部（Liang–Barsky 裁剪）。"""
        h = self.half + self.inflate
        p = np.asarray(p) - self.c
        d = np.asarray(q) - self.c - p
        t0, t1 = 0.0, 1.0
        for k in range(2):
            for pk, qk in ((-d[k], p[k] + h[k]), (d[k], h[k] - p[k])):
                if abs(pk) < 1e-12:
                    if qk < 0:
                        return False
                else:
                    t = qk / pk
                    if pk < 0:
                        t0 = max(t0, t)
                    else:
                        t1 = min(t1, t)
        return t1 - t0 > 1e-6

    def inside_inflated(self, p):
        return np.all(np.abs(np.asarray(p) - self.c) < self.half + self.inflate)

    def project_out(self, p):
        """若点在膨胀矩形内部，把它推到最近的边界外侧，返回 (新点, 推移距离)。"""
        p = np.asarray(p, float)
        rel = p - self.c
        h = self.half + self.inflate + 2e-3
        if np.all(np.abs(rel) < h):
            gap = h - np.abs(rel)
            k = int(np.argmin(gap))
            q = rel.copy()
            q[k] = np.sign(rel[k] if rel[k] != 0 else 1.0) * h[k]
            return self.c + q, float(gap[k])
        return p, 0.0

    def geodesic_path(self, start, goal):
        """返回绕开桌子的最短折线路径 (点列表) 和长度。"""
        start, extra0 = self.project_out(start)
        goal, extra1 = self.project_out(goal)
        nodes = [start] + self.corners() + [goal]
        n = len(nodes)
        dist = [np.inf] * n
        prev = [-1] * n
        dist[0] = 0.0
        pq = [(0.0, 0)]
        while pq:
            d0, i = heapq.heappop(pq)
            if d0 > dist[i]:
                continue
            if i == n - 1:
                break
            for j in range(1, n):
                if j == i:
                    continue
                if self._blocked(nodes[i], nodes[j]):
                    continue
                nd = d0 + np.linalg.norm(nodes[j] - nodes[i])
                if nd < dist[j]:
                    dist[j], prev[j] = nd, i
                    heapq.heappush(pq, (nd, j))
        if not np.isfinite(dist[-1]):          # 起点/终点在膨胀区内：退化为直线
            return [start, goal], float(np.linalg.norm(goal - start))
        path, k = [], n - 1
        while k != -1:
            path.append(nodes[k])
            k = prev[k]
        return path[::-1], float(dist[-1]) + extra0 + extra1

    def geodesic(self, start, goal):
        return self.geodesic_path(start, goal)[1]
