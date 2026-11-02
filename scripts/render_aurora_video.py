"""
scripts/render_aurora_video.py —— 把 sim_server.py --qpos-log 记录的轨迹离线渲染成 MP4
=========================================================================================
Aurora 仿真跑在无界面服务器 / 容器里时没有窗口可录屏：sim_server 以 30 Hz 记下完整 qpos
（和 'label:' 阶段字幕），这里逐帧回放 qpos → mj_forward → 离屏渲染 → 写视频。
需要 mujoco>=2.3.6 和离屏渲染（无显示器：MUJOCO_GL=egl 或 osmesa），Pillow，imageio + imageio-ffmpeg。
Aurora 镜像里没有 imageio，先装：  pip install imageio imageio-ffmpeg   （imageio-ffmpeg 自带 ffmpeg）
中文字幕需要中文字体：镜像里只有 DejaVu/Liberation，可以
  apt-get install -y fonts-noto-cjk        或   --font /workspace/msyh.ttc（把 C:/Windows/Fonts/msyh.ttc 拷进项目文件夹）
找不到中文字体时照样出视频，只是不画中文字幕（时间戳照画）。

  MUJOCO_GL=egl python scripts/render_aurora_video.py /tmp/aurora_logs/qpos_red_block.log \
      --xml /usr/local/resources/GR1T2/mjcf/scene_tabletop.xml --out videos/aurora_manip_task.mp4
"""
import argparse
import os

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONTS = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
         "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
         "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
         "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "/System/Library/Fonts/PingFang.ttc"]


def load_log(path):
    xml, T, L, Q, tray = None, [], [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#tray"):
                _, t, p = line.rstrip("\n").split("\t")
                tray.append((float(t), np.array(p.split(","), float)))
                continue
            if line.startswith("#"):
                xml = line.split("xml=", 1)[1].strip()
                continue
            t, lab, q = line.rstrip("\n").split("\t")
            T.append(float(t)); L.append(lab); Q.append(np.array(q.split(","), float))
    return xml, np.array(T), L, np.array(Q), tray


def find_cjk_font(user_font=None):
    for p in ([user_font] if user_font else []) + FONTS:
        if p and os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--xml", default=None, help="默认用日志头里记录的模型路径")
    ap.add_argument("--out", default="aurora_video.mp4")
    ap.add_argument("--size", default="960x540")
    ap.add_argument("--speed", type=float, default=1.0, help="播放倍速")
    ap.add_argument("--font", default=None, help="中文字体文件（.ttc/.ttf），不给就在常见位置找")
    ap.add_argument("--closeup-end", type=float, default=6.0, help="最后这么多秒（仿真时间）镜头拉近看托盘，0 = 不拉近")
    ap.add_argument("--tray", default=None, help="x,y：托盘位置（旧日志里没有记录托盘移动时手动指定）")
    ap.add_argument("--title", default="GR1T2 + Fourier Aurora（AuroraCore 控制，MuJoCo 物理）")
    a = ap.parse_args()
    xml0, T, L, Q, tray = load_log(a.log)
    m = mujoco.MjModel.from_xml_path(a.xml or xml0)
    d = mujoco.MjData(m)
    W, H = map(int, a.size.split("x"))
    m.vis.global_.offwidth, m.vis.global_.offheight = max(W, 640), max(H, 480)
    r = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = 3.6, -24.0, 135.0   # 镜头在桌子 +x、-y 一侧
    opt = mujoco.MjvOption()
    base = m.body("base_link").id
    fps = 30
    # 按仿真时间重采样：第 k 帧显示仿真时刻 T0 + k·speed/fps 的姿态（取最近的一条记录）。
    # 这样倍速与日志的实际记录频率无关（仿真卡顿、记录不足 30 Hz 时也准），也支持 1.5、0.5 这类倍速。
    t_frames = np.arange(T[0], T[-1] + 1e-9, a.speed / fps)
    idx = np.clip(np.searchsorted(T, t_frames), 0, len(T) - 1)
    prev = np.clip(idx - 1, 0, len(T) - 1)
    idx = np.where(np.abs(T[prev] - t_frames) < np.abs(T[idx] - t_frames), prev, idx)
    rate = (len(T) - 1) / max(T[-1] - T[0], 1e-9)
    print(f"日志：{T[-1] - T[0]:.1f} s 仿真时间，{len(T)} 条记录（约 {rate:.0f} Hz）→ 视频 {len(t_frames) / fps:.1f} s（{a.speed:g} 倍速，{fps} fps）")
    if rate < fps / a.speed * 0.8:
        print("[提示] 记录频率偏低，视频里同一姿态会重复几帧，看起来会卡顿（不影响倍速）")
    look = None
    try:
        import imageio.v2 as imageio
    except ImportError:
        raise SystemExit("缺少 imageio：先运行  pip install imageio imageio-ffmpeg")
    cjk = find_cjk_font(a.font)
    if cjk:
        f_big, f_small = ImageFont.truetype(cjk, 26), ImageFont.truetype(cjk, 18)
    else:
        print("[提示] 没找到中文字体，视频里不画中文字幕。装 fonts-noto-cjk 或用 --font 指定字体文件。")
        dv = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        f_big = f_small = ImageFont.truetype(dv, 18) if os.path.exists(dv) else ImageFont.load_default()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    writer = imageio.get_writer(a.out, fps=fps, codec="libx264", quality=7, macro_block_size=1)
    n = 0
    try:
        tray_bid = m.body("tray").id
    except KeyError:
        tray_bid = None
    if a.tray and tray_bid is not None:
        m.body_pos[tray_bid][:2] = [float(v) for v in a.tray.split(",")]
    elif tray and tray_bid is None:
        print("[提示] 日志里有托盘移动记录，但模型里没有托盘")
    for i in idx:
        if tray and tray_bid is not None:          # 托盘挪过：用当前时刻之前最后一次的位置
            for tt, p in tray:
                if tt <= T[i] + 1e-6:
                    m.body_pos[tray_bid] = p
        d.qpos[:] = Q[i][:m.nq]
        mujoco.mj_forward(m, d)
        p = d.xpos[base].copy()
        # 镜头：跟着骨盆走，但做低通，避免步态晃动带着画面抖；桌子中心也拉进画面
        tgt = np.array([0.55 * p[0], 0.55 * p[1], 0.75])
        dist, elev = 3.6, -24.0
        if tray_bid is not None and a.closeup_end > 0 and T[i] > T[-1] - a.closeup_end:
            tgt = d.xpos[tray_bid].copy()                 # 结尾特写：看清物体是不是在托盘上
            dist, elev = 1.0, -50.0
        look = tgt if look is None else 0.93 * look + 0.07 * tgt
        cam.lookat[:] = look
        cam.distance = 0.93 * cam.distance + 0.07 * dist
        cam.elevation = 0.93 * cam.elevation + 0.07 * elev
        r.update_scene(d, cam, opt)
        img = Image.fromarray(r.render())
        dr = ImageDraw.Draw(img)
        if cjk:
            dr.rectangle([0, 0, W, 44], fill=(0, 0, 0))
            dr.text((14, 8), L[i] or "", font=f_big, fill=(255, 255, 255))
            dr.text((14, H - 30), a.title, font=f_small, fill=(240, 240, 240))
        dr.text((W - 120, H - 30), f"t = {T[i] - T[0]:5.1f} s", font=f_small, fill=(240, 240, 240))
        writer.append_data(np.asarray(img))
        n += 1
    writer.close()
    try:
        r.close()                    # 主动释放 EGL 上下文，避免退出时打印一串无害的 EGLError
    except Exception:
        pass
    print(f"saved {a.out}: {n} 帧, {n / fps:.1f} s（{a.speed:g} 倍速）")


if __name__ == "__main__":
    main()
