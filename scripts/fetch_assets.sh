#!/bin/bash
# scripts/fetch_assets.sh —— 准备本仓库不分发的第三方内容（Fourier 的机器人模型和 SDK 文件）
# =============================================================================================
# 本仓库只包含自己写的代码、训练出的模型和实测结果。下面这些来自 Fourier，需要你自己从官方渠道获取：
#   assets/gr1t2/            GR1T2 模型（mjcf/gr1t2.xml + meshes/*.STL），取自 GR-1P Aurora SDK 镜像 /usr/local/resources/GR1T2
#   aurora_bridge/lcm_msgs/  simulator_pub_msg.py / simulator_sub_msg.py，取自 SDK 镜像 /workspace/aurora_sim
#                            （或 https://github.com/FFTAI/fourier_aurora_sdk 的 sim/mujoco/）
#
# 用法（宿主机，装了 docker 并已 docker load 过 SDK 镜像；在项目根目录运行）：
#   bash scripts/fetch_assets.sh <gr1p 镜像名>            # 例：bash scripts/fetch_assets.sh fourier_aurora_sdk_gr1p:v1.2.1
set -e
IMG=$1
[ -z "$IMG" ] && { echo "用法: bash scripts/fetch_assets.sh <gr1p 镜像名>    （docker images 查看镜像名）"; exit 1; }
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CID=$(docker create "$IMG")
trap 'docker rm -f "$CID" >/dev/null' EXIT
TMP=$(mktemp -d)

echo "[1/2] GR1T2 模型 → assets/gr1t2"
docker cp "$CID:/usr/local/resources/GR1T2/mjcf/gr1t2.xml" "$TMP/gr1t2.xml"
docker cp "$CID:/usr/local/resources/GR1T2/meshes" "$TMP/meshes"
mkdir -p "$ROOT/assets/gr1t2/mjcf" "$ROOT/assets/gr1t2/meshes"
cp "$TMP/gr1t2.xml" "$ROOT/assets/gr1t2/mjcf/"
cp "$TMP"/meshes/*.STL "$ROOT/assets/gr1t2/meshes/"

echo "[2/2] LCM 消息定义 → aurora_bridge/lcm_msgs"
for f in simulator_pub_msg.py simulator_sub_msg.py; do
    docker cp "$CID:/workspace/aurora_sim/$f" "$ROOT/aurora_bridge/lcm_msgs/$f"
done
rm -rf "$TMP"

echo "完成。快速操作环境的 assets/gr1t2/mjcf/manip_scene.xml 会在第一次训练/评估时自动生成。"
