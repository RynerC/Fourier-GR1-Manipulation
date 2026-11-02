#!/bin/bash
# 在 GR1P Aurora 镜像里启动：无界面仿真 (sim_server.py) + AuroraCore。
# 用法：bash aurora_bridge/start_stack.sh [sim_server 参数...]
#   例：bash aurora_bridge/start_stack.sh --scene tabletop --spawn -1.6,0,0 --truth-log /tmp/truth.csv
# 停止：bash aurora_bridge/stop_stack.sh
HERE=$(cd "$(dirname "$0")" && pwd)
LOG=${AURORA_LOG_DIR:-/tmp/aurora_logs}; mkdir -p $LOG
[ -f /env.sh ] && source /env.sh
bash "$HERE/stop_stack.sh" >/dev/null 2>&1
if pgrep -x AuroraCore >/dev/null; then echo "还有 AuroraCore 在运行，先停掉它"; exit 1; fi
python3 "$HERE/sim_server.py" "$@" > $LOG/sim.log 2>&1 &
echo $! > $LOG/sim.pid
sleep 2
cd /opt/fftai/fourier_aurora && ./bin/AuroraCore > $LOG/core.log 2>&1 &
echo $! > $LOG/core.pid
sleep 3
echo "started (logs in $LOG)"
