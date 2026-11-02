#!/bin/bash
# 停止 start_stack.sh 启动的仿真和 AuroraCore。
# 按 pid 文件停（不用 pkill -f，避免误杀自己的 shell）；SIGTERM 3 s 后还在就 SIGKILL。
# 注意：残留的 AuroraCore 会和新的抢同一个 DDS 域（日志里出现 "matched multiple publishers"），
# 机器人会收到两套指令并摔倒 —— 所以这里顺带清理所有名为 AuroraCore 的进程。
LOG=${AURORA_LOG_DIR:-/tmp/aurora_logs}
for f in sim.pid core.pid; do [ -f $LOG/$f ] && kill $(cat $LOG/$f) 2>/dev/null; done
sleep 3
for f in sim.pid core.pid; do [ -f $LOG/$f ] && kill -9 $(cat $LOG/$f) 2>/dev/null; rm -f $LOG/$f; done
for p in $(pgrep -x AuroraCore); do kill -9 $p 2>/dev/null; done
echo stopped
