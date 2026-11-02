#!/bin/bash
# 诊断 Aurora 仿真为什么摔倒：实时性 / 多个 AuroraCore / CPU 资源。
# 用法（容器里，仿真跑过之后）：bash aurora_bridge/check_realtime.sh
LOG=${AURORA_LOG_DIR:-/tmp/aurora_logs}
echo "== CPU：容器可用核数 $(nproc)，负载 $(cut -d' ' -f1-3 /proc/loadavg)"
echo "== 正在运行的 AuroraCore 个数：$(pgrep -xc AuroraCore)（应为 0 或 1）"
ps -eo pcpu,comm --sort=-pcpu | head -6
if [ -f $LOG/core.log ]; then
  echo "== AuroraCore 任务超时次数（行走 50 Hz 预算 20 ms，PdStand 400 Hz 预算 2.5 ms）"
  grep "time out" $LOG/core.log | awk '{print $5}' | sed 's/time$//' | sort | uniq -c
  echo "   行走任务最慢的 5 次（秒）："
  grep "LowerBodyCpgTasktime out" $LOG/core.log | awk '{print $NF+0, $0}' | sort -rn | head -5 | awk '{print "   ", $1}'
  echo "== 多个 Core 抢同一 DDS 域的迹象：$(grep -c 'multiple publishers' $LOG/*.log 2>/dev/null | awk -F: '{s+=$2} END{print s+0}') 次"
fi
if [ -f $LOG/sim.log ]; then
  echo "== 仿真落后实时的报告（每 5 s 一次，>50 ms 才报）：$(grep -c lag $LOG/sim.log) 次"
  grep lag $LOG/sim.log | awk '{print $3+0}' | sort -n | awk '{a[NR]=$1} END{if(NR) print "   中位", a[int(NR/2)+1], "ms，最大", a[NR], "ms"}'
fi
echo
echo "判断：行走任务超时上百次、或单次 >0.05 s、或仿真 lag 经常 >100 ms → CPU 不够，机器人会无故摔倒。"
