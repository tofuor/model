#!/bin/bash

SESSION="netns_iperf_server"

# 如果已有同名的 tmux 會話，則刪除它
tmux has-session -t $SESSION 2>/dev/null
if [ $? == 0 ]; then
  tmux kill-session -t $SESSION
  sleep 1
fi

# 創建一個新的 tmux 會話，不附加到終端
tmux new-session -d -s $SESSION -n 'netns_iperf_server'

# 開啟 tmux 的 mouse 模式
tmux set-option -g mouse on

tmux send-keys -t $SESSION 'sudo ip netns exec ue1 iperf -s -p 8787' C-m
tmux select-pane -T 'netns ue1'

tmux split-window -v
tmux send-keys -t $SESSION 'sudo ip netns exec ue2 iperf -s -p 5487' C-m
tmux select-pane -T 'netns ue2'

# 調整窗格布局
tmux select-layout tiled

# 回到第一個窗格
tmux select-pane -t 0

# 附加到會話
tmux attach-session -t $SESSION
