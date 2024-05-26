#!/bin/bash

SESSION="Double_UE_iperf_test"

# 如果已有同名的 tmux 會話，則刪除它
tmux has-session -t $SESSION 2>/dev/null
if [ $? == 0 ]; then
  tmux kill-session -t $SESSION
  sleep 1
fi

# 創建一個新的 tmux 會話，不附加到終端
tmux new-session -d -s $SESSION -n 'Double_UE_iperf_test'

# 開啟 tmux 的 mouse 模式
tmux set-option -g mouse on

# 在第一個窗格中執行命令並命名窗格
tmux send-keys -t $SESSION 'iperf -c 172.16.0.2 -p 8787 -n 10M' C-m
tmux select-pane -T 'server to ue1'

# 水平分割第一個窗格，並在新窗格中執行命令
tmux split-window -h
tmux send-keys -t $SESSION 'sudo ip netns exec ue1 iperf -s -p 8787' C-m
tmux select-pane -T 'netns ue1'

# 垂直分割左邊窗格，並在新窗格中執行命令
tmux split-window -v -t 0
tmux send-keys -t $SESSION 'iperf -c 172.16.0.3 -p 5487 -n 10M' C-m
tmux select-pane -T 'server to ue2'

# 垂直分割右邊窗格，並在新窗格中執行命令
tmux split-window -v -t 1
tmux send-keys -t $SESSION 'sudo ip netns exec ue2 iperf -s -p 5487' C-m
tmux select-pane -T 'netns ue2'

# 調整窗格布局
tmux select-layout tiled

# 回到第一個窗格
tmux select-pane -t 0

# 附加到會話
tmux attach-session -t $SESSION
