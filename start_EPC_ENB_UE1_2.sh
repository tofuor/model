#!/bin/bash

SESSION="2UE"

# 如果已有同名的 tmux 會話，則刪除它
tmux has-session -t $SESSION 2>/dev/null
if [ $? == 0 ]; then
  tmux kill-session -t $SESSION
  sleep 3
fi
sleep 3

# 創建一個新的 tmux 會話，不附加到終端
tmux new-session -d -s $SESSION -n '2UE'

# 開啟 tmux 的 mouse 模式
tmux set-option -g mouse on

# 在第一個窗格中執行命令並命名窗格
tmux send-keys -t $SESSION 'srsepc' C-m
tmux select-pane -T 'srsEPC'
sleep 1

# 水平分割第一個窗格，並在新窗格中執行命令
tmux split-window -h
tmux send-keys -t $SESSION 'export E2NODE_IP=$(hostname  -I | cut -f1 -d" ")' C-m
tmux send-keys -t $SESSION 'export E2NODE_PORT=5006' C-m
tmux send-keys -t $SESSION 'export E2TERM_IP=$(sudo kubectl get svc -n ricplt --field-selector metadata.name=service-ricplt-e2term-sctp-alpha -o jsonpath="{.items[0].spec.clusterIP}")' C-m
tmux send-keys -t $SESSION 'sudo srsenb --enb.n_prb=100 --enb.name=enb1 --enb.enb_id=0x19B --rf.device_name=zmq --rf.device_args="fail_on_disconnect=true,tx_port=tcp://*:2000,rx_port=tcp://localhost:2009,id=enb,base_srate=23.04e6" --ric.agent.remote_ipv4_addr=${E2TERM_IP} --log.all_level=warn --ric.agent.log_level=debug --log.filename=stdout --ric.agent.local_ipv4_addr=${E2NODE_IP} --ric.agent.local_port=${E2NODE_PORT} --slicer.enable=1 --slicer.workshare=0' C-m
tmux select-pane -T 'srsENB'
sleep 1

# 垂直分割左邊窗格，並在新窗格中執行命令
tmux split-window -v -t 1
tmux send-keys -t $SESSION 'sudo ip netns add ue1' C-m
tmux send-keys -t $SESSION 'srsue \' C-m
tmux send-keys -t $SESSION '  --rf.device_name=zmq --rf.device_args="tx_port=tcp://*:2010,rx_port=tcp://localhost:2008,id=ue,base_srate=23.04e6" --usim.algo=xor --usim.imsi=001010123456789 --usim.k=00112233445566778899aabbccddeeff --usim.imei=353490069873310 --log.all_level=warn --log.filename=stdout --gw.netns=ue1' C-m
tmux select-pane -T 'UE1'
sleep 1

# 垂直分割右邊窗格，並在新窗格中執行命令
tmux split-window -v -t 1
tmux send-keys -t $SESSION 'sudo ip netns add ue2' C-m
tmux send-keys -t $SESSION 'srsue \' C-m
tmux send-keys -t $SESSION '  --rf.device_name=zmq --rf.device_args="tx_port=tcp://*:2007,rx_port=tcp://localhost:2006,id=ue,base_srate=23.04e6" --usim.algo=xor --usim.imsi=001010123456780 --usim.k=00112233445566778899aabbccddeeff --usim.imei=353490069873310 --log.all_level=warn --log.filename=stdout --gw.netns=ue2' C-m
tmux select-pane -T 'UE2'

# 調整窗格布局
tmux select-layout tiled

# 回到第一個窗格
tmux select-pane -t 0

# 附加到會話
tmux attach-session -t $SESSION
