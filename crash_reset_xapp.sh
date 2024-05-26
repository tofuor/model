sudo tmux kill-session -t 2UE
sudo tmux kill-session -t Double_UE_iperf_test
sudo kubectl -n ricxapp rollout restart deployment ricxapp-nexran
