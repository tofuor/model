#!/bin/bash

while true
do
    # echo "Running iperf to 172.16.0.2 on port 8787"
    iperf -c 172.16.0.2 -p 8787 -n 0M
    sleep 1  # 等待1秒

    # echo "Running iperf to 172.16.0.3 on port 5487"
    iperf -c 172.16.0.3 -p 5487 -n 0M
    sleep 1  # 等待1秒
done
