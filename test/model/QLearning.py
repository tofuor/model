#!/usr/bin/env python
# coding: utf-8

# ### Display

# In[ ]:


# import matplotlib
# matplotlib.use('Agg')  # 使用Agg backend，它不需要GUI支持

import matplotlib.pyplot as plt
import numpy as np

def plot_picture(PacketLoss, streaming_datasize, KpmReport_data, slice):
    if not PacketLoss or not isinstance(PacketLoss[0], list):
        print("Error: global_PacketLoss is empty or not properly formatted.")
        return
    
    print(f"global_PacketLoss = {PacketLoss}")
    print(f"global_streaming_datasize = {streaming_datasize}")
    print(f"global_KpmReport_data = {KpmReport_data}")
    
    time_axis_length = 5 * (len(PacketLoss)-1) if PacketLoss else 0
    time_axis = np.linspace(0, time_axis_length, len(PacketLoss) if PacketLoss else 1)

    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, [sublist[0] for sublist in streaming_datasize], label='Video Streaming Server to (UE1)', color='blue')
    plt.plot(time_axis, [sublist[1] for sublist in streaming_datasize], label='Real-time Streaming Server (UE2)', color='darkblue')
    plt.plot(time_axis, [sublist[0] for sublist in KpmReport_data], label='Receive Data (UE1)', color='green')
    plt.plot(time_axis, [sublist[1] for sublist in KpmReport_data], label='Receive Data (UE2)', color='darkgreen')
    plt.plot(time_axis, [sublist[0] for sublist in PacketLoss], label='Packet Loss of (UE1)', color='red')
    plt.plot(time_axis, [sublist[1] for sublist in PacketLoss], label='Packet Loss of (UE2)', color='darkred')
    plt.legend()
    plt.title('Traffic Data Analysis Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Bits')
    plt.grid(True)
    plt.savefig('plot.png')  # 保存圖片到檔案，而不是顯示
    plt.close()


# ### QLearning Agent

# In[ ]:


import numpy as np
from gym import spaces
from collections import defaultdict
import random

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.action_space = spaces.MultiDiscrete([5, 5])
        self.q_table = defaultdict(lambda: np.zeros(25))
        self.last_state = [0,0]
        self.action = [0,0]
        
        self.PacketLoss = []
        self.streaming_datasize = []
        self.KpmReport_data = []
        self.slice = []
        
        self.alpha = alpha  # 學習率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索機率

    def choose_action(self, state):
        """(policy)，選擇動作：探索或利用"""
        if random.uniform(0, 1) < self.epsilon:
            return self.action_space.sample() # 探索：隨機選擇動作
        else:
            action_index = np.argmax(self.q_table[state])
            x = action_index // 5  # 整數除法得到行索引
            y = action_index % 5   # 取餘數得到列索引
            action = [x, y]
            return action  # 利用：選擇當前最佳動作

    def update_q_table(self, state, input_action, reward, next_state):
        """(value function)，更新 Q 表"""
        oneD_action = input_action[0]*5 + input_action[1]
        best_next_action = np.argmax(self.q_table[next_state])  # 下一狀態的最佳動作
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][oneD_action]
        self.q_table[state][oneD_action] += self.alpha * td_error
        
    def reward_function(self, report_data, total_collected_data):
        """(reward function)，計算reward"""
        if(report_data[0] > 0):
            Ue1PacketLoss = (total_collected_data[0] - report_data[0]) / report_data[0]
        else:
            Ue1PacketLoss = 0
            
        if(report_data[1] > 0):    
            Ue2PacketLoss = (total_collected_data[1] - report_data[1]) / report_data[1]
        else:
            Ue2PacketLoss = 0
        
        packet_loss_list = [Ue1PacketLoss, Ue2PacketLoss]
        
        return packet_loss_list

    def train(self, report_data, total_collected_data):
        # action = self.choose_action(state)
        # 等待5秒(step)
        # next_state = report_data
        # reward = self.reward_function(report_data, total_collected_data_ue1, total_collected_data_ue2)
        # self.update_q_table(state, action, reward, next_state)
        # state = next_state
        state = report_data
        reward = self.reward_function(report_data, total_collected_data)
        # self.update_q_table(self.last_state, self.action, reward, state)
        self.last_state = state
        # self.action = self.choose_action(state)
        self.action = [0,0]
        
        self.PacketLoss.append(reward)
        self.streaming_datasize.append(total_collected_data)
        self.KpmReport_data.append(report_data)
        self.slice.append(self.action)
        plot_picture(self.PacketLoss, self.streaming_datasize, self.KpmReport_data, self.slice)
        
        return self.action


# ### RAN control API

# In[ ]:


import requests
import subprocess

def put_to_nexran_xapp_fast(action):
    nexran_xapp_host = get_nexran_xapp_ip()
    
    url = f"http://{nexran_xapp_host}:8000/v1/slices/fast"
    headers = {'Content-type': 'application/json'}
    data = {
        "allocation_policy": {
            "type": "proportional",
            "share": action
        }
    }
    
    response = requests.put(url, json=data, headers=headers)
    print(response.status_code)  # 打印HTTP狀態碼
    print(response.text)  # 打印API響應的內容
    print()  # 額外的空行

def put_to_nexran_xapp_slow(action):
    nexran_xapp_host = get_nexran_xapp_ip()
    
    url = f"http://{nexran_xapp_host}:8000/v1/slices/slow"
    headers = {'Content-type': 'application/json'}
    data = {
        "allocation_policy": {
            "type": "proportional",
            "share": action
        }
    }
    
    response = requests.put(url, json=data, headers=headers)
    print(response.status_code)  # 打印HTTP狀態碼
    print(response.text)  # 打印API響應的內容
    print()  # 額外的空行


# 調用函式
def get_nexran_xapp_ip():
    try:
        # 執行 kubectl 命令並捕獲輸出
        cmd = [
            "kubectl", "get", "svc", "-n", "ricxapp",
            "--field-selector", "metadata.name=service-ricxapp-nexran-rmr",
            "-o", "jsonpath={.items[0].spec.clusterIP}"
        ]
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        return result.stdout.strip()  # 返回標準輸出，去除多餘空白
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return None


# ### Iperf message collect

# In[ ]:


from flask import Flask, request, jsonify
import threading
import time
import logging

app = Flask(__name__)

# 全局變數
collected_data_ue1 = []
collected_data_ue2 = []
last_calculate_time = time.time()

def collect_and_display():
    global collected_data_ue1, collected_data_ue2, last_calculate_time
    total_duration_ue1 = 0
    total_duration_ue2 = 0
    total_collected_data_ue1 = 0
    total_collected_data_ue2 = 0
    overflow_duration_ue1 = 0
    overflow_duration_ue2 = 0
    overflow_data_ue1 = 0
    overflow_data_ue2 = 0
        
    current_time = time.time()
    elapsed_time = current_time - last_calculate_time  # 計算時間差
    last_calculate_time = current_time  # 更新時間戳
    

    # 計算並顯示 ue1 的數據
    # -------------------------------------------------------------------------------------
    if collected_data_ue1:
        # print("in collected_data_ue1")
        for data in collected_data_ue1:
            total_duration_ue1 = total_duration_ue1 + data[2]
            total_collected_data_ue1 = total_collected_data_ue1 + data[0]
            
            if total_duration_ue1 > elapsed_time:
                overflow_duration_ue1 = total_duration_ue1 - elapsed_time
                overflow_data_ue1 =  data[0] * overflow_duration_ue1 / data[2]
                total_collected_data_ue1 = total_collected_data_ue1 - overflow_data_ue1
            else:
                overflow_duration_ue1 = 0
                overflow_data_ue1 = 0

        print(f"Total Collected Data for ue1: {total_collected_data_ue1}K, Total Collected duration for ue1: {elapsed_time} sec")
        # print(f"Overflow_data for Ue1 = {overflow_data_ue1}K, Overflow_duration for Ue1 = {overflow_duration_ue1} sec")
    
    

    # 計算並顯示 ue2 的數據
    # -------------------------------------------------------------------------------------
    if collected_data_ue2:
        # print("in collected_data_ue2")
        for data in collected_data_ue2:
            total_duration_ue2 = total_duration_ue2 + data[2]
            total_collected_data_ue2 = total_collected_data_ue2 + data[0]
            
            if total_duration_ue2 > elapsed_time:
                overflow_duration_ue2 = total_duration_ue2 - elapsed_time
                overflow_data_ue2 =  data[0] * overflow_duration_ue2 / data[2]
                total_collected_data_ue2 = total_collected_data_ue2 - overflow_data_ue2
            else:
                overflow_duration_ue2 = 0
                overflow_data_ue2 = 0

        print(f"Total Collected Data for ue2: {total_collected_data_ue2}K, Total Collected duration for ue2: {elapsed_time} sec")
        # print(f"Overflow_data for Ue2 = {overflow_data_ue2}K, Overflow_duration for Ue2 = {overflow_duration_ue2} sec")
    
    total_collected_data_list = [total_collected_data_ue1*1024, total_collected_data_ue2*1024]
    
    total_duration_ue1 = 0
    total_collected_data_ue1 = 0
    collected_data_ue1 = []
    collected_data_ue1.append((overflow_data_ue1, "ue1", overflow_duration_ue1))
    
    total_duration_ue2 = 0
    total_collected_data_ue2 = 0
    collected_data_ue2 = []
    collected_data_ue2.append((overflow_data_ue2, "ue2", overflow_duration_ue2))
    
    return total_collected_data_list

@app.route('/A1message', methods=['POST'])
def receive_message():
    global collected_data_ue1, collected_data_ue2
    data = request.json
    message = data['message']
    print(f'message = {message}')

    # 提取數據部分
    parts = message.split(' ')
    total_data = int(parts[1][:-1])  # 移除 'K' 並轉換為整數
    send_netns = parts[3][:-1]  # ue1 or ue2
    duration = float(parts[5])  # 轉換為浮點數
    # print(f'total_data = {total_data}, send_netns = {send_netns}, duration = {duration}')

    if send_netns == "ue1":
        collected_data_ue1.append((total_data, send_netns, duration))
        # print("collected_data_ue1")
    elif send_netns == "ue2":
        collected_data_ue2.append((total_data, send_netns, duration))
        # print("collected_data_ue2")

    return jsonify({"status": "Message received"}), 200


# ### Kpm Report log reader

# In[ ]:


import os
import glob
import re
import time
import subprocess
import json

def find_latest_log_file(base_pattern, pattern):
    directories = glob.glob(base_pattern) 
    for directory in directories:
        log_files = f"{directory}/{pattern}"
        print(f'log_files = {log_files}')
        return log_files

def extract_kpm_report_data(log_line):
    dl_bytes_list = []
    try:
        print("Finding KpmReport in the log...")
        outer_log = json.loads(log_line)  # 解析外层 JSON
        inner_log = json.loads(outer_log['log'])  # 解析内层 JSON

        if 'KpmReport' in inner_log['msg']:
            print("KpmReport found, processing data...")
            ue_data_pattern = re.compile(r'ue\[(\d+)\]=\{([^}]+)\}')  # 匹配 UE 数据
            matches = ue_data_pattern.finditer(inner_log['msg'])
            for match in matches:
                ue_index = match.group(1)
                ue_contents = match.group(2)
                dl_bytes = re.search(r'dl_bytes=(\d+)', ue_contents)
                dl_prbs = re.search(r'dl_prbs=(\d+)', ue_contents)
                if dl_bytes and dl_prbs:
                    dl_bytes_list.append(int(dl_bytes.group(1)))
                    print(f"UE[{ue_index}] dl_bytes: {dl_bytes.group(1)}, dl_prbs: {dl_prbs.group(1)}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except KeyError as e:
        print(f"Key error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return dl_bytes_list

def follow_log_file(log_file_path):
    try:
        agent = QLearningAgent()
        process = subprocess.Popen(['tail', '-F', log_file_path], stdout=subprocess.PIPE, text=True)
        print("Starting to follow the log file...")
        while True:
            line = process.stdout.readline()  # 使用 readline 读取一行输出
            if not line:
                continue  # 如果没有读到数据，继续等待

            # print(f"Processing line: {line.strip()}")
            try:
                log_data = json.loads(line)
                log_message = json.loads(log_data['log'])
                if 'KpmReport' in log_message['msg']:
                    report_data = extract_kpm_report_data(line.strip())
                    # print(f"report_data = {report_data}")
                    total_collected_data = collect_and_display()
                    # print(f"total_collected_data = {total_collected_data}")                  
                    action = agent.train(report_data, total_collected_data)
                    print("train test")
                    plot_picture()
                    
                    print(f"action = {action}")
                    
                    # put_to_nexran_xapp_fast(action[0])
                    # put_to_nexran_xapp_slow(action[1])

            # except json.JSONDecodeError:
                print("Error decoding JSON from log.")
            except Exception as e:
                # pass
                print(f"Error processing log line: {e}")

    except Exception as e:
        print(f"Error following the log file: {e}")
    finally:
        if process:
            process.terminate()
            print("Stopped following the log file.")


def log_file_thread():
    base_pattern = "/var/log/pods/ricxapp_ricxapp-nexran-*"
    file_pattern = "nexran-xapp/0.log"
    while True:
        latest_log_file = find_latest_log_file(base_pattern, file_pattern)
        # latest_log_file = "log_test.log"
        if latest_log_file:
            follow_log_file(latest_log_file)
        else:
            print("No log file found. Please check xApp")
            time.sleep(5)


# ### main function

# In[ ]:


if __name__ == '__main__': 
    
    global_PacketLoss = []
    global_streaming_datasize = []
    global_KpmReport_data = []
    global_slice = []
    
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    log_thread = threading.Thread(target=log_file_thread)
    log_thread.start()
    
    collection_thread = threading.Thread(target=collect_and_display)
    collection_thread.start()

    # task_thread = threading.Thread(target=plot_picture)
    # task_thread.start()

    app.run(port=1212, debug=True, use_reloader=False)

