#!/usr/bin/env python
# coding: utf-8

# ### Display

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

def plot_picture(PacketLoss, streaming_datasize, KpmReport_data, slice, reward):
    if not PacketLoss or not isinstance(PacketLoss[0], list):
        print("Error: global_PacketLoss is empty or not properly formatted.")
        return
    
    # print(f"global_PacketLoss = {PacketLoss}")
    # print(f"global_streaming_datasize = {streaming_datasize}")
    # print(f"global_KpmReport_data = {KpmReport_data}")
    
    time_axis_length = 5 * (len(PacketLoss)-1) if PacketLoss else 0
    time_axis = np.linspace(0, time_axis_length, len(PacketLoss) if PacketLoss else 1)

    # 第一張圖：數據流大小和KpmReport數據
    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, [sublist[0] for sublist in streaming_datasize], label='Video Streaming Server to (UE1)', color='blue')
    plt.plot(time_axis, [sublist[1] for sublist in streaming_datasize], label='Real-time Streaming Server (UE2)', color='darkblue')
    plt.plot(time_axis, [sublist[0] for sublist in KpmReport_data], label='Receive Video Data (UE1)', color='green')
    plt.plot(time_axis, [sublist[1] for sublist in KpmReport_data], label='Receive Real-time Data (UE2)', color='darkgreen')
    plt.legend()
    plt.title('Traffic Data Analysis Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Data Rate bits/s')
    plt.grid(True)
    plt.savefig('data_analysis_plot.png')  # 保存第一張圖
    plt.close()

    # 第二張圖：封包丟失率
    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, [sublist[0] for sublist in PacketLoss], label='Video Packet Loss rate of (UE1)', color='red')
    plt.plot(time_axis, [sublist[1] for sublist in PacketLoss], label='Real-time Packet Loss rates of (UE2)', color='darkred')
    plt.legend()
    plt.title('Packet Loss Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Packet Loss (bits)')
    plt.grid(True)
    plt.savefig('packet_loss_plot.png')  # 保存第二張圖
    plt.close()

    # 第三張圖：切片選擇
    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, [sublist[0] for sublist in slice], label='Slice of (UE1)', color='red')
    plt.plot(time_axis, [sublist[1] for sublist in slice], label='Slice of (UE2)', color='darkred')
    plt.yticks([64, 128, 256, 512, 1024])  # 設置縱軸的刻度
    plt.legend()
    plt.title('Slice Select')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Slice Size')
    plt.grid(True)
    plt.savefig('slice_select.png')  # 保存第二張圖
    plt.close()
    
    # 第三張圖：切片選擇
    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, reward, label='model reward', color='red')
    plt.legend()
    plt.title('Model reward')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.savefig('model_reward.png')  # 保存第二張圖
    plt.close()


# ### DQN 

# In[1]:


import numpy as np
from gym import spaces
import random
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque

def build_model(state_size, action_size):
    model = Sequential()
    model.add(Dense(128, input_dim=state_size, activation='linear'))
    model.add(Dense(128, activation='linear'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

class DQNAgent:
    def __init__(self):
        self.action_space = spaces.MultiDiscrete([5, 5])
        self.last_state = [0, 0, 4, 2]
        self.action = [4,2]
        self.reward = 0
        self.memory = deque(maxlen=1024*1024)

        self.gamma = 0.95    # discount rate
        self.epsilon = 1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.8
        
        self.memory_capacity = 10
        self.save_model = 17
        self.memory_counter = 0 
        
        self.real_PacketLoss = []
        self.streaming_datasize = []
        self.KpmReport_data = []
        self.slice = []
        self.reward_measurement = []

        self.ue1_weight = 1
        self.ue2_weight = 3
        self.ue1PacketLoss = 0
        self.ue2PacketLoss = 0

        self.model = build_model(4, 25)
        self.target_model = build_model(4, 25)
        # self.load("DQN_model_next_epoch.h5")
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def choose_action(self, state):
        """(policy)，選擇動作：探索或利用"""
        # state[0] = state[0] / (1024*1024)
        # state[1] = state[1] / (1024*1024)
        # print(f"convert state = {state}")
        # if((state[0] + state[1]) < 30):  
        #     action_intervals = [0.6, 1.1, 2, 3.5, 5, 7]
        #     sub_intervals = np.array(action_intervals)              

        #     idx = np.searchsorted(sub_intervals, state[1], side='right') + 2
        #     if(idx >= 6):
        #         action = [3,4]
        #     elif(idx == 5):
        #         action = [self.action[0], 4]
        #     else:
        #         action = [4, idx] 

        #     print(f"smaller: ue1 = {action[0]}, ue2 = {action[1]}")    
        # else:
        #     action_intervals = [1, 2, 4.3, 7, 14]
        #     sub_intervals = np.array(action_intervals)
        #     state[1] = state[1] * 3
        #     if(state[0] > state[1]):
        #         diff = state[0] / state[1] if state[1] != 0 else 15
        #         idx = np.searchsorted(sub_intervals, diff, side='right')
        #         if(idx == 5):
        #             action = [4,0]
        #         else:
        #             offset = idx%2
        #             # offset = 0
        #             action = [idx + offset, 4-idx + offset] 
        #     else:
        #         diff = state[1] / state[0] if state[0] != 0 else 15 
        #         idx = np.searchsorted(sub_intervals, diff, side='right')
        #         if(idx == 5):
        #             action = [0,4]
        #         else:
        #             offset = idx%2
        #             # offset = 0
        #             action = [4-idx + offset, idx + offset]
        #     print(f"larger: ue1 = {action[0]}, ue2 = {action[1]}")    
        
        # return action
        
        if random.uniform(0, 1) < self.epsilon:
            return self.action_space.sample() # 探索：隨機選擇動作
        else:
            state = np.array(state).reshape(1, -1)
            q_values = self.model.predict(state)
            print(f"q_values = {q_values}")
            action_index = np.argmax(q_values[0])
            print(f"action_index = {action_index}")
            x = math.floor(action_index / 5)  # 整數除法得到行索引
            y = action_index % 5   # 取餘數得到列索引
            action = [x, y]
            print(f"in choose action is {action}")
            return action  # 利用：選擇當前最佳動作
        
    def replay(self, batch_size):
        print("in reply")
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in minibatch:
            # print(f"state = {state}, action = {action}, reward = {reward}, next_state = {next_state}")
            state = np.array(state).reshape(1, -1)
            next_state = np.array(next_state).reshape(1, -1)
            target = self.model.predict(state)
            # print(f"target = {target}")
            t = self.target_model.predict(next_state)
            # print(f"t = {t}")
            action_1D = action[0] * 5 + action[1]
            target[0][action_1D] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
        
    # def packetLoss_function_linear_function(self, total_collected_data, report_data):
    #     current_ue1_packet_accumulation = self.ue1PacketLoss + total_collected_data[0] - report_data[0]
    #     current_ue2_packet_accumulation = self.ue2PacketLoss + total_collected_data[1] - report_data[1]
    #     packet_loss_list = [current_ue1_packet_accumulation, current_ue2_packet_accumulation]

    #     return packet_loss_list
        
    def packetLoss_calculate(self, report_data, total_collected_data):
        
        # 如果子列表不為空，獲取子列表的最後一個元素
        if (self.real_PacketLoss == []):
            last_list = [0, 0]
        else:
            last_list = self.real_PacketLoss[-1]
            
        ue1_current_packet_accmulation = last_list[0] + total_collected_data[0] - report_data[0]
        ue2_current_packet_accmulation = last_list[1] + total_collected_data[1] - report_data[1]
        if(ue1_current_packet_accmulation < 0): ue1_current_packet_accmulation = 0
        if(ue2_current_packet_accmulation < 0): ue2_current_packet_accmulation = 0
        print(f"ue1_current_packet_accmulation = {ue1_current_packet_accmulation}, ue2_current_packet_accmulation = {ue2_current_packet_accmulation}")
        
        packetloss_list = [ue1_current_packet_accmulation, ue2_current_packet_accmulation]
        return packetloss_list
    
    def data_preprocessing(self, report_data, total_collected_data):
        max_packet_accmulation = 9.5*5*1024*1024
        
        report_data = [x / max_packet_accmulation for x in report_data]
        total_collected_data = [y / max_packet_accmulation for y in total_collected_data]
                
        return report_data, total_collected_data
        
    def reward_function(self, report_data, total_collected_data):
        """(reward function)，計算reward"""
        ue1_reward = 0
        ue2_reward = 0
        
        ue1_current_packet_handle = report_data[0] - total_collected_data[0]
        ue2_current_packet_handle = report_data[1] - total_collected_data[1]
        self.ue1PacketLoss = self.ue1PacketLoss - ue1_current_packet_handle
        self.ue2PacketLoss = self.ue2PacketLoss - ue2_current_packet_handle
        print(f"ue1_current_packet_handle = {ue1_current_packet_handle}, ue2_current_packet_handle = {ue2_current_packet_handle}")
        print(f"self.ue1PacketLoss = {self.ue1PacketLoss}, self.ue2PacketLoss = {self.ue2PacketLoss}")
        
        # 特殊情況1，完美傳送
        if(self.ue1PacketLoss < 0): 
            self.ue1PacketLoss = 0
            ue1_reward = 1
        else:
            ue1_reward = ue1_current_packet_handle
            
        if(self.ue2PacketLoss < 0): 
            self.ue2PacketLoss = 0
            ue2_reward = 1
        else:
            ue2_reward = ue2_current_packet_handle
            
        # 特殊情況2，傳送資料超過總負荷 (streaming server直接設計不會有)
        
        # ue1 跟 ue2合併考慮
        x = abs(self.ue2PacketLoss*3 - self.ue1PacketLoss)
        if( x < 0.1):
            merge_reward = 1-100*(x**2)
        else:
            # x = x/600
            merge_reward = (200/361)*(x-0.1)*(x-3.9)
            if(x >= 1.5): 
                merge_reward = -1.5
                # self.ue1PacketLoss = self.ue1PacketLoss/2
            
        # 通用情況
        single_reward = (ue1_reward + ue2_reward) 
        reward = single_reward + merge_reward
        print(f"timestamp = {self.memory_counter*5}")
        print(f"action = {self.action}")
        print(f"ue1_reward = {ue1_reward}")
        print(f"ue2_reward = {ue2_reward}")
        print(f"x = {x}, merge_reward = {merge_reward}")
        print(f"total reward = {reward}")
        
        return reward

    def train(self, report_data, total_collected_data):
        packet_cal = self.packetLoss_calculate(report_data, total_collected_data)
        # print("real_packetLoss")
        cell_report, server_send= self.data_preprocessing(report_data, total_collected_data)
        print(f"cell_report = {cell_report}, server_send = {server_send}")
        
        state = [*cell_report, *self.action]
        # print(f"state = {state}")               
        self.reward = self.reward + self.reward_function(cell_report, server_send)
        # print("train1")
        self.remember(self.last_state, self.action, self.reward, state)
        # print("train2")
        self.last_state = state
        
        self.memory_counter += 1 
        if (((self.memory_counter%self.memory_capacity) == 0) and self.memory_counter > 10):
            self.update_target_model()
            self.replay(self.memory_capacity)
            
        # print("train3")
        if ((self.memory_counter%self.save_model) == 0):
            self.save(f'DQN_model_14_{self.memory_counter}.h5')           
        
        # packetLoss_list_linear_function = self.packetLoss_function_linear_function(report_data, total_collected_data)
        
        self.action = self.choose_action(state)
        # self.action = self.choose_action(packetLoss_list_linear_function)

        real_action_space = [64, 128, 256, 512, 1024]
        real_action = [real_action_space[self.action[0]] , real_action_space[self.action[1]]]        
        
        
        
        self.real_PacketLoss.append(packet_cal)
        # self.real_PacketLoss.append(packetLoss_list)
        self.streaming_datasize.append(total_collected_data)
        self.KpmReport_data.append(report_data)
        self.slice.append(real_action)
        self.reward_measurement.append(self.reward)
        plot_picture(self.real_PacketLoss, self.streaming_datasize, self.KpmReport_data, self.slice, self.reward_measurement)
        
        return real_action
    
    def inference(self, report_data, total_collected_data):
        
        state = report_data + self.action       
        self.reward = self.reward + self.reward_function(report_data, total_collected_data)
        self.action = self.choose_action(state)
        
        real_action_space = [64, 128, 256, 512, 1024]
        real_action = [real_action_space[self.action[0]] , real_action_space[self.action[1]]]

        packetLoss_list = self.packetLoss_function()
        self.real_PacketLoss.append(packetLoss_list)
        self.streaming_datasize.append(total_collected_data)
        self.KpmReport_data.append(report_data)
        self.slice.append(real_action)
        self.reward_measurement.append(self.reward)
        plot_picture(self.real_PacketLoss, self.streaming_datasize, self.KpmReport_data, self.slice, self.reward_measurement)
        
        return real_action


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
    # print(response.text)  # 打印API響應的內容
    # print()  # 額外的空行

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
    # print(response.text)  # 打印API響應的內容
    # print()  # 額外的空行


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
    # print(f'message = {message}')

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
                    # print(f"UE[{ue_index}] dl_bytes: {dl_bytes.group(1)}, dl_prbs: {dl_prbs.group(1)}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except KeyError as e:
        print(f"Key error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        
    # dl_bytes_list[0], dl_bytes_list[1] = dl_bytes_list[1], dl_bytes_list[0]
    return dl_bytes_list

def follow_log_file(log_file_path):
    try:
        agent = DQNAgent()
        process = subprocess.Popen(['tail', '-F', log_file_path], stdout=subprocess.PIPE, text=True)
        print("Starting to follow the log file...")
        while True:
            line = process.stdout.readline()  # 使用 readline 读取一行输出
            if not line:
                continue  # 如果没有读到数据，继续等待

            try:
                log_data = json.loads(line)
                log_message = json.loads(log_data['log'])
                if 'KpmReport' in log_message['msg']:
                    report_data = extract_kpm_report_data(line.strip())
                    # print(f"report_data = {report_data}")
                    total_collected_data = collect_and_display()
                    # print(f"total_collected_data = {total_collected_data}")                  
                    action = agent.train(report_data, total_collected_data)
                    # action = agent.inference(report_data, total_collected_data)
                    put_to_nexran_xapp_fast(action[0])
                    put_to_nexran_xapp_slow(action[1])

            except json.JSONDecodeError:
                pass
                # print("Error decoding JSON from log.")
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

    app.run(port=1212, debug=True, use_reloader=False)

