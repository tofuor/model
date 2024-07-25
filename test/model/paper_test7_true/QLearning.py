#!/usr/bin/env python
# coding: utf-8

# ### Display

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogFormatter, LogLocator

class SMAPlotType():
    def __init__(self): 
        self.RR_packetloss:list = []
        self.BET_packetloss:list = []
        self.DQN_packetloss:list = []
        
        self.video_packet_queue:list = []
        self.real_time_packetloss_rate:list = []
        self.video_PRB_utilization: list = []
        self.real_time_PRB_utilization: list = []
        
    def reset(self):
        self.RR_packetloss = []
        self.BET_packetloss = []
        self.DQN_packetloss = []
        
        self.video_packet_queue = []
        self.real_time_packetloss_rate = []
        self.video_PRB_utilization = []
        self.real_time_PRB_utilization = []
    
    def utilization_calculate(self, kpm_data:list, slice_select:list, max_base_station_throughput:int) -> list:
        '''
        video_PRB_utilization = video_throughput / max_throughput , 
        max_throughput = max_base_station_throughput * video_slice/(video_slice + real_time_slice)
        '''
        mapping = {0: 64, 1: 128, 2: 256, 3: 512, 4: 1024}
        video_max_throughput = max_base_station_throughput * mapping[slice_select[0]]/(mapping[slice_select[0]] + mapping[slice_select[1]])
        temp_video_PRB_utilization = kpm_data[0] / video_max_throughput
        real_time_max_throughput = max_base_station_throughput * mapping[slice_select[1]]/(mapping[slice_select[0]] + mapping[slice_select[1]])
        temp_real_time_PRB_utilization = kpm_data[1] / real_time_max_throughput
        
        return [temp_video_PRB_utilization, temp_real_time_PRB_utilization]

def plot_picture(KpmReport_data, streaming_datasize, PacketLoss, reward, slice, tag):
    if not PacketLoss or not isinstance(PacketLoss[0], list):
        print("Error: global_PacketLoss is empty or not properly formatted.")
        return
    
    # print(f"KpmReport_data = {KpmReport_data}")
    # print(f"streaming_datasize = {streaming_datasize}")
    print(f"PacketLoss = {PacketLoss}")
    # print(f"reward = {reward}")
    # print(f"slice = {slice}")
    
    time_axis_length = 5 * (len(PacketLoss)-1) if PacketLoss else 0
    time_axis = np.linspace(0, time_axis_length, len(PacketLoss) if PacketLoss else 1)
    
    if(tag%3 == 0):
        label = f"RR {tag//3}"
    elif(tag%3 == 1):
        label = f"BET {tag//3}"
    else:
        label = f"DDQN {tag//3}"
    
    # print("picture 1")

    # 第一張圖：數據流大小和KpmReport數據
    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, [sublist[0] for sublist in streaming_datasize], label='Video Streaming Server to (UE1)', color='darkblue')
    plt.plot(time_axis, [sublist[1] for sublist in streaming_datasize], label='Real-time Streaming Server (UE2)', color='darkviolet')
    plt.plot(time_axis, [sublist[0] for sublist in KpmReport_data], label='Video Data Receive (UE1)', color='darkgreen')
    plt.plot(time_axis, [sublist[1] for sublist in KpmReport_data], label='Real-time Data Receive (UE2)', color='darkorange')
    plt.legend()
    plt.title(f'Traffic Data Analysis Over Time ({label})')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Data Rate (bits/s)')
    plt.grid(True)
    plt.savefig(f'data_analysis_plot_{label}.png')  # 保存第一張圖
    plt.close()

    # print("picture 2")
    # 第二張圖：封包丟失率
    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, [sublist[0] for sublist in PacketLoss], label='Video Packet Loss rate of (UE1)', color='orangered')
    plt.plot(time_axis, [sublist[1] for sublist in PacketLoss], label='Real-time Packet Loss rates of (UE2)', color='darkred')
    
    # 设置 y 轴为对数刻度，底数为 10
    plt.yscale('log', base=10)
    # 设置 y 轴的刻度
    ticks = [10**i for i in range(11)]  # Generates [1, 10, 100, ..., 10^10]
    plt.yticks(ticks, [f"10^{i}" for i in range(11)])  # Labels as 10^0, 10^1, ..., 10^10
    
    plt.legend()
    plt.title(f'Packet Loss Over Time ({label})')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Log10 of Packet Loss (bits)')
    plt.grid(True)
    plt.savefig(f'packet_loss_plot_{label}.png')  # 保存第二張圖
    plt.close()

    # print("picture 3")
    # 第三張圖：切片選擇
    mapping = {0: 64, 1: 128, 2: 256, 3: 512, 4: 1024}
    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, [mapping[sublist[0]] for sublist in slice], label='Slice of (UE1)', color='orangered')
    plt.plot(time_axis, [mapping[sublist[1]] for sublist in slice], label='Slice of (UE2)', color='darkred')
    
    plt.yscale('log', base=2)  # 設置 y 軸為對數刻度，底數為 2
    # 設定對數刻度的標籤
    powers = np.arange(6, 11)  # 從 2^6 到 2^10
    ticks = 2 ** powers  # 計算 2 的冪次值
    labels = [f"2^{p}" for p in powers]  # 生成標籤 2^6, 2^7, 等...

    plt.yticks(ticks, labels)  # 設置 y 軸的刻度和標籤
    plt.legend()
    plt.title(f'Slice Select ({label})')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Log2 of Slice Size')
    plt.grid(True)
    plt.savefig(f'slice_select_{label}.png')  # 保存第二張圖
    plt.close()
    
    # print("picture 4")
    # 第四張圖：模型獎勵
    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, reward, label='model reward', color='red')
    plt.legend()
    plt.title(f'Model reward ({label})')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.savefig(f'model_reward_{label}.png')  # 保存第二張圖
    plt.close()
    
    # Packet Queue Length, Video Packet Packet Queue Length (SMA)
    
def sma_plot_preprocess(data:list, window_size:int, plot_label:str, title:str, ylabel:str, save_name:str, ylog_enabled:int):
    
    underload_time = 117
    print(f"data[0] = {data[0]}")
    print(f"data[1] = {data[1]}")
    print(f"data[2] = {data[2]}")
    data_length = min(len(data[0]), len(data[1]), len(data[2]))
    print(f"data_length = {data_length}")
    
    df_sma_data = pd.DataFrame(data, index=['RR', 'BET', 'DDQN'])
    
    
    length = min(len(df_sma_data.loc['RR']), len(df_sma_data.loc['BET']), len(df_sma_data.loc['DDQN']))
    print(f"length = {length}")
    ticks = np.arange(length) * 5
    
    # 計算SMA
    RR_sma = df_sma_data.loc['RR'].rolling(window=window_size, min_periods=1).mean()
    BET_sma = df_sma_data.loc['BET'].rolling(window=window_size, min_periods=1).mean()
    DDQN_sma = df_sma_data.loc['DDQN'].rolling(window=window_size, min_periods=1).mean()
    
    plt.figure(figsize=(10, 5))
    plt.plot(ticks, RR_sma[:length], label=f'RR {plot_label}', color='darkorange')
    plt.plot(ticks, BET_sma[:length], label=f'BET {plot_label}', color='olive')
    plt.plot(ticks, DDQN_sma[:length], label=f'DDQN {plot_label}', color='darkred')
    
    # save result
    with open('packet_loss_data.txt', 'a') as file:
        file.write(f"{save_name}, All RR data = {data[0]}\n")
        file.write(f"{save_name}, All BET data = {data[1]}\n")
        file.write(f"{save_name}, All DDQN data = {data[2]}\n")
        if(ylog_enabled):
            file.write(f"{save_name}, (underload) RR = {data[0][underload_time]}, BET = {data[1][underload_time]}, DDQN = {data[2][underload_time]}\n")
            file.write(f"{save_name}, (last) RR = {data[0][data_length-1]}, BET = {data[1][data_length-1]}, DDQN = {data[2][data_length-1]}\n")
        else:
            RR_average_underload = sum(data[0][:underload_time-1]) / underload_time
            BET_average_underload = sum(data[1][:underload_time-1]) / underload_time
            DDQN_average_underload = sum(data[2][:underload_time-1]) / underload_time
            file.write(f"{save_name}, (underload) RR = {RR_average_underload}, BET = {BET_average_underload}, DDQN = {DDQN_average_underload}\n")
            RR_average_full_time = sum(data[0]) / len(data[0])
            BET_average_full_time = sum(data[1]) / len(data[1])
            DDQN_average_full_time = sum(data[2]) / len(data[2])
            file.write(f"{save_name}, (last) RR = {RR_average_full_time}, BET = {BET_average_full_time}, DDQN = {DDQN_average_full_time}\n")
    
    plt.legend(prop={'size': 15})
    plt.title(f'{title}')
    plt.xlabel('Time (seconds)')
    
    if(ylog_enabled):    
        # 设置 y 轴为对数刻度，底数为 10
        plt.yscale('log', base=10)
        # 设置 y 轴的刻度
        ticks = [10**i for i in range(6, 10)]  # Generates [1, 10, 100, ..., 10^8]
        plt.yticks(ticks, [f"10^{i}" for i in range(6, 10)])  # Labels as 10^0, 10^1, ..., 10^8
        
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(save_name)
    plt.close()
    
def SMA_plot_picture(sma_plot:SMAPlotType, tag:int):
    
    video_window_size = 1
    window_size = 20
    
    
    sma_plot_preprocess(sma_plot.video_packet_queue, video_window_size, "Packet Queue Length", "Video Packet Packet Queue Length (SMA)", "Log10 of Packet Queue length (bits)", f"Video_packet_loss_plot_SMA_{tag}.png", 1)
    sma_plot_preprocess(sma_plot.real_time_packetloss_rate, window_size, "Packet Loss", "Real time Packet Loss Over Time (SMA)", "Log10 of Packet Loss (bits)", f"Real-time_packet_loss_plot_SMA_{tag}.png", 0)
    sma_plot_preprocess(sma_plot.video_PRB_utilization, window_size, "PRB_utilization", "Video PRB_utilization ", "PRB_utilization (%)", f"Video_PRB_utilization_{tag}.png", 0)
    sma_plot_preprocess(sma_plot.real_time_PRB_utilization, window_size, "PRB_utilization", "Real time PRB_utilization ", "PRB_utilization (%)", f"Real_time_PRB_utilization_{tag}.png", 0)


# ### Other Methed

# In[ ]:


import math

class RoundRobin:
    ''' This function make sure UE1 & UE2 have fair resource '''
    def __init__(self):
        self.slice = 4
        
    def choose_action(self):
        return [self.slice, self.slice]

class BlindEqualThroughput:
    ''' This function consider UE1 & UE2 average throughput '''
    def __init__(self):
        self.ue1_average_throughput = 0
        self.ue2_average_throughput = 0
        self.statistcs_time = 0
        
    def choose_action(self, kpm_data):
        self.statistcs_time += 1
        self.ue1_average_throughput = ((self.ue1_average_throughput*(5)) + kpm_data[0]) / 6
        self.ue2_average_throughput = ((self.ue2_average_throughput*(5)) + kpm_data[1]) / 6
        
        if(self.ue1_average_throughput > self.ue2_average_throughput):
            diff = self.ue1_average_throughput / self.ue2_average_throughput if self.ue2_average_throughput != 0 else 16
            
            limited_result = max(1, min(16, diff))
            log_result = math.log2(limited_result)
            rounded_result = round(log_result)            
            action = [4, 4-rounded_result] 
        else:
            diff = self.ue2_average_throughput / self.ue1_average_throughput if self.ue1_average_throughput != 0 else 16
            
            limited_result = max(1, min(16, diff))
            log_result = math.log2(limited_result)
            rounded_result = round(log_result)            
            action = [4-rounded_result, 4] 
        
        return action


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
        # define action、memory、model_size
        self.action_space = spaces.MultiDiscrete([5, 5])
        self.model = build_model(4, 25)
        self.target_model = build_model(4, 25)
        self.memory = deque(maxlen=1024*1024)
        # define super parameters
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.15  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9
        self.memory_capacity = 10

        # init        
        self.agent_init()

    def agent_init(self):
        # model training parameter
        self.accumluation_reward = 0
        self.last_state = [0, 0, 4, 4]
        print(f"self.epsilon = {self.epsilon}")
        if(self.epsilon != 1):  self.load("DQN_model_next_epoch.h5")
        self.epsilon *= self.epsilon_decay
        self.memory_counter = 0
        # model parameter
        
        self._update_target_model()
        
        self.temp_epsilon = self.epsilon
        
    def step_reset(self):
        self.step = 0

    def _update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def _remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def choose_action(self, input_state):
        """(policy)，選擇動作：探索或利用"""        
        if random.uniform(0, 1) < self.epsilon:
            action = self.action_space.sample()
            return action.tolist() # 探索：隨機選擇動作
        else:
            state = np.array(input_state).reshape(1, -1)
            q_values = self.model.predict(state)
            # print(f"q_values = {q_values}")
            action_index = np.argmax(q_values[0])
            # print(f"action_index = {action_index}")
            x = math.floor(action_index / 5)  # 整數除法得到行索引
            y = action_index % 5   # 取餘數得到列索引
            action = [x, y]
            # print(f"in choose action is {action}")
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
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def train(self, kpm_data, action, reward):
        # self.step += 1
        # if(self.step < 120):    self.epsilon = 0
        # else:                   self.epsilon = self.temp_epsilon
            
        state = [*kpm_data, *action]
        # print(f"state = {state}")               
        self.accumluation_reward = self.accumluation_reward + reward  # 累積獎勵
        # print("train1")
        self._remember(self.last_state, action, self.accumluation_reward, state)
        # print("train2")
        self.last_state = state
        
        self.memory_counter += 1 
        if (((self.memory_counter%self.memory_capacity) == 0) and self.memory_counter > 10):
            self._update_target_model()
            self.replay(self.memory_capacity)
        
        predict_action = self.choose_action(state)
        
        return predict_action


# ### RAN control API

# In[ ]:


import requests
import subprocess

class RANControl:
    def __init__(self):
        self.xappIP = self.get_nexran_xapp_ip()
    
    def covert_real_action(self, action):
        real_action_space = [64, 128, 256, 512, 1024]
        return [real_action_space[action[0]], real_action_space[action[1]]]
        
    def put_to_nexran_xapp(self, action, slice_name):
                
        url = f"http://{self.xappIP}:8000/v1/slices/{slice_name}"
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
    def get_nexran_xapp_ip(self):
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


# ### Scheduler

# In[ ]:


import copy
class PacketLossType():
    '''
    '''
    def __init__(self):
        self.accumulation: list = [0,0]
        self.real_time_current: int = 0
        
    def reset(self):
        self.accumulation: list = [0,0]
        self.real_time_current: int = 0
        
        
class PlotType():
    def __init__(self):
        self.kpmReport_data: list = []
        self.iperf_data: list = []
        self.packetloss_accumulation: list = []
        self.model_reward: list = []
        self.slice_select: list = []
        
    def reset(self):
        self.kpmReport_data = []
        self.iperf_data = []
        self.packetloss_accumulation = []
        self.model_reward = []
        self.slice_select = []

class Scheduler:
    def __init__(self):
        self.epoch:int = 0
        self.server_time:int = 0
        self.best_reward:int = -1000
        self.video_max_packet_current:int = 7*1024*1024*5
        self.real_time_max_packet_current:int = 6*1024*1024*5
        self.max_base_station_throughput:int = 13.5*1024*1024*5
        
        # method input addition define
        self.last_action:list = [0,0]
        
        # plot storage
        self.plot = PlotType()
        self.sma_plot = SMAPlotType()
        self.temp_sma_plot = SMAPlotType()
        # self.temp_sma_plot = []
        
        # scheduler
        self.DQN = DQNAgent()
        self.RR = RoundRobin()               # null
        self.BET = BlindEqualThroughput()    # average_throughput
        
        # temp
        self.packetloss = PacketLossType()
        self.reward = 0
        self.reward_accmulation = 0
        self.init_state_dont_store = 0 
        
    def _packetloss_calculate(self, kpmReport_data:list, iperf_data:list, packetloss:PacketLossType) -> PacketLossType:
        ''' 
            輸出過往累積的packetloss量 
        '''
        temp_packetloss = PacketLossType()
        # 如果子列表不為空，獲取子列表的最後一個元素
        print("in packetloss calculate")
        if (packetloss.accumulation == []):
            last_list = [1, 1]
        else:
            last_list = packetloss.accumulation
        
        temp_packetloss.real_time_current = iperf_data[1] - kpmReport_data[1] 
        if(temp_packetloss.real_time_current <= 0): 
            temp_packetloss.real_time_current = 1
        
        temp_packetloss.accumulation[0] = last_list[0] + iperf_data[0] - kpmReport_data[0]
        if(temp_packetloss.accumulation[0] <= 0): temp_packetloss.accumulation[0] = 1 
               
        temp_packetloss.accumulation[1] = last_list[1] + temp_packetloss.real_time_current
       
        print(f"ue1_current_packet_accmulation = {temp_packetloss.accumulation[0]}, ue2_current_packet_accmulation = {temp_packetloss.accumulation[1]}, ue2_current_packet = {temp_packetloss.real_time_current}")

        return temp_packetloss
    
    def _data_preprocessing(self, data:list, max_video_data:int, max_real_time_data:int) -> list:       
        normalized_data = [data[0]/max_video_data, data[1]/max_real_time_data]
                
        return normalized_data
    
    def _jains_fairness_index(self, x: list) -> float:
        x = np.array(x)  # 將列表轉換為 NumPy 陣列
        return (np.sum(x) ** 2) / (len(x) * np.sum(x ** 2))
    
    def _reward_function(self, kpm_data:list, iperf_data:list, packetloss_accumulation:PacketLossType, slice_select:list) -> int:
        """ 輸出目前狀態評分出來的reward，而不是累積reward """
        
        normalized_kpm_data = self._data_preprocessing(kpm_data, self.video_max_packet_current, self.real_time_max_packet_current)
        normalized_iperf_data = self._data_preprocessing(iperf_data, self.video_max_packet_current, self.real_time_max_packet_current)
        
        real_action_space = [64, 128, 256, 512, 1024]
        real_action_PRB = [real_action_space[slice_select[0]], real_action_space[slice_select[1]]]
        PRB_utilization = [kpm_data[0]/real_action_PRB[0], kpm_data[1]/real_action_PRB[1]]
        PRB_fairness_reward = self._jains_fairness_index(PRB_utilization)*4-3
        
        # 
        ue1_current_packet_handle = normalized_iperf_data[0] - normalized_kpm_data[0] - (1/7)
        ue2_current_packet_handle = normalized_iperf_data[1] - normalized_kpm_data[1]
        # ue1_current_packet_handle = normalized_kpm_data[0] - normalized_iperf_data[0]
        # ue2_current_packet_handle = normalized_kpm_data[1] - normalized_iperf_data[1]
        print(f"ue1_current_packet_handle = {ue1_current_packet_handle}, ue2_current_packet_handle = {ue2_current_packet_handle}")
        print(f"ue1_packetloss_accumulation = {packetloss_accumulation[0]}, ue2_packetloss_accumulation = {packetloss_accumulation[1]}")
        # if(ue2_current_packet_handle):
        #     print(f"ue2_packetloss rate = {normalized_kpm_data[1]/normalized_iperf_data[1] * 100}%")
        
        # 特殊情況1，
        if(packetloss_accumulation[0] <= 1):    ue1_reward = 1
        else:                                   ue1_reward = 0-ue1_current_packet_handle
            
        if(ue2_current_packet_handle <= 0):     ue2_reward = 1
        else:                                   ue2_reward = 1-2*ue2_current_packet_handle
            
        reward = ue1_reward + ue2_reward + PRB_fairness_reward
        print(f"ue1_reward = {ue1_reward}")
        print(f"ue2_reward = {ue2_reward}")
        print(f"total reward = {reward}")
        
        return reward
        
    def _decide_method(self, kpm_data:list, last_action:list, reward:int) -> list:
        ''' 
            決定要使用哪種演算法，每epoch一個method
            0 : Blind Equal Throughput
            1 : RoundRobin
            2 : DQN
            
            output = action
        '''      
        
        if (self.epoch%3) == 0:
            action = self.RR.choose_action()
            print("RR decision")
            # normalized_kpm_data = self._data_preprocessing(kpm_data, self.video_max_packet_current, self.real_time_max_packet_current)
            # action = self.DQN.train(normalized_kpm_data, last_action, reward)
            # print("DQN decision")
            
        elif (self.epoch%3) == 1:
            action = self.BET.choose_action(kpm_data)
            print("BET decision")
            # normalized_kpm_data = self._data_preprocessing(kpm_data, self.video_max_packet_current, self.real_time_max_packet_current)
            # action = self.DQN.train(normalized_kpm_data, last_action, reward)
            # print("DQN decision")
            
        elif (self.epoch%3) == 2:
            normalized_kpm_data = self._data_preprocessing(kpm_data, self.video_max_packet_current, self.real_time_max_packet_current)
            action = self.DQN.train(normalized_kpm_data, last_action, reward)
            print("DQN decision")
        else:
            action = [4,4]
            
        return action
    
    def _step_display(self, kpm_data:list, iperf_data:list, packetloss:PacketLossType, reward:int, action:list):
        
        if(self.plot.model_reward): model_reward_accumulation = self.plot.model_reward[-1] + reward
        else:                       model_reward_accumulation = reward
        
        temp_PRB_utilization = self.temp_sma_plot.utilization_calculate(kpm_data, action, self.max_base_station_throughput)
        
        self.plot.kpmReport_data.append(kpm_data)
        self.plot.iperf_data.append(iperf_data)
        self.plot.packetloss_accumulation.append(packetloss.accumulation)
        self.plot.model_reward.append(model_reward_accumulation)
        self.plot.slice_select.append(action)
        
        self.temp_sma_plot.video_packet_queue.append(packetloss.accumulation[0])
        self.temp_sma_plot.real_time_packetloss_rate.append(packetloss.accumulation[1])
        self.temp_sma_plot.video_PRB_utilization.append(temp_PRB_utilization[0])
        self.temp_sma_plot.real_time_PRB_utilization.append(temp_PRB_utilization[1])
        
        plot_picture(self.plot.kpmReport_data, self.plot.iperf_data, self.plot.packetloss_accumulation, self.plot.model_reward, self.plot.slice_select, self.epoch)
            
    def _execute_action(self, action:list):
        ''' 根據method action打API到nexran_xapp '''
        RAN_control = RANControl()
        if action:
            real_action = RAN_control.covert_real_action(action)
            RAN_control.put_to_nexran_xapp(real_action[0], "fast")
            RAN_control.put_to_nexran_xapp(real_action[1], "slow")
        
    def run(self, kpm_data:list, iperf_data:list):
        ''' 
            1. 計算packetloss, output = UE1、UE2累積packetloss、UE2瞬時packetloss
            2. 計算reward
            3. 根據目前的state、reward決定輸出動作
            4. 繪製圖表
            5. 執行action
        '''
        # print(f"in run, self.kpmReport_data = {self.kpmReport_data}")
        self.packetloss = self._packetloss_calculate(kpm_data, iperf_data, self.packetloss)        
        # print(f"pocketloss success, packetloss = {self.packetloss}")
        self.reward = self._reward_function(kpm_data, iperf_data, self.packetloss.accumulation, self.last_action)
        self.reward_accmulation = self.reward_accmulation + self.reward
        # print(f"reward success, self.kpmReport_data = {self.kpmReport_data}")         
        action = self._decide_method(kpm_data, self.last_action, self.reward)
        # print(f"action success, self.kpmReport_data = {self.kpmReport_data}")       
        self._step_display(kpm_data, iperf_data, self.packetloss, self.reward, action)
        # print(f"step_display, self.kpmReport_data = {self.kpmReport_data}")
        self._execute_action(action)
        # print(f"execute_action success, self.kpmReport_data = {self.kpmReport_data}")

# ------------------------------------------------------------------------------------------------------------------------------------        
        
    def _all_scheduler_display(self, sma_plot_packetloss:SMAPlotType):
        ''' 繪製三種方法的圖表 + 計算real-time累積packetloss '''
        SMA_plot_picture(sma_plot_packetloss, self.server_time)

        print("after SMA_plot_picture")
        self.server_time += 1
        
    def store_plot(self):
        
        temp = SMAPlotType()
        temp = copy.deepcopy(self.temp_sma_plot)
        
        self.sma_plot.video_packet_queue.append(temp.video_packet_queue)
        # print(f"sma_plot.video_packet_queue = {self.sma_plot.video_packet_queue}")
        self.sma_plot.real_time_packetloss_rate.append(temp.real_time_packetloss_rate)
        self.sma_plot.video_PRB_utilization.append(temp.video_PRB_utilization)
        self.sma_plot.real_time_PRB_utilization.append(temp.real_time_PRB_utilization)
        
        if (self.epoch%3 == 2):
            # plot SMA 
            self._all_scheduler_display(self.sma_plot)
            self.sma_plot.reset()
            
            # save model
            if(self.best_reward < self.reward_accmulation):
                # if(self.epoch > 0): self.DQN.save("DQN_model_next_epoch.h5")
                self.DQN.save("DQN_model_next_epoch.h5")
                self.best_reward = self.reward_accmulation
                print(f"best_reward = {self.best_reward}, save model")
            self.DQN.agent_init()
            
        self.init_state_dont_store = 0 
            
                            
    def next_epoch(self):
        self.plot.reset()
        self.temp_sma_plot.reset()
        self.packetloss.reset()
        self.reward_accmulation = 0
        self.reward = 0
        self.epoch += 1
        self.DQN.step_reset()


# ### Iperf message collect

# In[2]:


from flask import Flask, request, jsonify
import threading
import time
import logging

app = Flask(__name__)

class CalculateDataType:
    def __init__(self):
        self.overflow_data: int = 0
        self.overflow_duration: int = 0
        self.collected_data: int = 0


class IperfServer:
    def __init__(self):
        
        self.iperf_temp_data_ue1 = []
        self.iperf_temp_data_ue2 = []
        self.last_calculate_time = time.time()
        self.next_epoch_flag = False
        self.plot_flag = False
        
    def _calculate_data(self, collected_data: list, elapsed_time: int) -> CalculateDataType:
        ''' 
            collected_data = self.iperf_temp_data_ue1 = [{data_volume, "ue1", next_send_interval}, {data_volume, "ue1", next_send_interval}, ...]
        '''
        temp_total_duration = 0
        res = CalculateDataType()

        for data in collected_data:
            temp_total_duration += data[2]
            res.collected_data += data[0]
            
            # 因為是server會先告知資料大小和下次傳輸時間，讓基站決定排程
            # 當收集到的資料總時間 > KpmReport的固定時間，累積到接近kpmReport的interval，將超過的資料和時間設為overflow合併於下次計算
            if temp_total_duration > elapsed_time:
                res.overflow_duration = temp_total_duration - elapsed_time
                res.overflow_data = data[0] * res.overflow_duration / data[2]
                
                res.collected_data -= res.overflow_data
        
        return res
    
    def _reset_data(self, ue1:CalculateDataType , ue2:CalculateDataType ):
        self.iperf_temp_data_ue1 = []
        self.iperf_temp_data_ue2 = []
        self.iperf_temp_data_ue1.append((ue1.overflow_data, "ue1", ue1.overflow_duration))
        self.iperf_temp_data_ue2.append((ue2.overflow_data, "ue2", ue2.overflow_duration))
        
    def collect_iperf_data(self):
        current_time = time.time()
        elapsed_time = current_time - self.last_calculate_time
        self.last_calculate_time = current_time

        ue1 = self._calculate_data(self.iperf_temp_data_ue1, elapsed_time)
        ue2 = self._calculate_data(self.iperf_temp_data_ue2, elapsed_time)

        # Reset and store overflow data
        self._reset_data(ue1, ue2)

        return [ue1.collected_data * 1024, ue2.collected_data * 1024]
    

    def detect_streaming_restart(self):
        return self.next_epoch_flag
    
    def plot_flag_detect(self):
        return self.plot_flag


# ### Kpm Report log reader

# In[ ]:


import os
import glob
import re
import time
import subprocess
import json


class KpmReportLogReader:
    def __init__(self):
        self.base_pattern = "/var/log/pods/ricxapp_ricxapp-nexran-*"
        self.file_pattern = "nexran-xapp/0.log"
        self.dl_bytes_list = []
        self.temp_epoch = -1
        self.iperf_server = IperfServer()
        self.method_scheduler = Scheduler()

        
    def _find_latest_log_file(self, base_pattern, pattern):
        ''' find log from (base_pattern + pattern) file'''
        directories = glob.glob(base_pattern) 
        for directory in directories:
            log_files = f"{directory}/{pattern}"
            # print(f'log_files = {log_files}')
            return log_files

    def _follow_log_file(self, log_file_path):
        process = subprocess.Popen(['tail', '-F', log_file_path], stdout=subprocess.PIPE, text=True)
        print("Starting to follow the log file...")

        try:
            while True:
                line = self._read_log_line(process)
                if line:
                    self._handle_log_line(line)
        except Exception as e:
            print(f"Error following the log file: {e}")
        finally:
            process.terminate()
            print("Stopped following the log file.")

    def _read_log_line(self, process):
        ''' read log from xApp'''
        line = process.stdout.readline()
        return line.strip() if line else None

    def _handle_log_line(self, line):
        ''' find log with 'KpmReport' line '''
        try:
            log_data = json.loads(line)
            log_message = json.loads(log_data['log'])
            if 'KpmReport' in log_message['msg']:
                # print(log_message)
                self._collect_all_data(log_message)
        except json.JSONDecodeError:
            # Handle JSON decode errors if needed
            pass
        except Exception as e:
            print(f"Error processing log line: {e}")

    def _collect_all_data(self, log_message):
        if self.iperf_server.detect_streaming_restart():
            if(self.temp_epoch == self.method_scheduler.epoch):
                print("next_epoch start")
                self.method_scheduler.next_epoch()
            self.iperf_server.next_epoch_flag = False
        elif self.iperf_server.plot_flag_detect():
            if(self.temp_epoch != self.method_scheduler.epoch):
                print("plot_flag start")
                self.method_scheduler.store_plot()
                self.temp_epoch = self.method_scheduler.epoch
            self.iperf_server.plot_flag = False
        else:
            KpmReport_data = self._extract_kpm_report_data(log_message)
            # print(f"report_data = {KpmReport_data}")
            iperf_collected_data = self.iperf_server.collect_iperf_data()
            # print(f"total_collected_data = {iperf_collected_data}")
            if(self.method_scheduler.init_state_dont_store >= 2):
                self.method_scheduler.run(KpmReport_data, iperf_collected_data)
            self.method_scheduler.init_state_dont_store += 1
        
    def _extract_kpm_report_data(self, log_line):
        ''' extract data with dl_bytes '''
        try:
            self.dl_bytes_list = []
            print("Finding KpmReport in the log...")

            if 'KpmReport' in log_line['msg']:
                # print("KpmReport found, processing data...")
                ue_data_pattern = re.compile(r'ue\[(\d+)\]=\{([^}]+)\}')  # 匹配 UE 數據
                matches = ue_data_pattern.finditer(log_line['msg'])
                for match in matches:
                    ue_index = match.group(1)
                    ue_contents = match.group(2)
                    dl_bytes = re.search(r'dl_bytes=(\d+)', ue_contents)
                    dl_prbs = re.search(r'dl_prbs=(\d+)', ue_contents)
                    if dl_bytes and dl_prbs:
                        self.dl_bytes_list.append(int(dl_bytes.group(1)))
                        # print(f"UE[{ue_index}] dl_bytes: {dl_bytes.group(1)}, dl_prbs: {dl_prbs.group(1)}")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
        except KeyError as e:
            print(f"Key error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
            
        self.dl_bytes_list[0], self.dl_bytes_list[1] = self.dl_bytes_list[1], self.dl_bytes_list[0]
        
        return self.dl_bytes_list

    def log_file_thread(self):
        while True:
            latest_log_file = self._find_latest_log_file(self.base_pattern, self.file_pattern)
            # latest_log_file = "log_test.log"
            if latest_log_file:
                self._follow_log_file(latest_log_file)
            else:
                print("No log file found. Please check xApp")
                time.sleep(5)


# ### main function

# In[ ]:


if __name__ == '__main__': 
    
    xapp_log_reader = KpmReportLogReader()   
    
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    log_thread = threading.Thread(target=xapp_log_reader.log_file_thread)
    log_thread.start()
    
    @app.route('/A1message', methods=['POST'])
    def receive_message():
        ''' message = {}'''
        data = request.json
        message = data['message']
        # print(f'data = {message}')
        
        if(message == "restart streaming server"):
            print("restart streaming server")
            xapp_log_reader.iperf_server.next_epoch_flag = True
            return jsonify({"status": "Message received"}), 200
        
        if(message == "wait for packet end"):
            print("wait for packet end")
            xapp_log_reader.iperf_server.plot_flag = True
            return jsonify({"status": "Message received"}), 200

        # 提取數據部分
        parts = message.split(' ')
        total_data = int(parts[1][:-1])  # 移除 'K' 並轉換為整數
        send_netns = parts[3][:-1]  # ue1 or ue2
        duration = float(parts[5])  # 轉換為浮點數
        # print(f'total_data = {total_data}, send_netns = {send_netns}, duration = {duration}')

        if send_netns == "ue1":
            xapp_log_reader.iperf_server.iperf_temp_data_ue1.append((total_data, send_netns, duration))
            # print("iperf_temp_data_ue1")
        elif send_netns == "ue2":
            xapp_log_reader.iperf_server.iperf_temp_data_ue2.append((total_data, send_netns, duration))
            # print("iperf_temp_data_ue1")

        return jsonify({"status": "Message received"}), 200
    
    app.run(host="0.0.0.0", port=1212, debug=True, use_reloader=False)

