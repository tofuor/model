import json
import pandas as pd
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import random
from collections import deque
from gym import spaces

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import csv
import os
import re
import numpy as np
import copy

# 環境參數
running_time = 184
# 超參數
EPISODES = 500
EPS_START = 0.9  # 隨機選擇行動的概率
EPS_END = 0
EPS_DECAY = EPISODES*0.25  # epsilon 衰減的速度
GAMMA = 0.98  # 折扣因子
LR = 0.001  # 學習率
MEMORY_SIZE = 10000  # 經驗回放的記憶體大小
BATCH_SIZE = 128  # 訓練批次的大小

# # 讀取Raw資料
def loadCsvFileToDict(filename, ueid):
    # 读取CSV文件
    data = pd.read_csv(filename)

    headers = data.columns.tolist()
    # # print("headers in cell csv = ",headers)

    data_dict = {header: [] for header in headers}

    if ueid != 'null':
        if isinstance(ueid, list):
            ueid_rows = data[data['UeID'].isin(ueid)].reset_index(drop=True)
        else:
            ueid_rows = data[data['UeID'] == ueid].reset_index(drop=True)
    else:
        ueid_rows = data
    
    for _, row in ueid_rows.iterrows():
        for header, value in row.items():
            if str(value).isdigit():
                data_dict[header].append(int(value))
            elif re.match(r'^[-+]?\d*\.?\d*$', str(value)):
                data_dict[header].append(float(value))
            elif re.match(r'^S.*$', str(value)):
                # # print(value)
                data_dict[header].append(int(value[1]))
            # else :
            #     # print(row)
            #     # print("loadCsvFileToDict have an unknown value : ", value)

    # Access the processed data
    # for header, values in data_dict.items():
    #     # print(header, values)

    ## print("example to show clown [Viavi.Geo.x] = ",data['Viavi.Geo.x'])
    # print("load " + filename + " successed")
    # print("************")
    return data_dict

# label handover的資料
def handoverDetect(Timestamp, ServingCellId):
    # print("****** function: handoverDetect ******")
    merged_list = [Timestamp, ServingCellId]

    # 创建与原始列表相同的副本列表
    handover_list = copy.deepcopy(merged_list)
    # 计算新副本列表中的每个元素与原始列表中前一个元素的差值
    for i in range(1, len(handover_list[1])):
        if(merged_list[1][i] == merged_list[1][i-1]) :
            handover_list[1][i] = 0
        else :
            handover_list[1][i] = 1
    handover_list[1][0] = 0
    handover_dict = {
        "Timestamp" : handover_list[0],
        "Handover" : handover_list[1],
    }
    
    # print("handover label finish")
    # print("************")
    return handover_dict

# Raw資料轉換，擷取time、cell_ID、cell_RSRP資料
def PrepareInput(ue_data):
    # print("****** function: PrepareInput ******")
    unmerge_data = {
        "Timestamp" : ue_data['Timestamp'],
        "ServingCellId": ue_data['ServingCellId'],
        "ServingCellRsrp": ue_data['ServingCellRsrp'],
        #"ServingCellRsSinr": ue_data['ServingCellRsSinr'],
        "neighbourCell1": ue_data['neighbourCell1'],
        "neighbourCell1Rsrp": ue_data['neighbourCell1Rsrp'],
        #"neighbourCell1RsSinr": ue_data['neighbourCell1RsSinr'],
        "neighbourCell2": ue_data['neighbourCell2'],
        "neighbourCell2Rsrp": ue_data['neighbourCell2Rsrp'],
        #"neighbourCell2RsSinr": ue_data['neighbourCell2RsSinr'],
        "neighbourCell3": ue_data['neighbourCell3'],
        "neighbourCell3Rsrp": ue_data['neighbourCell3Rsrp'],
        #"neighbourCell3RsSinr": ue_data['neighbourCell3RsSinr'],
        "neighbourCell4": ue_data['neighbourCell4'],
        "neighbourCell4Rsrp": ue_data['neighbourCell4Rsrp'],
        #"neighbourCell4RsSinr": ue_data['neighbourCell4RsSinr'],
        "neighbourCell5": ue_data['neighbourCell5'],
        "neighbourCell5Rsrp": ue_data['neighbourCell5Rsrp'],
        #"neighbourCell5RsSinr": ue_data['neighbourCell5RsSinr'],
    }
    unmerge_data_list = list(unmerge_data.items())
    # # print(unmerge_data_list)
    # # print(unmerge_data_list[1][0])
    # # print(unmerge_data_list[1][1])
    # # print(unmerge_data_list[1][1][1])
    serving_cell_id = unmerge_data_list[1][1]

    time = unmerge_data_list[0][1]
    cell_id = np.zeros((4000, 6))
    cell_id_index = [1,3,5,7,9,11]
    cell_rsrp = np.zeros((4000, 6))


    for time in range (len(time)):
        serving_id_temp = unmerge_data_list[1][1][time]
        cell_id[time][serving_id_temp - 1] = 1
        
        for cell_num in cell_id_index:
            cell_num_temp = unmerge_data_list[cell_num][1][time]
            cell_rsrp[time][cell_num_temp - 1] = unmerge_data_list[cell_num + 1][1][time]
            
    # # print(cell_id)

    # 處理重複點
    zero_points = np.argwhere(cell_rsrp == 0)
    # # print(zero_points)
    for zero_points_index in range(len(zero_points)-2):
        time = zero_points[zero_points_index][0]
        cell = zero_points[zero_points_index][1]
        # 前兩點的插值
        last_two_point_diff = cell_rsrp[time-2][cell] - cell_rsrp[time-1][cell]
        consider_last_two_point = cell_rsrp[time-1][cell] - last_two_point_diff
        # 後兩點的插值
        future_two_point_diff = cell_rsrp[time+2][cell] - cell_rsrp[time+1][cell]
        consider_future_two_point = cell_rsrp[time+1][cell] - future_two_point_diff
        # 隔壁鄰居的數值 12
        avg_neighbor = sum(cell_rsrp[time])/5
        if(abs(avg_neighbor - consider_last_two_point) < 12):
            cell_rsrp[time][cell] = consider_last_two_point
        if(abs(avg_neighbor - consider_future_two_point) < 12):
            cell_rsrp[time][cell] = consider_future_two_point
    
    # print("************")
    return time, cell_id, cell_rsrp, serving_cell_id

# 取得基站的放置位置
def getGnbLocation(cell_data):
    # print("****** function: getGnbLocation ******")
    cell_x = cell_data["Viavi.Geo.x"]
    cell_y = cell_data["Viavi.Geo.y"]
    cell_ngi = cell_data["Viavi.NrPci"]
    # 去除重复的点
    gnb_location_temp = list(set(zip(cell_x, cell_y, cell_ngi)))
    # 输出去重后的点
    # # print(gnb_location)
    gnb_location = [list(t) for t in gnb_location_temp]
    
    # print("successfully get gnb location")
    # print("************")
    return gnb_location

# model
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(6, 96)
        self.fc2 = nn.Linear(96, 192)
        self.fc3 = nn.Linear(192, 6)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# inference predict function
def predict_act(model, state):
    action_value = model.forward(state)  # state 放到 NN 裡
    top_indices = torch.topk(action_value, k=1)[1].flatten() # 從6個ouput中找最大值
    top_indices_np = top_indices.numpy()    # 轉成numpy
    action = top_indices_np[0]
    return action

#############################################
############### 開始Inference ###############
#############################################
# 載入NN及權重
def load_model(ue_num):
    model = DQN()
    model_num = f'model/ddqn_model_ue{ue_num + 1}.pth'
    model.load_state_dict(torch.load(model_num, map_location=torch.device('cpu')))
    return model

# 區域變數
action_store = [[0] for _ in range(9)]
time_store = [[0] for _ in range(9)]
handover_store = [[0] for _ in range(9)]
serving_cell = [[0] for _ in range(9)]
cell_one_round = [282, 184, 264, 184, 180, 115, 184, 184, 174]
ue_csv_file = 'output_ue.csv'

print("loading data and predict......")
# Inference
for ue_num in range(0,9) :
    model = load_model(ue_num)
    # 讀取CSV資料
    ue_data_temp = loadCsvFileToDict(ue_csv_file, ue_num + 1) # cell id 由1到9
    time_store[ue_num], _, cell_rsrp_temp, serving_cell[ue_num] = PrepareInput(ue_data_temp)
    # init
    action_store[ue_num][0] = serving_cell[ue_num][0]

    # 開始預測
    for i in range(len(cell_rsrp_temp[:cell_one_round[ue_num]])):
        state = cell_rsrp_temp[i]
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        # predicted_action = model.forward(state) 
        predicted_action = predict_act(model, state)
        # if(i ==0):
            # print("init state = ", predicted_action)
        action_store[ue_num].append(predicted_action + 1)

    # print(f"ue {ue_num + 1} finish predict")
    
    # 顯示結果
    handover_store[ue_num] = handoverDetect(time_store[ue_num], action_store[ue_num])
    # print(f"ue {ue_num + 1} model handover          =  {handover_store[ue_num]['Handover']}")
    # print(f"ue {ue_num + 1} model action            =  {action_store[ue_num]}")
    # print(f"ue {ue_num + 1} original serving_cell   =  {serving_cell[ue_num][:cell_one_round[ue_num]]}")
    
    # print("handover seconds             = ", np.nonzero(handover_store[ue_num]['Handover']))
print("predict finish")
# load csv data
ue_csv_file = 'output_ue.csv'
cell_csv_file = 'cells_05_15.csv'
# 指定ue_id
ue_location = []

for ue_id in range(1,10):
    ue_data = loadCsvFileToDict(ue_csv_file, ue_id)
    # # print("ue_data timestamps= ", len(ue_data['Timestamp']))
    # # print("ue_data columns = ", len(ue_data))
    # 9*3*4000的list
    handover_label = handoverDetect(ue_data['Timestamp'], ue_data['ServingCellId'])
    ue_location.append([ue_data['Timestamp'], ue_data['X'], ue_data['Y'], handover_label['Handover']])

cell_data = loadCsvFileToDict(cell_csv_file, 'null')
cell_location = getGnbLocation(cell_data)


import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import os
from IPython.display import clear_output

# DisplayPicture class
class DisplayPicture:
    def __init__(self, cell_location):
        # external parameter
        self.cell_location = cell_location
        self.collect_list = []
        # class generate
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.picture_num = 0

    def picture_setting(self):
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')

        # 設定x軸和y軸範圍
        self.ax.set_xlim(-500, 1100)
        self.ax.set_ylim(-600, 250)

        # 設定x軸和y軸格線間隔
        self.ax.set_xticks(np.arange(-500, 1100, 500))
        self.ax.set_yticks(np.arange(-600, 250, 200))

    def plot_cell(self):
        for point in self.cell_location:
            cell_x = point[0] 
            cell_y = point[1]
            self.ax.scatter(cell_x, cell_y, color='red')
            circle = Circle((cell_x, cell_y), 100, edgecolor='red', facecolor='none')
            self.ax.add_patch(circle)

    def display_part_and_save(self):
        clear_output(wait=True)
        
        self.picture_num = self.picture_num + 1
        self.addition_label()

        plt.grid()
        plt.show()
        # 設定檔名、位置，並儲存
        file_name = f"plot_picture{self.picture_num}.png"
        file_path = os.path.join("picture", file_name)
        self.fig.savefig(file_path, dpi=300)
        # 清除圖片
        # time.sleep(0.1)
    
    def pre_display_data(self, ue_num, handover_data, x, y, cell_one_round, color_map, first_time):
        A3_handover_num = 0
        model_handover_num = 0
        color = color_map[ue_num]
        if first_time:
            line, = self.ax.plot(x, y, color=color, label=f'Ue {ue_num}')
        else:
            line, = self.ax.plot(x, y, color=color)
        self.ax.scatter(x[0], y[0], color=line.get_color())  # 加入起始點
        # 添加handover label點
        for j in range(cell_one_round):
            if handover_data[j] == 1:
                self.ax.scatter(x[j], y[j], color='orange', marker='x', s=100)
                A3_handover_num = A3_handover_num + 1
            if handover_store[ue_num-1]['Handover'][j] == 1 and j > 2: # ue從1開始到9，所以要-1
                self.ax.scatter(x[j], y[j], facecolors='none', edgecolors='blue', marker='o', s=100)
                model_handover_num = model_handover_num + 1
        # print(f"ue {i + 1}, A3 handover number = {A3_handover_num}, model handover number = {model_handover_num}")
        self.collect_list.append([A3_handover_num, model_handover_num])
        
    def addition_label(self):
        legend_elements = [
            Line2D([0], [0], marker='x', color='orange', label='Handover', markerfacecolor='orange', markersize=10, linestyle='None'),
            Line2D([0], [0], marker='o', color='blue', label='model Handover', markerfacecolor='none', markersize=10, linestyle='None'),
        ]
        existing_handles, existing_labels = self.ax.get_legend_handles_labels()
        legend_elements += existing_handles
        labels = ['Handover', 'model Handover'] + existing_labels

        self.ax.legend(handles=legend_elements, labels=labels)
    
    def hanover_calculate(self):
        
        clear_output(wait=True)
        # 将数据分解为两个单独的列表
        heights, widths = zip(*self.collect_list)
        # y轴的位置
        y_values = np.arange(9)
        # 长条的宽度
        bar_width = 0.35
        # 创建一个新的figure对象
        fig, ax = plt.subplots()
        # 创建两个横条图
        bar1 = ax.barh(y_values + bar_width, widths, bar_width, label='model handover number')
        bar2 = ax.barh(y_values, heights, bar_width, label='A3 handover number')
        # 设置y轴的刻度和刻度标签
        ax.set_yticks(y_values + bar_width / 2)
        labels = ['ue' + str(i+1) for i in range(9)]
        ax.set_yticklabels(labels)
        # 添加图例
        ax.legend()
        # 显示图形
        plt.show()
        # 設定檔名、位置，並儲存
        file_name = f"plot_picture{self.picture_num}.png"
        file_path = os.path.join("picture", file_name)
        self.fig.savefig(file_path, dpi=300)
        
    def parameter_reset(self):
        self.collect_list = []

# Sub_DisplayPicture class
class Sub_DisplayPicture:
    def __init__(self, ue_location):
        self.ue_location = ue_location
        
    def divide_percent(self, x, precent):
        subset = []
        # 使用for循环按照百分比precent%进行分割
        for percentage in range(1, precent+1, 1):
            subset.append(int(x * percentage / precent))
        return subset

    def find_ue_location(self, ue_num, data_num):
        x = []
        y = []
        handover = []
        for k in range(data_num):
            x.append(ue_location[ue_num][1][k])
            y.append(ue_location[ue_num][2][k])
            handover.append(ue_location[ue_num][3][k])
        
        return x, y, handover


cell_one_round = [281, 184, 264, 184, 180, 113, 184, 184, 174]
ue = [1, 2, 3, 4, 5, 6, 7, 8, 9]
time_divisions = 100
color_map = {
    1: 'red', 
    2: 'green', 
    3: 'blue', 
    4: 'yellow', 
    5: 'purple', 
    6: 'orange', 
    7: 'brown', 
    8: 'pink', 
    9: 'cyan'
}

dp = DisplayPicture(cell_location)
sub_dp = Sub_DisplayPicture(ue_location)

dp.picture_setting()
dp.plot_cell()

print("exporting image......")
# 部分顯示，並存檔
for time_segment in range(time_divisions) :
    for ue_num in range(9):
        first_time = (time_segment == 0)
        part_cell_one_round = sub_dp.divide_percent(cell_one_round[ue_num], time_divisions)
        x, y, handover = sub_dp.find_ue_location(ue_num, part_cell_one_round[time_segment])
        dp.pre_display_data(ue[ue_num], handover, x ,y, part_cell_one_round[time_segment], color_map, first_time)
    dp.display_part_and_save()
    if (time_segment != time_divisions - 1):
        dp.parameter_reset()
    
heights, widths = zip(*dp.collect_list)
# y轴的位置
y_values = np.arange(9)
# 长条的宽度
bar_width = 0.35
# 创建一个新的figure对象
fig, ax = plt.subplots()
# 创建两个横条图
bar1 = ax.barh(y_values + bar_width, widths, bar_width, label='model-based handover number')
bar2 = ax.barh(y_values, heights, bar_width, label='A3-based handover number')
# 设置y轴的刻度和刻度标签
ax.set_yticks(y_values + bar_width / 2)
labels = ['ue' + str(i+1) for i in range(9)]
ax.set_yticklabels(labels)
# 添加图例
ax.legend()
# 显示图形
plt.show()
# 設定檔名、位置，並儲存
file_name = f"plot_picture101.png"
file_path = os.path.join("picture", file_name)
fig.savefig(file_path, dpi=300)



