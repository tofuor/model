import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import random
from collections import deque


# 環境參數
running_time = 282
# 超參數
EPISODES = 500
EPS_START = 0.9  # 隨機選擇行動的概率
EPS_END = 0.05
EPS_DECAY = 200  # epsilon 衰減的速度
GAMMA = 0.8  # 折扣因子
LR = 0.001  # 學習率
MEMORY_SIZE = 10000  # 經驗回放的記憶體大小
BATCH_SIZE = 128  # 訓練批次的大小

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 讀取Raw資料
def loadCsvFileToDict(filename, ueid):
    # 读取CSV文件
    data = pd.read_csv(filename)

    headers = data.columns.tolist()
    # print("headers in cell csv = ",headers)

    data_dict = {header: [] for header in headers}

    if ueid != 'null' :
        ueid_rows = data[data['UeID'] == ueid].reset_index(drop=True)
    else :
        ueid_rows = data
    
    for _, row in ueid_rows.iterrows():
        for header, value in row.items():
            if str(value).isdigit():
                data_dict[header].append(int(value))
            elif re.match(r'^[-+]?\d*\.?\d*$', str(value)):
                data_dict[header].append(float(value))
            elif re.match(r'^S.*$', str(value)):
                # print(value)
                data_dict[header].append(int(value[1]))
            # else :
            #     print(row)
            #     print("loadCsvFileToDict have an unknown value : ", value)

    # Access the processed data
    # for header, values in data_dict.items():
    #     print(header, values)

    #print("example to show clown [Viavi.Geo.x] = ",data['Viavi.Geo.x'])
    print("load " + filename + " successed")
    return data_dict
# label handover的資料
def handoverDetect(Timestamp, ServingCellId):
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
    
    print("handover label finish")
    return handover_dict
# Raw資料轉換，擷取time、cellID、cell RSRP資料
def PrepareInput(ue_data):
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
    print(unmerge_data_list[1][0])
    print(unmerge_data_list[1][1])
    # print(unmerge_data_list[1][1][1])

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
            
    # print(cell_id)

    # 處理重複點
    zero_points = np.argwhere(cell_rsrp == 0)
    # print(zero_points)
    for zero_points_index in range(len(zero_points)):
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
    
    return time, cell_id, cell_rsrp
# 自訂環境
class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    def __init__(self, max_steps=10):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.max_steps = max_steps
        self.current_step = 0

        # 行動空間有兩種行動：向左和向右
        self.action_space = spaces.Discrete(2)

        # 觀察空間是當前的位置，範圍在 -max_steps 到 max_steps 之間
        self.observation_space = spaces.Box(low=-self.max_steps, high=self.max_steps, shape=(1,), dtype=np.float32)

    def step(self, action):
        # 行動0表示向左，行動1表示向右
        if action == 0:
            self.current_step -= 1
        elif action == 1:
            self.current_step += 1
        else:
            raise ValueError("Invalid action")

        done = self.current_step >= self.max_steps or self.current_step <= -self.max_steps

        # 報酬為當前步數的負數（我們的目標是儘量保持在原點附近）
        reward = -abs(self.current_step)
#zz如下 reward的
#zz        if(last serving cell != now serving cell)
#zz            reward = max(RSRP) - serving cell RSRP - handover punish
#zz        else
#zz            reward = max(RSRP) - serving cell RSRP

        # 返回當前觀察，報酬，是否結束，以及額外的訊息
        return np.array([self.current_step]).astype(np.float32), reward, done, {}

    def reset(self):
        # 重置環境，把當前步數設為0
#zz        self.current_step = init_cell_rsrp
        return np.array([self.current_step]).astype(np.float32)

    def render(self, mode='human'):
        # 環境的渲染，這裡簡單地打印出當前步數
        print(f"Current step: {self.current_step}")
# model
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(4, 24)
        self.fc2 = nn.Linear(24, 48)
        self.fc3 = nn.Linear(48, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# target model
class DoubleDQNAgent:
    def __init__(self):
        self.model = DQN().to(device)
        self.target_model = DQN().to(device)
        self.update_target_model()
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.optimizer = optim.Adam(self.model.parameters(), LR)
        self.steps_done = 0

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            np.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if np.random.rand() <= eps_threshold:
            return random.randrange(2)
#zz         return 三個最近的隨機
        else:
            return torch.argmax(self.model(Variable(state))).item()
#zz         return 最大值的cell

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in batch:
            state = Variable(state).to(device)
            next_state = Variable(next_state).to(device)
            reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0).to(device)
            done = torch.from_numpy(np.array([done], dtype=np.int32)).unsqueeze(0).to(device)

            q_values = self.model(state)
            next_q_values = self.model(next_state)
            next_q_state_values = self.target_model(next_state)

            q_value = q_values.gather(1, torch.LongTensor([[action]]).to(device))
            next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1))
            expected_q_value = reward + GAMMA * next_q_value * (1 - done)

            loss = nn.MSELoss()(q_value, expected_q_value.detach())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# 資料處理
ue_csv_file = 'output_ue.csv'
ue_data = loadCsvFileToDict(ue_csv_file, 1)
handover_label = handoverDetect(ue_data['Timestamp'], ue_data['ServingCellId'])
time, cell_id, cell_rsrp = PrepareInput(ue_data)

# 必要資料
init_cell_rsrp = cell_rsrp[0]


# 引入環境
env = gym.make('CartPole-v0')
agent = DoubleDQNAgent()

# 開始訓練
for e in range(EPISODES):
    state = env.reset()
    # print(type(state))  # 印出state的類型
    # print(state)  # 印出state的值
    state = torch.tensor(state[0], dtype=torch.float32).unsqueeze(0).to(device)
    print(state)  # 印出state的值
    for time_t in range(running_time):
        action = agent.act(state)
        # step_result = env.step(action)
        # print(step_result)

        next_state, reward, done, _, _ = env.step(action)
        reward = reward if not done else -10
        next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(device)
        agent.memorize(state, action, reward, next_state, done)
        state = next_state
        agent.replay()
        if done:
            print(f"結束於 {time_t+1} 秒，於第 {e+1} 輪")
            break
    if e % 10 == 0:
        agent.update_target_model()
        
# ------------------------------------------------ #
# ------------------------------------------------ #
# ------------------------------------------------ #
env = CustomEnv()
observation = env.reset()
for _ in range(10):
    action = env.action_space.sample() # 隨機選擇一個行動
    observation, reward, done, info = env.step(action)
    env.render()
    if done:
        break
# ------------------------------------------------ #
# ------------------------------------------------ #
# ------------------------------------------------ #
