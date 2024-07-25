import subprocess
import time
import random
import threading
from flask import Flask, jsonify, request
import requests
import re
import json

# Flask 應用程式設定
app = Flask(__name__)

# 目標伺服器的 URL
target_url = "http://0.0.0.0:1212/A1message"
server_epoch = 24
cell_max_ratio = 11*1024
video_max_ratio = 5*1024
first_time = 300
second_time = 600
third_time = 900

def send_post_to_next_epoch():

    # 发送消息到目标 URL  
    message = f"restart streaming server"
    payload = {"message": message}
    try:
        response = requests.post(target_url, json=payload)
        print(f"Sent: {payload}, Response: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

def send_post_to_wait():
    # 发送消息到目标 URL  
    message = f"wait for packet end"
    payload = {"message": message}
    try:
        response = requests.post(target_url, json=payload)
        print(f"Sent: {payload}, Response: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

def run_iperf(total_data, duration, server_ip, port, send_netns):
    if(send_netns == 'ue1'):
        command = f'iperf -c {server_ip} -p {port} -n {total_data}'
    else:
        bandwidth_per_sec = total_data*8
        command = f'iperf -c {server_ip} -p {port} -n {total_data} -u -b {bandwidth_per_sec}K'
    print(f"Transfer {total_data} to {send_netns}, duration {duration} sec")
    
    # 使用 subprocess.run 运行 iperf 并捕获输出
    # result = subprocess.run(command, shell=True, text=True, capture_output=True)
    # output = result.stdout
    # print(f"output = {output}")

    # # 解析丢包率
    # packet_loss_match = re.search(r'\((\d+\.\d+)%\)', output)
    # packet_loss_rate = packet_loss_match.group(1) if packet_loss_match else "No data found"

    # # 发送消息到目标 URL
    # message = f"Transfer {total_data} to {send_netns}, duration {duration} sec, Packet Loss Rate: {packet_loss_rate}%"
    # payload = {"message": message}
    
    # 使用 Popen 运行 iperf，不收集输出
    subprocess.Popen(command, shell=True, text=True)

    # 发送消息到目标 URL
    message = f"Transfer {total_data} to {send_netns}, duration {duration} sec"
    payload = {"message": message}
    try:
        response = requests.post(target_url, json=payload)
        print(f"Sent: {payload}, Response: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

# 随机生成的时间段和流量速率
# def generate_random_traffic_schedule(target_duration):
    
#     traffic_schedule = []
#     duration = 1
    
#     for _ in range(target_duration):
#         rate = random.randint(0, int((cell_max_ratio - video_max_ratio) * duration)) 
#         traffic_schedule.append((rate, duration))
        
#     save_traffic_schedule_to_json(traffic_schedule, "real-time")
    
#     return traffic_schedule

def generate_random_traffic_schedule():
    json_file = 'iperf_random_seed.json'
    traffic_schedule = []
    duration = 1
    # 讀取 JSON 文件
    with open(json_file, 'r') as file:
        data = json.load(file)

    # 存取 real-time 裡面的 rate 列表
    real_time_rate_list = data['real-time']['rate']
    
    for rate in real_time_rate_list:
        traffic_schedule.append((rate, duration))
    
    print(len(traffic_schedule))
    return traffic_schedule

def generate_static_traffic_schedule(target_duration):
    traffic_schedule = []
    time_duration = 0 
    duration = 1
    for _ in range(target_duration):
        time_duration += 1 

        if(time_duration > second_time):
            rate = video_max_ratio * duration
        else:        
            rate = video_max_ratio * duration  / 2
        # rate = random.uniform(0, max_rate)
        traffic_schedule.append((rate, duration))
    
    # save_traffic_schedule_to_json(traffic_schedule, "video")    
    print(len(traffic_schedule))
    return traffic_schedule

def save_traffic_schedule_to_json(traffic_schedule, user_equipment, filename='traffic_schedule.json'):
    # 将列表转换成字典形式，便于理解和阅读
    data = {
        user_equipment: {
            "rate": [rate for rate, _ in traffic_schedule],
            "duration": [duration for _, duration in traffic_schedule]
        }
    }
    with open(filename, 'a') as file:
        json.dump(data, file, indent=4)  # 使用 indent 参数美化输出

def run_traffic_schedule(traffic_schedule, server_ip, port, netns):
    for _ in range(server_epoch):
        for rate, duration in traffic_schedule:
            total_data = f"{int(rate * duration)}K"  # 根据速率和时间计算总数据量，转换为KB
            run_iperf(total_data, duration, server_ip, port, netns)
            # print(f"sleep for {duration} sec")
            time.sleep(duration)
        
        send_post_to_wait()
        
        for _ in range(first_time):
            run_iperf("0k", 1, server_ip, port, netns)
            time.sleep(1) 
            
        send_post_to_next_epoch()  
            
        
# --------------------------------------主程式-------------------------------------------

ue1_server_ip = '172.16.0.2'
ue2_server_ip = '172.16.0.3'
ue1_port = 8787 
ue2_port = 5487 
target_duration = third_time  # 定义需要多少组数据

# 生成随机的 traffic_schedule
ue1_traffic_schedule = generate_static_traffic_schedule(target_duration)
# ue2_traffic_schedule = generate_random_traffic_schedule(target_duration)
ue2_traffic_schedule = generate_random_traffic_schedule()
print(f"[ ID] Interval       Transfer     Bandwidth")

# 创建线程
thread_ue1 = threading.Thread(target=run_traffic_schedule, args=(ue1_traffic_schedule, ue1_server_ip, ue1_port, "ue1"))
thread_ue2 = threading.Thread(target=run_traffic_schedule, args=(ue2_traffic_schedule, ue2_server_ip, ue2_port, "ue2"))

# 启动线程
thread_ue1.start()
thread_ue2.start()

# Flask 应用程序路由
@app.route('/')
def home():
    return jsonify({"message": "Flask app is running and sending requests to /A1message based on traffic schedule"}), 200

# Flask 应用程序的执行函数
def run_flask_app():
    app.run(port=5000, debug=False)

# 启动 Flask 应用程序
flask_thread = threading.Thread(target=run_flask_app)
flask_thread.start()

# 等待线程完成
thread_ue1.join()
thread_ue2.join()
flask_thread.join()
