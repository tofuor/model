import subprocess
import time
import random
import threading

def run_iperf(total_data, duration, server_ip, port):
    # 使用 iperf 发送数据并捕获输出
    command = f'iperf -c {server_ip} -p {port} -n {total_data}'
    
    if(port == 8787):
        send_netns = "ue1"
    elif(port == 5487):
        send_netns = "ue2"
    print(f"Transfer {total_data} to {send_netns}, duration {duration} sec")
    result = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # 解析输出
    for line in result.stdout:
        if 'bits/sec' in line:
            print(line.strip())
            
# 随机生成的时间段和流量速率
def generate_random_traffic_schedule(num_entries):
    traffic_schedule = []
    for _ in range(num_entries):
        duration = round(random.uniform(0, 2), 1)  # duration 范围是 0 到 2 秒
        max_rate = 1500 * duration  # rate 范围是 0 到 duration * 1.5
        rate = random.uniform(0, max_rate)
        traffic_schedule.append((rate, duration))
    return traffic_schedule

def run_traffic_schedule(traffic_schedule, server_ip, port):
    for rate, duration in traffic_schedule:
        total_data = f"{int(rate * duration)}K"  # 根据速率和时间计算总数据量，转换为KB
        run_iperf(total_data, duration, server_ip, port)
        time.sleep(duration)

# ---------------------------------------------------------------------------------

ue2_server_ip = '140.118.122.151'
ue1_server_ip = '140.118.122.151'
# ue1_server_ip = '172.16.0.3' 
ue1_port = 8787 
# ue2_server_ip = '172.16.0.2'
ue2_port = 5487 
num_entries = 5  # 定义需要多少组数据

# 生成随机的 traffic_schedule
ue1_traffic_schedule = generate_random_traffic_schedule(num_entries)
ue2_traffic_schedule = generate_random_traffic_schedule(num_entries)
print(f"[ ID] Interval       Transfer     Bandwidth")

# 创建线程
thread_ue1 = threading.Thread(target=run_traffic_schedule, args=(ue1_traffic_schedule, ue1_server_ip, ue1_port))
thread_ue2 = threading.Thread(target=run_traffic_schedule, args=(ue2_traffic_schedule, ue2_server_ip, ue2_port))

# 启动线程
thread_ue1.start()
thread_ue2.start()

# 等待线程完成
thread_ue1.join()
thread_ue2.join()