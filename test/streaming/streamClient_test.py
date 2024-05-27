from flask import Flask, request, jsonify
import threading
import time
import logging

app = Flask(__name__)

# 全局變數
collected_data_ue1 = []
collected_data_ue2 = []
last_calculate_time = time.time()

@app.route('/calculate', methods=['GET'])
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
        print(f"Overflow_data for Ue1 = {overflow_data_ue1}K, Overflow_duration for Ue1 = {overflow_duration_ue1} sec")
    
    total_duration_ue1 = 0
    total_collected_data_ue1 = 0
    collected_data_ue1 = []
    collected_data_ue1.append((overflow_data_ue1, "ue1", overflow_duration_ue1))

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
        print(f"Overflow_data for Ue2 = {overflow_data_ue2}K, Overflow_duration for Ue2 = {overflow_duration_ue2} sec")
        
    total_duration_ue2 = 0
    total_collected_data_ue2 = 0
    collected_data_ue2 = []
    collected_data_ue2.append((overflow_data_ue2, "ue2", overflow_duration_ue2))
    
    return ('', 204)

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

def run_other_tasks():
    # 運行其他任務的代碼
    pass

if __name__ == '__main__':
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)  # 只記錄錯誤信息
    
    # last_calculate_time = time.time()
    collection_thread = threading.Thread(target=collect_and_display)
    collection_thread.start()

    task_thread = threading.Thread(target=run_other_tasks)
    task_thread.start()

    app.run(port=1212, debug=True, use_reloader=False)