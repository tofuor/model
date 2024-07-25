# import vlc
# import yt_dlp

# def get_live_url(youtube_url):
#     ydl_opts = {
#         'format': 'best',  # 選擇最好的品質
#         'quiet': True,     # 運行時不輸出訊息
#         'force_generic_extractor': True,
#     }

#     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#         info_dict = ydl.extract_info(youtube_url, download=False)
#         video_url = info_dict.get('url', None)  # 取得直播串流的URL
#         print(video_url)
#         return video_url

# def play_stream(url):
#     player = vlc.MediaPlayer(url)
#     player.play()
#     try:
#         input("按下 Enter 鍵退出並停止播放...\n")
#     finally:
#         player.stop()

# # 例子: 替換成您要的YouTube直播網址
# youtube_url = "https://www.youtube.com/watch?v=LuQ4S-i5zoE"
# live_url = get_live_url(youtube_url)
# print("直播URL:", live_url)

# if live_url:
#     print("正在嘗試播放直播...")
#     play_stream(live_url)
# else:
#     print("無法獲取直播URL")

import yt_dlp
import vlc
import psutil
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pyshark
import threading
import time
import os

def get_live_url(youtube_url):
    ydl_opts = {
        'format': 'bestvideo',
        'quiet': True,
        'force_generic_extractor': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        video_url = info_dict.get('url', None)
        return video_url

def play_stream(url):
    # 設置 VLC 使其不輸出音訊
    instance = vlc.Instance('--no-video', '--aout=none')  # 添加 '--aout=none' 以禁用音訊
    player = instance.media_player_new()
    media = instance.media_new(url)
    player.set_media(media)
    player.play()
    try:
        input("按下 Enter 鍵退出並停止播放...\n")
    finally:
        player.stop()


def find_pid_by_name(process_name):
    "Given a process name, return a list of psutil.Process objects."
    ls = []
    for proc in psutil.process_iter(['pid', 'name']):
        if process_name.lower() in proc.info['name'].lower():
            ls.append(proc.pid)
    return ls

def capture_traffic(pid, interface, fig, ax, duration=60):
    capture = pyshark.LiveCapture(interface=interface)
    start_time = time.time()
    times = []
    byte_counts = []

    for packet in capture.sniff_continuously():
        if 'ip' in packet:
            try:
                if packet.ip.src in pid or packet.ip.dst in pid:
                    byte_count = int(packet.length)
                    current_time = time.time() - start_time
                    byte_counts.append(byte_count)
                    times.append(current_time)

                    ax.clear()
                    ax.plot(times, byte_counts, label='Bytes over time')
                    ax.legend()
                    ax.set_title('Network Traffic')
                    ax.set_xlabel('Time (seconds)')
                    ax.set_ylabel('Bytes')

                    plt.pause(0.01)  # 更新圖表

                    if current_time > duration:
                        break
            except AttributeError:
                continue

def monitor_network_traffic(pid, interface):
    fig, ax = plt.subplots()
    traffic_thread = threading.Thread(target=capture_traffic, args=(pid, interface, fig, ax))
    traffic_thread.start()
    plt.show()
    traffic_thread.join()

# 主要執行流程
youtube_url = "https://www.youtube.com/watch?v=LuQ4S-i5zoE"
interface = 'enp4s2'

live_url = get_live_url(youtube_url)
if live_url:
    print("Direct live URL:", live_url)
    player = play_stream(live_url)
    
    # 請先手動確認 yt-dlp 的進程名稱
    yt_dlp_processes = find_pid_by_name('yt-dlp')
    if yt_dlp_processes:
        monitor_network_traffic(yt_dlp_processes[0], interface)
    else:
        print("No yt-dlp process found.")
    
    input("Press Enter to exit and stop playing...\n")
    player.stop()
else:
    print("Could not retrieve live URL")


