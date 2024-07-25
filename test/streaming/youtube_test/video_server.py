import requests
import time
import yt_dlp
import vlc
import threading


def get_stream_url(video_url):
    ydl_opts = {
        'format': '134/18',  # 或其他適合你需求的格式
        'noplaylist': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=False)
        # print(f"info_dict = {info_dict}")
        return info_dict['url']  # 返回最適合播放的串流 URL
        
def play_video(stream_url, play_finished):
    player = vlc.MediaPlayer(stream_url)
    player.set_rate(10) 
    player.play()
    
    while True:
        time.sleep(0.1)  # 每0.1秒檢查一次
        state = player.get_state()
        # 檢查影片是否播放完畢或停止
        if state == vlc.State.Ended or state == vlc.State.Stopped:
            play_finished.append(True)  # 使用列表作為一個可變對象來傳遞狀態
            break
    player.stop()
    
def download_and_play(video_url):
    stream_url = get_stream_url(video_url)
    play_finished = []  # 用於通知下載循環結束的標記

    # 使用 threading 啟動播放
    play_thread = threading.Thread(target=play_video, args=(stream_url,play_finished))
    play_thread.start()

    # 初始化計時器和數據計數器
    start_time = time.time()
    last_time = start_time
    data_downloaded = 0

    # 使用 requests 進行串流
    with requests.get(stream_url, stream=True) as r:
        # total_data = r.text
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=8192*4):  # 1 MB 每個數據塊
            if play_finished:
                break  # 如果播放結束，停止下載
            current_time = time.time()
            # 累計下載的數據量、時間
            data_downloaded += len(chunk)
            # 每次chunk的數據量、時間
            data_per_time = len(chunk)
            period_per_time = current_time - last_time

            # 計算經過的時間和下載速度
            if period_per_time > 0:
                speed = data_per_time / period_per_time  # bytes per second
                print(f"data_download: {data_per_time} bytes, elapsed_time = {period_per_time} s, Current streaming speed: {speed} bytes/sec")

            last_time = current_time
            # 處理數據塊（可選，視你的需要）
            # process_data(chunk)  # 假設的數據處理函數
    
    # data_downloaded = len(chunk)
    # print(f"total_data = {total_data}")
    current_time = time.time()
        # 累計下載的數據量、時間
    elapsed_time = current_time - start_time
    speed = data_downloaded / elapsed_time  # bytes per second
    print(f"total_data_download: {data_downloaded} bytes, total_elapsed_time = {elapsed_time} s, total streaming speed: {speed} bytes/sec")

    play_thread.join()  # 等待播放線程結束

# -----------------------------------------------------------------------------------------
video_url = "https://www.youtube.com/watch?v=JPkXxoxpoXY"
download_and_play(video_url)