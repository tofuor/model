import matplotlib.pyplot as plt
import numpy as np
import time

# 初始化圖表和軸對象
fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro-', animated=True)  # 使用紅色的點和線來繪制

def init():
    ax.set_xlim(0, 60)  # 初始 x 軸範圍
    ax.set_ylim(0, 500)  # y 軸範圍
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('KB per second')
    return ln,

def update(frame):
    # 每 5 秒添加 3 組數據
    t = frame * 5  # 模擬時間
    kbps = np.random.randint(100, 500, 3)  # 隨機生成 3 組數據
    
    for i, k in enumerate(kbps):
        xdata.append(t + i)  # 模擬每秒接收一次數據
        ydata.append(k)
        ln.set_data(xdata, ydata)
    
    if t >= ax.get_xlim()[1] - 20:  # 若接近當前 x 軸的上限
        ax.set_xlim(ax.get_xlim()[0] + 5, ax.get_xlim()[1] + 5)  # 向右移動 x 軸範圍
    
    return ln,

# 使用 FuncAnimation 創建動畫
from matplotlib.animation import FuncAnimation
ani = FuncAnimation(fig, update, frames=np.arange(1, 100), init_func=init, blit=True, interval=1000)  # 每 5 秒更新一次

plt.show()
