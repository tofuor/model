import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# 假設 ylog_enabled 是設定的一個變量，如果啟用則使用對數刻度
ylog_enabled = 1

if ylog_enabled:
    # 設置 y 軸為對數刻度，底數為 10
    plt.yscale('log', base=10)

    # 定義刻度的範圍
    ticks = [10**i for i in range(5, 10)]  # 從 10^5 到 10^9

    # 使用 FuncFormatter 來自定義刻度標籤
    # def custom_formatter(value, pos):
    #     if value < 10**6:
    #         return ""  # 對於小於 10^6 的值不顯示標籤
    #     else:
    #         return f"$10^{{{int(np.log10(value))}}}$"  # 顯示標籤

    # plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
    plt.gca().yaxis.set_major_locator(ticker.LogLocator(base=10))  # 設置主要刻度定位器

# 繪製示例數據
x = np.linspace(1.4, 1.5, 20)
z = np.linspace(0, 0, 50)
x = np.append(z,x)
print(x)
y = 10**(6 * x)
print(y)
z = np.linspace(1,len(y),70)
plt.plot(z, y)

# 顯示圖表
plt.savefig(f'test.png')