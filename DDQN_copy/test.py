# from PIL import Image
# import time

# # 循环遍历100张图像
# for i in range(1, 101):
#     # 图像的路径，假设它们按照"plot_picture1.png", "plot_picture2.png", ... 的格式命名
#     img_path = f"picture/plot_picture{i}.png"

#     # 打开图片
#     img = Image.open(img_path)

#     # 显示图片
#     img.show()

#     # 持续一秒
#     time.sleep(0.5)

#     # 关闭显示窗口
#     # img.close()
    
from flask import Flask, render_template
import os

app = Flask(__name__)

@app.route('/')
def index():
    image_names = ['plot_picture'+str(i)+'.png' for i in range(1, 101)]
    return render_template('index.html', image_names=image_names)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)