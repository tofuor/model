# 使用官方 Python 镜像作为基础镜像
FROM pytorch/pytorch:latest

# 将工作目录设置为 /app
WORKDIR /app
RUN mkdir /picture

# 将当前目录的内容复制到工作目录中
COPY . /app

# 安装项目依赖包
RUN pip install --no-cache-dir -r requirements.txt

# 运行你的应用
CMD ["python", "main.py"]
