# 使用Python 3.11的官方镜像（与本地开发环境一致，兼容 torch 2.5.1）
FROM python:3.11-slim

# 安装系统依赖（解决RDKit问题；bookworm 中 libgl1-mesa-glx 已拆分为 libgl1/libglx-mesa0）
RUN apt-get update && apt-get install -y \
    libxrender1 \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libgl1 \
    libglx-mesa0 \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 安装 Smina（分子对接引擎）：C++ 原生二进制，PyPI 没有对应 pip 包（pip install simna 会 404），
# 只能下载官方静态包放进镜像。下载失败也不让构建失败——对接页会自动降级提示。
RUN set -eux; \
    ( curl -fsSL -o /usr/local/bin/smina \
        https://sourceforge.net/projects/smina/files/smina.static/download \
      && chmod +x /usr/local/bin/smina \
      && /usr/local/bin/smina --version ) \
    || echo "⚠️ smina 下载失败：镜像仍可用，但分子对接/批量对接页会提示未检测到 Smina";

# 复制所有应用文件
COPY . .

# 暴露端口
EXPOSE 8501

# 启动命令
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]