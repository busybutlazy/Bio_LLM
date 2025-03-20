# FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel


# 設定 NVIDIA 環境變數，確保 Docker 可以正確存取 GPU
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 設定環境變數
ENV PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# 建立工作目錄
WORKDIR /app

# 安裝系統依賴 (修復 OpenCV 相關錯誤)
RUN apt update && apt install -y libgl1-mesa-glx\
	libglib2.0-0

# 安裝額外的 Python 套件（若需要）
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# 預設執行指令
CMD ["bash"]
