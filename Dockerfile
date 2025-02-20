FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel

# 設定環境變數
ENV PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# 建立工作目錄
WORKDIR /app

# 安裝系統依賴 (修復 OpenCV 相關錯誤)
RUN update && apt install -y libgl1-mesa-glx\
	libglib2.0-0\

# 安裝額外的 Python 套件（若需要）
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# 預設執行指令
CMD ["bash"]
