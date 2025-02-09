from transformers import VisionEncoderDecoderModel, DonutProcessor
from PIL import Image
import torch

# 載入 Donut 模型與處理器
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

# 設定模型運行於 GPU（如果可用）
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 讀取圖像並進行預處理
image = Image.open("/home/busybutlazy/Bio_LLM/datas/imgs/img1.png").convert("RGB")
prompt = "<chart>請解析這張圖像，並輸出結構化資料：</chart>"

inputs = processor(image,prompt, return_tensors="pt",legacy=False).to(device)

# 模型生成解析結果
outputs = model.generate(**inputs)
parsed_output = processor.decode(outputs[0], skip_special_tokens=True)

print("解析結果：", parsed_output)
