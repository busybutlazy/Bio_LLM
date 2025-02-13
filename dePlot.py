from transformers import AutoProcessor, Pix2StructForConditionalGeneration
from PIL import Image
import torch

# 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 載入處理器和模型
processor = AutoProcessor.from_pretrained("google/deplot")
model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot")
model.to(device)
model.eval()

# url = "https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/5090.png"

# 讀取和處理圖像
image_path = "/workspaces/Bio_LLM/datas/imgs/img2.png"
image = Image.open(image_path).convert("RGB")
# image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, text="Describe the information of this chart.", return_tensors="pt").to(device)
predictions = model.generate(**inputs, max_new_tokens=512)


# # 解碼輸出結果
# output_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
# print("原始輸出張量：", outputs)
# print("模型描述結果：", output_text)
print(processor.decode(predictions[0], skip_special_tokens=True))
