import json
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
from peft import PeftModel
from bert_score import score
from tqdm import tqdm
from datetime import datetime

# ==== CONFIG ====
BASE_MODEL = "yentinglin/Llama-3-Taiwan-8B-Instruct"
LORA_PATH = "/app/outputs/lora-bio-checkpoint-final"
VAL_PATH = "/app/datas/json/eval.jsonl"
MAX_NEW_TOKENS = 512
MAX_EVAL_SAMPLES=50


# ==== Load Model ====
print("🔧 Loading model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_threshold=6.0
)


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()







# ==== Load Data ====
print("📚 Loading validation data...")
with open(VAL_PATH, "r", encoding="utf-8") as f:
    data = [json.loads(line.strip()) for line in f]

random.shuffle(data)

data = data[:MAX_EVAL_SAMPLES]


predictions = []
references = []

print("🚀 Generating predictions...")
for item in tqdm(data):
    instruction = item["instruction"]
    input_text = item["input"]
    ref_output = item["output"]

    prompt = f"### 指令：{instruction}\n### 輸入：{input_text}\n### 回應："
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            top_p=0.95,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response = decoded[len(prompt):].strip()

    predictions.append(response)
    references.append(ref_output)

# ==== Compute BERTScore ====
print("📏 Calculating BERTScore...")
P, R, F1 = score(predictions, references, lang="zh", model_type="bert-base-chinese")

print(f"\n✅ 評估結果 (平均):")
print(f"Precision: {P.mean():.4f}")
print(f"Recall   : {R.mean():.4f}")
print(f"F1 Score : {F1.mean():.4f}")


nowtime=datetime.now().strftime("%m%d%H%M")
# ==== (Optional) Save report ====
with open(f"/app/outputs/evaluate_result/eval_result{nowtime}.csv", "w", encoding="utf-8") as f:
    f.write("instruction,input,reference,prediction,F1\n")
    for i in range(len(data)):
        row = f'"{data[i]["instruction"]}","{data[i]["input"]}","{references[i]}","{predictions[i]}",{F1[i].item():.4f}\n'
        f.write(row)

print(f"\n📄 已儲存詳細結果至 /app/outputs/evaluate_result/eval_result{nowtime}.csv ✅")