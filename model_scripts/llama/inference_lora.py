from transformers import AutoModelForCausalLM ,AutoTokenizer,BitsAndBytesConfig
from peft import PeftModel
import torch

tokenizer = AutoTokenizer.from_pretrained("yentinglin/Llama-3-Taiwan-8B-Instruct", use_fast=True)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_threshold=6.0
)
base_model = AutoModelForCausalLM.from_pretrained(
    "yentinglin/Llama-3-Taiwan-8B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(base_model, "./lora-bio-checkpoint-final")
model.eval()

# 📝 Test input
instruction = "詳細解釋名詞"
input_text = "天擇說"

prompt = f"### 指令：{instruction}\n### 輸入：{input_text}\n### 回應："

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 🔮 Generate output
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n🔍 回應：\n", response[len(prompt):].strip())