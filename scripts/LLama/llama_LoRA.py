import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

model_name = "yentinglin/Llama-3-Taiwan-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Use 4-bit quantization
    bnb_4bit_compute_dtype=torch.float16,  # Reduce precision for efficiency
    bnb_4bit_use_double_quant=True,  # Further optimize VRAM usage
)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"  # Automatically use the best device (GPU if available)
)