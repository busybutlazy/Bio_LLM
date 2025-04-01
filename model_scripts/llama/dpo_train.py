from trl import DPOTrainer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from datasets import load_dataset
from peft import PeftConfig, PeftModel
import torch

# === 路徑設定 ===
BASE_MODEL = "yentinglin/Llama-3-Taiwan-8B-Instruct"
LORA_PATH = "/app/outputs/lora-bio-checkpoint-final"
DPO_DATA_PATH = "/app/datas/json/dpo_train_1.jsonl"
OUTPUT_DIR = "/app/outputs/dpo_bio_lora_output"
OFFLOAD_FOLDER = "/app/offload"

# === Tokenizer ===
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# === BitsAndBytes 設定 (4-bit 量化) ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# === 載入 LoRA 模型 ===
print("Loading LoRA base model...")
peft_config = PeftConfig.from_pretrained(LORA_PATH)

base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    device_map="auto",
    quantization_config=bnb_config,
    offload_folder=OFFLOAD_FOLDER,
)

model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH,
    is_trainable=True,  # ✅ 明確指定為訓練模式！
)
model.print_trainable_parameters()
model.train()

# === 載入資料集 ===
print("Loading dataset...")
dataset = load_dataset("json", data_files=DPO_DATA_PATH, split="train")

# === 訓練參數 ===
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,  # 省顯存
    num_train_epochs=1,
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=5e-6,
    output_dir=OUTPUT_DIR,
    bf16=False,
    fp16=True,
    report_to="none"
)

# === 建立 DPO Trainer ===
trainer = DPOTrainer(
    model=model,
    ref_model=None,  # 不使用 ref_model 可以省大量記憶體
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    beta=0.1,
    max_prompt_length=256,
    max_length=256,
    loss_type="sigmoid",
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
