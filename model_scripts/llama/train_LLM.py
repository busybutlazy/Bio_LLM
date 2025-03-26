import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig,get_peft_model, TaskType, prepare_model_for_kbit_training
import torch
# import argparse





with open("/app/train_LLM_config.json","r")as f:
    config=json.load(f)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_threshold=6.0
)


tokenizer= AutoTokenizer.from_pretrained(config['base_model'], use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    config["base_model"],
    quantization_config=bnb_config,
    device_map="auto"
)

model=prepare_model_for_kbit_training(model)

peft_config=LoraConfig(
    r=config["lora_r"],
    lora_alpha=config["lora_alpha"],
    lora_dropout=config["lora_dropout"],
    task_type=TaskType.CAUSAL_LM
)

model= get_peft_model(model,peft_config)


def format_example(example):
    prompt=f"### 指令：{example['instruction']}\n### 輸入：{example['input']}\n### 回應："
    full_text = prompt + example["output"]
    return {"text":full_text}


train_dataset=load_dataset("json",data_files=config["train_file"])["train"].map(format_example)
eval_dataset =load_dataset("json",data_files=config["eval_file"])["train"].map(format_example)

def tokenize_function(examples):
    return tokenizer(examples["text"],truncation=True,padding="max_length",max_length=1024)

train_dataset=train_dataset.map(tokenize_function,batched=True)
eval_dataset=eval_dataset.map(tokenize_function,batched=True)

training_args = TrainingArguments(
    output_dir=config["output_dir"],
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    per_device_train_batch_size=config["per_device_train_batch_size"],
    per_device_eval_batch_size=config["per_device_eval_batch_size"],
    num_train_epochs=config["num_train_epochs"],
    learning_rate=config["learning_rate"],
    fp16=True,
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator
)

trainer.train()

model.save_pretrained("./lora-bio-checkpoint-final")
tokenizer.save_pretrained("./lora-bio-checkpoint-final")