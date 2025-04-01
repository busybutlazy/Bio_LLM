import json
import torch
import random
import gc
import time
import multiprocessing

from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig,logging
from peft import PeftModel
from bert_score import score
from tqdm import tqdm
from datetime import datetime
from zoneinfo import ZoneInfo

logging.set_verbosity_error()

# ==== CONFIG ====
BASE_MODEL = "yentinglin/Llama-3-Taiwan-8B-Instruct"
LORA_PATH1 = "/app/outputs/dpo_bio_lora_output"
LORA_PATH2="/app/outputs/lora-bio-checkpoint-final"
VAL_PATH = "/app/datas/json/all_datas.jsonl"
MAX_NEW_TOKENS = 256
MAX_EVAL_SAMPLES=5
EPOCH=1
TIMEZONE="Asia/Taipei"

# ==== Load Model ====
def test(lora_path,data):
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
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()






    # ==== Load Data ====
    for i in range(EPOCH):
        if MAX_EVAL_SAMPLES:
            data = data[:MAX_EVAL_SAMPLES]


        predictions = []
        references = []

        print(f"🚀 Generating predictions...Epoch:{i+1}/{EPOCH}")

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

        
        # ==== (Optional) Save report ====
        nowtime=datetime.now(ZoneInfo(TIMEZONE)).strftime("%m%d%H%M")
        with open(f"/app/outputs/evaluate_result/eval_result{nowtime}.csv", "w", encoding="utf-8-sig") as f:
            f.write("instruction,input,reference,prediction\n")
            for i in range(len(data)):
                references[i]=references[i].replace('\n','\\n').replace('\\\\\\n','\\n')
                predictions[i]=predictions[i].replace('\n','\\n').replace('\\\\\\n','\\n')
                row = f'"{data[i]["instruction"]}","{data[i]["input"]}","{references[i]}","{predictions[i]}"\n'
                f.write(row)
        # print(f"\n📄 已儲存詳細結果至 /app/outputs/evaluate_result/eval_result{nowtime}.csv ✅")

        # ==== Compute BERTScore ====
        print("📏 Calculating BERTScore...")
        P, R, F1 = score(predictions, references, lang="zh", model_type="bert-base-chinese")

        print(f"\n✅ 評估結果 (平均):")
        print(f"Precision: {P.mean():.4f}")
        print(f"Recall   : {R.mean():.4f}")
        print(f"F1 Score : {F1.mean():.4f}")
    del model
    del tokenizer
    torch.cuda.empty_cache()  # important to actually free the GPU
    gc.collect()
    

    

    
if __name__=="__main__":
    print("📚 Loading validation data...")
    with open(VAL_PATH, "r", encoding="utf-8") as f:
        data = [json.loads(line.strip()) for line in f]
        random.shuffle(data)
    
    multiprocessing.set_start_method("spawn", force=True)

    p1=multiprocessing.Process(target=test,args=(LORA_PATH1,data))
    print(f"Start processing model:{LORA_PATH1}")
    p1.start()
    p1.join()

    time.sleep(5)
    
    p2=multiprocessing.Process(target=test,args=(LORA_PATH2,data))
    print(f"Start processing model:{LORA_PATH1}")
    p2.start()
    p2.join()
