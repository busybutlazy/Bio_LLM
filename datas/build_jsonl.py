import json

import csv
import openai

from dotenv import load_dotenv
import os

import random
from pathlib import Path


# 載入 .env 檔案
load_dotenv()

# 使用環境變數
api_key = os.getenv('OPENAI_KEY')



client = openai.OpenAI(api_key=api_key)
# print(api_key)


def gpt_explain_simple(csv_path,target_jsonl_path,cloume='Input',start_line=0):
    counter = 0
    
    with open(csv_path,newline='')as csvfile:
        reader= csv.DictReader(csvfile)
        with open(target_jsonl_path,"a",encoding="utf-8")as f:
            for row in reader:
                counter +=1
                
                if counter<start_line:
                    continue
                
                try:
                    prompt = f"請用繁體中文提供「{row[cloume]}」的簡短定義，50~100字，需來自維基百科或權威資料。"
                    response = client.chat.completions.create(
                        model="gpt-4-turbo",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    answer = response.choices[0].message.content
                
                    jsonl_entry = {
                        "Instruction": "簡介專有名詞",
                        "Input": row[cloume],
                        "Category": "bio",
                        "Response": answer
                    }

                    f.write(json.dumps(jsonl_entry,ensure_ascii=False)+"\n")
                except Exception as e:
                    print(f"Error processing {row[cloume]}")
                
                print(f"已完成查詢{row[cloume]}")

def gpt_explain_detailed(sorce_jsonl_path,target_jsonl_path,start_line=0):
    counter = 0
    with open(sorce_jsonl_path, "r", encoding="utf-8") as sorce_f:
        with open(target_jsonl_path,"a",encoding="utf-8")as target_f:
            try:
                for line in sorce_f:
                    line_json=json.loads(line)
                    counter+=1
                    if counter<start_line:
                        continue
                    prompt = f"請以臺灣高中生物課本（108課綱）或教學角度介紹以下生物名詞，語言須使用繁體中文，內容符合高中教學用語與科學術語。請依以下格式撰寫：清楚條列定義、重要機制、類型或分類（如有）、並搭配具體實例說明。可參考臺灣常用教材（翰林、南一、康軒）或其他公開教學資源。不需開場白與結尾。生物名詞：{line_json['input']}。目前已知內容為：{line_json['output']}"
                    response = client.chat.completions.create(
                        model="gpt-4-turbo",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    answer = response.choices[0].message.content
                    jsonl_entry = {
                        "instruction": "詳細解釋名詞",
                        "input": line_json["input"],
                        "category": "bio",
                        "output": answer
                    }
                    target_f.write(json.dumps(jsonl_entry,ensure_ascii=False)+"\n")
                    
                    
                    print(f"已完成查詢{line_json['input']}")
                    # if counter==100:break
                    

            except Exception as e:
                print(f"Error processing {counter}")

def combine_all_jsonl(source_paths:list,target_path:str):
    """
    This function is only for merging multiple JSONL datasets into a single file.
    Do not perform any data processing here — the output file is not persistent and will be overwritten each time.

    Args:
        source_paths (list): A list of paths to the source JSONL files.
        target_path (str): The path of the output file to store the merged result.
    """
    
    with open(target_path,"w",encoding="utf-8")as target_f:
        for sorce_path in source_paths:
            with open(sorce_path,'r',encoding="utf-8") as sorce_f:
                try:
                    for line in sorce_f:
                        line_json=json.loads(line)
                        target_f.write(json.dumps(line_json,ensure_ascii=False)+"\n")

                except Exception as e:
                    print(f"Error processing {sorce_path}")


def creat_train_eval(source_paths:list,eval_ratio=0.2,shuffle=True):
    train_path=Path("/app/datas/train.jsonl")
    eval_path=Path("/app/datas/eval.jsonl")
    
    train_path.write_text('',encoding='utf-8')
    eval_path.write_text('',encoding='utf-8')

    
    for sorce_path in source_paths:
        with open(sorce_path,'r',encoding="utf-8") as sorce_f:
            lines=[]
            
            try:
                for line in sorce_f:
                    line_json=json.loads(line)
                    lines.append(line_json)


                if shuffle:
                    random.shuffle(lines)

                split_index = int(len(lines) * eval_ratio)
                eval_lines=lines[:split_index]
                train_lines=lines[split_index:]


                with open(train_path,'a',encoding="utf-8") as train_f:
                    for line in train_lines:
                        train_f.write(json.dumps(line,ensure_ascii=False)+"\n")

                with open(eval_path,'a',encoding="utf-8") as eval_f:
                    for line in eval_lines:
                        eval_f.write(json.dumps(line,ensure_ascii=False)+"\n")
                    
            except Exception as e:
                print(f"Error processing {sorce_path}")
    print(f"資料彙整完畢。\n train_data有{count_jsonl_lines(train_path)}筆資料。 \n eval_data有{count_jsonl_lines(eval_path)}筆資料。")
    

def count_jsonl_lines(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

csv_path=Path("/app/datas/bio_terms_csv.csv")
simple_path=Path("/app/datas/gpt_explain_simple.jsonl")

# gpt_explain_simple (csv_path,jsonl_path,start_line=445)

detailed_path=Path("/app/datas/gpt_explain_detailed.jsonl")

# gpt_explain_detailed(simple_path,detailed_path,start_line=101)

compare_path=Path("/app/datas/gpt_explain_compare.jsonl")

source_paths=[simple_path,detailed_path,compare_path]

combine_path=Path("/app/datas/all_datas.jsonl")
# This file is only for merging all datasets into one.  
# Avoid performing any processing here — the content is not saved and will reset every time.

# combine_all_jsonl(source_paths,combine_path)

creat_train_eval(source_paths)